"""Neural-network velocity fields.

This module defines the parametric velocity field
:math:`v_\\phi : \\mathbb{R}^D \\to \\mathbb{R}^D` that the library learns.

The field is *autonomous*: it depends only on the spatial coordinate ``x`` and
not on time. The cyclic dynamics emerge from integrating the same stationary
field forward in time (see :mod:`velocity_ot.dynamics`).
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.init as init

# Registry of supported point-wise activation functions.
ACTIVATION_FN: dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class VelocityNet(nn.Module):
    """A flexible MLP parameterising an autonomous velocity field.

    The network maps a batch of ``D``-dimensional coordinates to a batch of
    ``D``-dimensional velocity vectors, :math:`x \\mapsto v_\\phi(x)`. It works
    for an arbitrary input dimension ``D`` and can be evaluated at
    out-of-sample points simply by calling :meth:`forward`.

    Args:
        dim: Dimensionality ``D`` of the ambient Euclidean space. The input and
            output of the network both have this dimension.
        hidden_dims: Width of each hidden layer. The number of hidden layers is
            ``len(hidden_dims)``.
        activation: Name of the activation function to use between layers. One
            of the keys of :data:`ACTIVATION_FN` (e.g. ``"silu"`` or
            ``"gelu"``).
        layer_norm: If ``True`` a :class:`torch.nn.LayerNorm` is applied after
            every hidden linear layer (before the activation).
        residual: If ``True`` residual connections are added around hidden
            layers of equal width. Layers whose input and output widths differ
            fall back to a plain feed-forward connection.

    Attributes:
        dim: The ambient dimension ``D``.
    """

    def __init__(
        self,
        dim: int,
        hidden_dims: Sequence[int] = (128, 128, 128),
        activation: str = "silu",
        layer_norm: bool = True,
        residual: bool = False,
        out_dim: int | None = None,
    ) -> None:
        super().__init__()

        if dim <= 0:
            raise ValueError(f"`dim` must be a positive integer, got {dim}.")
        if activation not in ACTIVATION_FN:
            raise ValueError(
                f"Activation '{activation}' not recognised. "
                f"Choose one of {sorted(ACTIVATION_FN)}."
            )

        self.dim = int(dim)
        self.residual = bool(residual)
        act_fn = ACTIVATION_FN[activation]

        # Build the stack of hidden blocks: Linear -> (LayerNorm) -> Activation.
        blocks: list[nn.Module] = []
        widths: list[int] = []
        in_dim = self.dim
        for h in hidden_dims:
            block_layers: list[nn.Module] = [nn.Linear(in_dim, h)]
            if layer_norm:
                block_layers.append(nn.LayerNorm(h))
            block_layers.append(act_fn())
            blocks.append(nn.Sequential(*block_layers))
            widths.append((in_dim, h))
            in_dim = h

        self.hidden_blocks = nn.ModuleList(blocks)
        self._hidden_widths = widths
        # The read-out is linear (no activation): a velocity can take any sign.
        # ``out_dim`` defaults to ``dim`` (a full velocity); set to 1 for a
        # scalar read-out (e.g. the magnitude sub-net of a factored field).
        self.out_dim = int(dim if out_dim is None else out_dim)
        self.output_layer = nn.Linear(in_dim, self.out_dim)

        self._initialise_weights()

    def _initialise_weights(self) -> None:
        """Xavier-uniform initialisation for all linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the velocity field at ``x``.

        Args:
            x: Coordinates of shape ``[B, D]``.

        Returns:
            Velocity vectors of shape ``[B, D]``.
        """
        if x.dim() != 2:
            raise ValueError(f"Expected input of shape [B, D], got {tuple(x.shape)}.")
        if x.shape[1] != self.dim:
            raise ValueError(
                f"Input dimension {x.shape[1]} does not match network dim {self.dim}."
            )

        h = x
        for (in_w, out_w), block in zip(self._hidden_widths, self.hidden_blocks):
            out = block(h)
            if self.residual and in_w == out_w:
                h = h + out
            else:
                h = out
        return self.output_layer(h)


class FactoredVelocityField(nn.Module):
    r"""Velocity field factored into a direction and a magnitude network.

    The field is written :math:`v_\phi(x) = m_\psi(x)\,\hat d_\omega(x)`, where

    * :math:`\hat d_\omega : \mathbb{R}^D \to \mathbb{R}^D` is a **direction**
      network whose output is normalised to unit length, and
    * :math:`m_\psi : \mathbb{R}^D \to \mathbb{R}_{\ge 0}` is a **magnitude**
      (speed) network with a non-negative (softplus) read-out.

    Because :meth:`forward` still maps ``[B, D] -> [B, D]``, this module is a
    drop-in replacement for :class:`VelocityNet`: the integrator, ``predict``
    and every loss are unchanged. Separating speed from direction lets each
    sub-problem be trained on its own losses and in its own stage (freeze one
    network, train the other via :meth:`set_trainable`).

    Args:
        dim: Ambient dimension ``D``.
        direction_hidden_dims: Hidden widths of the direction network.
        magnitude_hidden_dims: Hidden widths of the magnitude network.
        activation, layer_norm, residual: Passed to both sub-networks.
        eps: Floor on the direction norm before normalising (stability).
    """

    def __init__(
        self,
        dim: int,
        direction_hidden_dims: Sequence[int] = (128, 128, 128),
        magnitude_hidden_dims: Sequence[int] = (128, 128, 128),
        activation: str = "silu",
        layer_norm: bool = True,
        residual: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.direction_net = VelocityNet(
            dim, direction_hidden_dims, activation, layer_norm, residual
        )
        self.magnitude_net = VelocityNet(
            dim, magnitude_hidden_dims, activation, layer_norm, residual, out_dim=1
        )

    def direction(self, x: torch.Tensor) -> torch.Tensor:
        """Unit-norm direction ``[B, D]``."""
        r = self.direction_net(x)
        return r / r.norm(dim=-1, keepdim=True).clamp_min(self.eps)

    def magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Non-negative speed ``[B, 1]``."""
        return nn.functional.softplus(self.magnitude_net(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Velocity ``v(x) = magnitude(x) * direction(x)`` of shape ``[B, D]``."""
        return self.magnitude(x) * self.direction(x)

    def set_trainable(self, direction: bool = True, magnitude: bool = True) -> None:
        """Freeze/unfreeze each sub-network (for per-network staging)."""
        for p in self.direction_net.parameters():
            p.requires_grad_(direction)
        for p in self.magnitude_net.parameters():
            p.requires_grad_(magnitude)
