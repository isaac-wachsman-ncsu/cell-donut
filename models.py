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
        self.output_layer = nn.Linear(in_dim, self.dim)

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
