"""Loss terms for optimal-transport velocity-field learning.

The composite training objective is

.. math::

    \\mathcal{L} = \\lambda_1 \\mathcal{L}_{KE}
                 + \\lambda_2 \\mathcal{L}_{OT\\_global}
                 + \\lambda_3 \\mathcal{L}_{align}
                 + \\lambda_4 \\mathcal{L}_{OT\\_sub}

with the four terms defined below. All optimal-transport terms use the
``POT`` library through its PyTorch backend so gradients propagate to the
network parameters.

Numerical note
--------------
Entropic OT computed in the primal domain underflows for small regularisation
relative to the cost scale (the classic Sinkhorn instability). We therefore
(a) run the log-domain solver (``method="sinkhorn_log"``, mathematically the
same problem as :func:`ot.sinkhorn`) and (b) optionally rescale the cost matrix
so ``reg`` acts on an ``O(1)`` scale. :func:`ot.bregman.sinkhorn_epsilon_scaling`
is an alternative stabilisation strategy for the same objective.
"""

from __future__ import annotations

import ot
import torch


def _uniform_weights(n: int, ref: torch.Tensor) -> torch.Tensor:
    """Return a uniform histogram of length ``n`` matching ``ref``'s device/dtype."""
    return torch.full((n,), 1.0 / n, device=ref.device, dtype=ref.dtype)


def sinkhorn_divergence(
    x: torch.Tensor,
    y: torch.Tensor,
    reg: float = 0.05,
    a: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    p: int = 2,
    normalize_cost: bool = True,
    debias: bool = True,
    n_iter: int = 200,
) -> torch.Tensor:
    """Entropic Sinkhorn divergence between two point clouds.

    Computes the debiased Sinkhorn divergence

    .. math::

        S_\\varepsilon(x, y) = \\mathrm{OT}_\\varepsilon(x, y)
            - \\tfrac{1}{2}\\mathrm{OT}_\\varepsilon(x, x)
            - \\tfrac{1}{2}\\mathrm{OT}_\\varepsilon(y, y),

    where :math:`\\mathrm{OT}_\\varepsilon` is the entropy-regularised optimal
    transport cost for the ``p``-th power Euclidean ground cost. The debiasing
    makes :math:`S_\\varepsilon(x, x) = 0` and removes the entropic bias, giving
    a well-behaved distributional discrepancy.

    Args:
        x: Source samples of shape ``[N, D]``.
        y: Target samples of shape ``[M, D]``.
        reg: Entropic regularisation strength :math:`\\varepsilon`.
        a: Optional source histogram ``[N]``. Defaults to uniform.
        b: Optional target histogram ``[M]``. Defaults to uniform.
        p: Ground-cost exponent (``2`` gives squared-Euclidean cost).
        normalize_cost: If ``True``, divide every cost matrix by the mean of
            the (detached) cross cost so ``reg`` acts on an ``O(1)`` scale.
        debias: If ``True`` return the debiased divergence; if ``False`` return
            the raw regularised OT cost :math:`\\mathrm{OT}_\\varepsilon(x, y)`.
        n_iter: Maximum Sinkhorn iterations.

    Returns:
        A scalar tensor carrying gradients w.r.t. ``x`` (and ``y``).
    """
    if x.dim() != 2 or y.dim() != 2:
        raise ValueError(
            f"Expected 2-D point clouds, got x={tuple(x.shape)}, y={tuple(y.shape)}."
        )
    if x.shape[1] != y.shape[1]:
        raise ValueError(
            f"Dimension mismatch: x has D={x.shape[1]}, y has D={y.shape[1]}."
        )

    a = _uniform_weights(x.shape[0], x) if a is None else a
    b = _uniform_weights(y.shape[0], y) if b is None else b

    def _ot(u: torch.Tensor, v: torch.Tensor, wu: torch.Tensor, wv: torch.Tensor):
        cost = torch.cdist(u, v, p=2) ** p
        if normalize_cost:
            scale = cost.detach().mean().clamp_min(1e-8)
            cost = cost / scale
        return ot.sinkhorn2(wu, wv, cost, reg=reg, method="sinkhorn_log", numItermax=n_iter)

    ot_xy = _ot(x, y, a, b)
    if not debias:
        return ot_xy
    ot_xx = _ot(x, x, a, a)
    ot_yy = _ot(y, y, b, b)
    return ot_xy - 0.5 * ot_xx - 0.5 * ot_yy


def kinetic_energy_loss(velocities: torch.Tensor, dt: float) -> torch.Tensor:
    r"""Average kinetic energy accumulated along the trajectories.

    Approximates, per sample, the action integral

    .. math::

        \int_{0}^{T} \tfrac{1}{2}\, \lVert v_\phi(x_i(t)) \rVert^2 \, dt

    with the trapezoidal rule over the trajectory nodes, then averages over the
    batch.

    Args:
        velocities: Node velocities of shape ``[K + 1, B, D]`` as returned by
            :func:`velocity_ot.dynamics.integrate`.
        dt: Integration step size.

    Returns:
        A scalar tensor: the batch-averaged kinetic energy over one integration.
    """
    if velocities.dim() != 3:
        raise ValueError(
            f"Expected velocities of shape [K+1, B, D], got {tuple(velocities.shape)}."
        )
    # Per-node, per-sample kinetic energy 0.5 * ||v||^2  ->  [K+1, B]
    ke_node = 0.5 * velocities.pow(2).sum(dim=-1)
    # Integrate over time (node axis) with the trapezoidal rule -> [B]
    ke_path = torch.trapezoid(ke_node, dx=dt, dim=0)
    return ke_path.mean()


def global_ot_loss(
    x0: torch.Tensor,
    x_cycle: torch.Tensor,
    reg: float = 0.05,
    **kwargs,
) -> torch.Tensor:
    """Stationarity loss over one full cycle.

    Because the data distribution is invariant under the cyclic flow, the cloud
    obtained by evolving ``x0`` over exactly one cycle should match ``x0`` in
    distribution. This term is the Sinkhorn divergence between the initial cloud
    and the once-around cloud.

    Args:
        x0: Initial coordinates ``[B, D]`` (the empirical stationary sample).
        x_cycle: Coordinates after one full cycle ``[B, D]``.
        reg: Entropic regularisation strength.
        **kwargs: Forwarded to :func:`sinkhorn_divergence`.

    Returns:
        A scalar tensor.
    """
    return sinkhorn_divergence(x0, x_cycle, reg=reg, **kwargs)


def angular_alignment_loss(
    velocity: torch.Tensor,
    grad_theta: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Encourage the velocity to align with the circular-coordinate gradient.

    .. math::

        \mathcal{L}_{align} = \frac{1}{N} \sum_{i=1}^N
            \left( 1 - \frac{v_\phi(x_i) \cdot \nabla\theta(x_i)}
                             {\lVert v_\phi(x_i)\rVert\,\lVert \nabla\theta(x_i)\rVert} \right)

    The term is scale-invariant in both arguments; it fixes the *direction* of
    the field (its rotational orientation), leaving the *speed* to the kinetic
    and sub-population terms.

    Args:
        velocity: Predicted velocities ``[N, D]``.
        grad_theta: Estimated :math:`\nabla\theta` at the same points ``[N, D]``
            (e.g. from :mod:`velocity_ot.circular_gradient`).
        eps: Small constant guarding against division by zero.

    Returns:
        A scalar tensor in ``[0, 2]`` (``0`` = perfectly aligned everywhere).
    """
    if velocity.shape != grad_theta.shape:
        raise ValueError(
            f"Shape mismatch: velocity={tuple(velocity.shape)}, "
            f"grad_theta={tuple(grad_theta.shape)}."
        )
    cos = (velocity * grad_theta).sum(dim=-1) / (
        velocity.norm(dim=-1) * grad_theta.norm(dim=-1) + eps
    )
    return (1.0 - cos).mean()


def subpopulation_ot_loss(
    x_sub_evolved: torch.Tensor,
    x_sub_target: torch.Tensor,
    reg: float = 0.05,
    **kwargs,
) -> torch.Tensor:
    r"""Temporal-anchor loss for a labelled sub-population.

    Given starting points :math:`X_{sub}(0)` evolved under the learned field for
    a physical time :math:`t'`, this term is the Sinkhorn divergence between the
    model-evolved cloud and the known target cloud :math:`X_{sub}(t')`. It ties
    the model's latent time to physical cycle time and thereby fixes the overall
    *speed* (and phase) of the field, anchoring the flow so that its
    time-integrated density reproduces the stationary data distribution.

    Args:
        x_sub_evolved: Model-evolved sub-population at ``t'`` ``[N_sub, D]``.
        x_sub_target: Ground-truth targets at ``t'`` ``[M_sub, D]``.
        reg: Entropic regularisation strength.
        **kwargs: Forwarded to :func:`sinkhorn_divergence`.

    Returns:
        A scalar tensor.
    """
    return sinkhorn_divergence(x_sub_evolved, x_sub_target, reg=reg, **kwargs)
