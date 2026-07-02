"""Loss terms for optimal-transport velocity-field learning.

The composite training objective is

.. math::

    \\mathcal{L} = \\lambda_1 \\mathcal{L}_{KE}
                 + \\lambda_2 \\mathcal{L}_{stationarity}
                 + \\lambda_3 \\mathcal{L}_{align}
                 + \\lambda_4 \\mathcal{L}_{OT\\_sub}

with the four terms defined below (``stationarity`` matches the flow's
time-marginal density to the data; ``OT_sub`` enforces that a subset returns to
itself after one cycle). All optimal-transport terms use the ``POT`` library
through its PyTorch backend so gradients propagate to the network parameters.

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


def _subsample(
    x: torch.Tensor, n: int, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Randomly select at most ``n`` rows of ``x`` (returns ``x`` unchanged if small)."""
    if x.shape[0] <= n:
        return x
    idx = torch.randperm(x.shape[0], device=x.device, generator=generator)[:n]
    return x[idx]


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


def stationarity_loss(
    trajectory: torch.Tensor,
    data: torch.Tensor,
    reg: float = 0.05,
    n_points: int = 256,
    include_endpoint: bool = False,
    generator: torch.Generator | None = None,
    **kwargs,
) -> torch.Tensor:
    r"""Match the flow's time-marginal density to the data distribution.

    Take a subset of the data, evolve it under the field over one cycle, pool
    the positions sampled *uniformly in time* along the resulting trajectories,
    and compare that pooled cloud to the entire original data distribution. If
    the learned dynamics are stationary in the same sense as the true dynamics,
    the time-average of the pushed-forward density,

    .. math::

        \bar{\rho}(x) = \frac{1}{T} \int_0^T (\Phi_t)_\# \mu_{\mathrm{sub}}(x)\, dt,

    equals the stationary data density :math:`p_{\mathrm{data}}`. This term is
    the Sinkhorn divergence between an empirical estimate of
    :math:`\bar{\rho}` (the pooled trajectory nodes) and ``data``.

    Note that this constrains the *shape* of the density along the cycle (hence
    the relative speed profile) but is invariant to the number of loops
    completed in ``[0, T]``; the loop count is pinned by
    :func:`cycle_consistency_loss` and the kinetic-energy term.

    Important:
        The ``trajectory`` should be integrated from a **localized** subset (a
        small clump / arc that does *not* already resemble ``p_data``).
        Otherwise the term is trivially satisfied at zero speed, because a
        random subset already matches the data distribution without any motion.
        :class:`velocity_ot.VelocityFieldEstimator` handles this by seeding from
        a contiguous arc of the circular coordinate.

    Args:
        trajectory: Node positions from one integration over ``[0, T]``, shape
            ``[K + 1, B, D]`` (as returned by
            :func:`velocity_ot.dynamics.integrate`).
        data: The full data cloud ``[N, D]`` (or a large sample of it).
        reg: Entropic regularisation strength.
        n_points: Cap on the number of points used on each side of the Sinkhorn
            divergence (both the pooled cloud and ``data`` are randomly
            subsampled to at most this many points for tractability).
        include_endpoint: If ``True`` include the ``t = T`` node in the pool;
            by default it is dropped so a closed cycle is not double-counted
            (uniform sampling of ``[0, T)``).
        generator: Optional RNG for the subsampling.
        **kwargs: Forwarded to :func:`sinkhorn_divergence`.

    Returns:
        A scalar tensor.
    """
    if trajectory.dim() != 3:
        raise ValueError(
            f"Expected trajectory [K+1, B, D], got {tuple(trajectory.shape)}."
        )
    nodes = trajectory if include_endpoint else trajectory[:-1]
    pooled = nodes.reshape(-1, nodes.shape[-1])  # [K*B, D], uniform in time

    pooled = _subsample(pooled, n_points, generator)
    target = _subsample(data, n_points, generator)
    return sinkhorn_divergence(pooled, target, reg=reg, **kwargs)


def cycle_consistency_loss(
    x0: torch.Tensor,
    x_cycle: torch.Tensor,
    reg: float = 0.05,
    **kwargs,
) -> torch.Tensor:
    r"""Periodicity anchor: a subset returns to itself after one cycle.

    Sinkhorn divergence between a subset ``x0`` and the *same* subset evolved
    over exactly one full cycle ``x_cycle = \Phi_T(x0)``. Source and sink are
    the same set of points, so minimising this term enforces that the flow
    carries the subset once around and back to its own distribution in time
    ``T`` (i.e. an integer number of loops; combined with the kinetic-energy
    term this selects a single loop). This is the term referred to as
    ``OT_sub``.

    Args:
        x0: Subset start points ``[B, D]``.
        x_cycle: The same subset after one full cycle ``[B, D]``.
        reg: Entropic regularisation strength.
        **kwargs: Forwarded to :func:`sinkhorn_divergence`.

    Returns:
        A scalar tensor.
    """
    return sinkhorn_divergence(x_cycle, x0, reg=reg, **kwargs)


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
