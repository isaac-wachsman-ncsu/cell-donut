"""Graph optimal-transport initialisation targets for the velocity field.

High-dimensional velocity fields are hard to learn from the composite OT
objective alone. This module builds a cheap, informative *initialisation
target*: a directed transition plan ``P`` obtained by entropic optimal
transport on the data graph, using an **effective-resistance** ground cost and
a convex **angular barrier** that forces mass to move a moderate step *forward*
along the circular coordinate. The plan is turned into a per-cell displacement
target ``u_i`` that supervises :math:`v_\\phi` during an initialisation stage.

Pipeline
--------
1. ``C``   = effective resistance between cells (``get_eff_res`` style).
2. ``Phi`` = convex barrier of the wrapped forward angular step (0, delta).
3. ``P``   = Sinkhorn plan for uniform marginals with cost ``C + w * Phi``.
4. ``u_i`` = (row-normalised ``P`` @ X)_i - x_i   (expected forward displacement).

The effective resistance mirrors ``dist_utils.get_eff_res`` (symmetric kNN graph,
Laplacian pseudo-inverse, optional von-Luxburg correction) but depends only on
numpy / scipy / scikit-learn, so it runs without the wider ``vis_utils`` stack.
If the real ``dist_utils.get_eff_res`` is importable (or supplied) it is used
instead.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import ot


# --------------------------------------------------------------------------- #
#  Graph + effective resistance
# --------------------------------------------------------------------------- #
def sknn_adjacency(X: np.ndarray, k: int) -> np.ndarray:
    """Symmetric, unweighted k-nearest-neighbour adjacency ``[N, N]``."""
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    k = int(min(k, n - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)  # includes self as column 0
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].reshape(-1)
    A = np.zeros((n, n), dtype=float)
    A[rows, cols] = 1.0
    return np.maximum(A, A.T)  # symmetrise


def effective_resistance(X: np.ndarray, k: int = 15, corrected: bool = True) -> np.ndarray:
    """Effective-resistance distance on the unweighted symmetric kNN graph.

    Self-contained re-implementation of ``dist_utils.get_eff_res`` (unweighted,
    von-Luxburg-corrected by default): ``EffR = L^+_{ii} + L^+_{jj} - 2 L^+_{ij}``
    with the connected-graph pseudo-inverse trick, then the degree correction
    ``- (1/d_i + 1/d_j) + 2 A_ij /(d_i d_j)``.

    Args:
        X: Coordinates ``[N, D]``.
        k: Neighbours for the kNN graph.
        corrected: Apply the von-Luxburg correction.

    Returns:
        ``[N, N]`` effective-resistance matrix (zero diagonal, non-negative).
    """
    A = sknn_adjacency(X, k)
    n = A.shape[0]
    deg = A.sum(1)
    L = np.diag(deg) - A
    ones = np.ones((n, n)) / n
    Lp = np.linalg.inv(L + ones) - ones            # Moore-Penrose pseudo-inverse
    dL = np.diag(Lp).reshape(n, 1)
    EffR = dL + dL.T - 2.0 * Lp
    if corrected:
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = np.where(deg > 0, 1.0 / deg, 0.0)
            corr = inv.reshape(n, 1) + inv.reshape(1, n)
            np.fill_diagonal(corr, 0.0)
            deg_out = np.outer(deg, deg)
            EffR = EffR - corr + 2.0 * A / np.where(deg_out > 0, deg_out, 1.0)
    EffR = np.clip(EffR, 0.0, None)
    np.fill_diagonal(EffR, 0.0)
    return EffR


# --------------------------------------------------------------------------- #
#  Angular barrier
# --------------------------------------------------------------------------- #
def auto_delta(theta: np.ndarray, A: np.ndarray, factor: float = 3.0) -> float:
    """Heuristic ``delta``: ``factor`` x median forward angular step over edges."""
    i, j = np.nonzero(np.triu(A, 1))
    if i.size == 0:
        return float(np.pi / 2)
    z = np.mod(theta[j] - theta[i], 2.0 * np.pi)
    step = np.minimum(z, 2.0 * np.pi - z)          # undirected angular gap
    step = step[step > 1e-9]
    med = float(np.median(step)) if step.size else float(np.pi / 8)
    return float(np.clip(factor * med, 1e-3, np.pi))


def angle_barrier(
    theta: np.ndarray, delta: float, phi_max: float = 50.0
) -> np.ndarray:
    """Convex forward-step barrier ``Phi_ij`` over the wrapped angular step.

    With ``z_ij = (theta_j - theta_i) mod 2*pi`` the barrier is

    .. math::  \\Phi(z) = \\mathrm{clip}\\Big(\\frac{\\delta^2}{4\\,z(\\delta-z)} - 1,\\,
                                          0,\\, \\phi_{\\max}\\Big) \\ \\text{on}\\ (0,\\delta),

    and ``phi_max`` elsewhere. It is ``0`` at the ideal mid-step ``z=delta/2``,
    grows convexly toward both ends, and walls off backward (``z<0`` -> large
    ``mod``), too-far (``z>=delta``) and self (``z=0``) transitions. Normalised
    by ``delta`` so its interior scale is comparable across ``delta``.

    Args:
        theta: Circular coordinate ``[N]`` (radians).
        delta: Maximum allowed forward step (radians).
        phi_max: Wall value for forbidden / near-boundary transitions.

    Returns:
        ``[N, N]`` barrier matrix.
    """
    theta = np.asarray(theta, dtype=float).reshape(-1)
    z = np.mod(theta[None, :] - theta[:, None], 2.0 * np.pi)  # z_ij = theta_j - theta_i
    Phi = np.full_like(z, phi_max)
    inside = (z > 0.0) & (z < delta)
    zz = z[inside]
    val = (delta ** 2) / (4.0 * zz * (delta - zz)) - 1.0
    Phi[inside] = np.minimum(np.clip(val, 0.0, None), phi_max)
    return Phi


# --------------------------------------------------------------------------- #
#  Transition plan + displacement targets
# --------------------------------------------------------------------------- #
def transition_plan(
    C: np.ndarray,
    Phi: np.ndarray,
    reg: float = 0.05,
    angle_weight: float = 1.0,
    n_iter: int = 500,
    normalize_cost: bool = True,
) -> np.ndarray:
    """Entropic OT transition plan for uniform marginals.

    Effective cost is ``M = C_norm + angle_weight * Phi`` (barrier as a penalty;
    a negative ``angle_weight`` reproduces a literal ``C - Phi``). Solved in the
    log domain for stability.

    Returns:
        ``[N, N]`` transport plan ``P`` (rows/cols sum to ``1/N``).
    """
    C = np.asarray(C, dtype=float)
    if normalize_cost:
        scale = np.median(C[C > 0]) if np.any(C > 0) else 1.0
        C = C / max(scale, 1e-8)
    M = C + angle_weight * np.asarray(Phi, dtype=float)
    n = M.shape[0]
    a = np.full(n, 1.0 / n)
    b = np.full(n, 1.0 / n)
    return ot.sinkhorn(a, b, M, reg=reg, method="sinkhorn_log", numItermax=n_iter)


def displacement_targets(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Expected forward displacement ``u_i = (P_norm @ X)_i - x_i`` ``[N, D]``."""
    X = np.asarray(X, dtype=float)
    row = P.sum(1, keepdims=True)
    P_norm = P / np.clip(row, 1e-12, None)
    barycenter = P_norm @ X
    return (barycenter - X).astype(np.float32)


def graph_ot_init_targets(
    X: np.ndarray,
    theta: np.ndarray,
    knn: int = 15,
    delta: float | None = None,
    angle_weight: float = 1.0,
    reg: float = 0.05,
    phi_max: float = 50.0,
    corrected: bool = True,
    eff_res_fn: Callable | np.ndarray | None = None,
    n_iter: int = 500,
) -> tuple[np.ndarray, dict]:
    """Build the graph-OT displacement targets ``u`` for NN initialisation.

    Args:
        X: Coordinates ``[N, D]`` (the fit space).
        theta: Circular coordinate ``[N]`` (radians).
        knn: Neighbours for the effective-resistance graph.
        delta: Forward-step barrier width; auto-set from the graph if ``None``.
        angle_weight: Weight of the angular penalty (negative flips its sign).
        reg: Entropic regularisation for the plan.
        phi_max: Barrier wall value.
        corrected: von-Luxburg correction for the built-in effective resistance.
        eff_res_fn: A precomputed ``[N, N]`` cost matrix, or a callable
            ``fn(X, knn) -> [N, N]`` (e.g. ``dist_utils.get_eff_res``). If
            ``None``, ``dist_utils.get_eff_res`` is used when importable, else
            the built-in :func:`effective_resistance`.
        n_iter: Max Sinkhorn iterations.

    Returns:
        ``(u, info)`` where ``u`` is ``[N, D]`` and ``info`` holds ``delta``,
        the plan ``P`` and the cost ``C``.
    """
    X = np.asarray(X, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)

    if isinstance(eff_res_fn, np.ndarray):
        C = np.asarray(eff_res_fn, dtype=float)
    elif callable(eff_res_fn):
        C = np.asarray(eff_res_fn(X, knn), dtype=float)
    else:
        try:  # prefer the project's real implementation when available
            from dist_utils import get_eff_res  # type: ignore

            C = np.asarray(get_eff_res(X, knn, corrected=corrected), dtype=float)
        except Exception:
            C = effective_resistance(X, k=knn, corrected=corrected)

    A = sknn_adjacency(X, knn)
    if delta is None:
        delta = auto_delta(theta, A)
    Phi = angle_barrier(theta, delta, phi_max=phi_max)
    P = transition_plan(C, Phi, reg=reg, angle_weight=angle_weight, n_iter=n_iter)
    u = displacement_targets(X, P)
    return u, {"delta": float(delta), "P": P, "cost": C}
