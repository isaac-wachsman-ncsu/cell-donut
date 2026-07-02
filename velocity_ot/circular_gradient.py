"""
circular_gradient.py
====================================================================
Method A: local weighted least-squares estimation of the gradient
field of a circle-valued coordinate  theta : R^D -> S^1.

Context
-------
Given a point cloud X in R^D and a circular coordinate theta produced by
persistent-cohomology circular coordinates (e.g. de Silva-Morozov-
Vejdemo-Johansson, or the density-robust weighted-Laplacian variant of
Paik & Park), we want grad(theta) at arbitrary query points.

theta has no single-valued global lift, so we cannot fit theta values
directly.  What IS well defined is the *wrapped difference* of theta
along an edge, which equals the (small, single-valued) harmonic cocycle
that the circular-coordinate optimisation already produced:

        d_ij  :=  wrap(theta_j - theta_i)  =  2*pi * harmonic_cocycle(i->j)

These wrapped differences are noisy linear measurements of the unknown
gradient g = grad(theta)(x0):

        d_ij  ~=  g . (x_j - x_i)             (first-order Taylor)

Collecting the edges near a query point x0 and solving the kernel-
weighted least-squares problem

        g_hat(x0) = argmin_g  sum_ij  w_ij ( g . e_ij  -  d_ij )^2
                  = ( E^T W E )^{-1} E^T W d                       (*)

with  e_ij = x_j - x_i ,  w_ij = K(||midpoint_ij - x0|| / h) ,
gives the gradient at x0.  Because the weights depend on x0, the map
x0 -> g_hat(x0) is a smooth vector field.

If the data lies on a d-dimensional manifold (d < D) the gradient is a
cotangent vector; we then solve (*) inside the local tangent space
(top-d eigenvectors of the structure tensor M = E^T W E) and lift back,
which removes the noise-amplifying normal directions.

Units convention
----------------
theta is assumed to be in RADIANS in [0, 2*pi).  Edge differences and
the returned gradient are therefore in radians per unit length.

Author: implementation of the "Method A" recipe.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------- #
#  Wrapping helpers
# --------------------------------------------------------------------------- #


def wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap angle differences (radians) into the principal branch (-pi, pi]."""
    return np.angle(np.exp(1j * np.asarray(x, dtype=float)))


# --------------------------------------------------------------------------- #
#  Building the local reconstruction graph
# --------------------------------------------------------------------------- #


def build_neighbor_graph(
    X: np.ndarray,
    k: int | None = 8,
    radius: float | None = None,
    symmetric: bool = True,
) -> np.ndarray:
    """
    Build a set of undirected edges over the point cloud X.

    The gradient-reconstruction graph is independent of whatever complex
    was used for the cohomology computation; any reasonable local
    neighbourhood graph works.  Provide EITHER k (k-nearest-neighbours)
    OR radius (epsilon-ball graph).

    Parameters
    ----------
    X : (N, D) array of point coordinates.
    k : number of nearest neighbours per point (ignored if radius given).
    radius : connect all pairs closer than this distance.
    symmetric : if True, store each undirected edge once with i < j.

    Returns
    -------
    edges : (M, 2) int array of vertex index pairs with edges[:, 0] < edges[:, 1].
    """
    X = np.asarray(X, dtype=float)
    N = X.shape[0]
    tree = cKDTree(X)

    pairs = set()
    if radius is not None:
        for i, j in tree.query_pairs(r=radius):
            pairs.add((i, j) if i < j else (j, i))
    else:
        if k is None:
            raise ValueError("Provide either k or radius.")
        kq = min(k + 1, N)  # +1 because the first neighbour is the point itself
        _, idx = tree.query(X, k=kq)
        idx = np.atleast_2d(idx)
        for i in range(N):
            for j in idx[i, 1:]:  # skip self
                a, b = (i, int(j))
                pairs.add((a, b) if a < b else (b, a))

    if not pairs:
        raise ValueError("No edges produced; loosen k / radius.")
    edges = np.array(sorted(pairs), dtype=int)
    return edges


# --------------------------------------------------------------------------- #
#  Edge measurements (the data 1-form)
# --------------------------------------------------------------------------- #


def edge_wrapped_differences(theta: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Wrapped differences d_ij = wrap(theta_j - theta_i) in radians, oriented i->j
    along the canonical edge orientation (edges[:,0] -> edges[:,1]).

    This equals 2*pi times the harmonic cocycle of the circular coordinate,
    so it is the correct, branch-cut-free data 1-form for Method A.
    """
    theta = np.asarray(theta, dtype=float)
    i, j = edges[:, 0], edges[:, 1]
    return wrap_to_pi(theta[j] - theta[i])


def harmonic_cocycle_from_cochain(
    edges: np.ndarray,
    f: np.ndarray,
    alpha: np.ndarray,
    scale: float = 2.0 * np.pi,
) -> np.ndarray:
    """
    Build the edge measurements directly from the optimisation internals,
    for users who kept them.  Given the real 0-cochain f and the integer
    cocycle alpha (aligned to the SAME canonical edge orientation i<j as
    `edges`), the harmonic cocycle in turns is

            bar_alpha(i->j) = (f_j - f_i) + alpha_ij

    Multiplying by `scale` (= 2*pi by default) returns it in radians.
    This is exactly equal to edge_wrapped_differences(2*pi*(f % 1), edges)
    when the representative is harmonic, but avoids any re-wrapping.
    """
    f = np.asarray(f, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    i, j = edges[:, 0], edges[:, 1]
    return scale * ((f[j] - f[i]) + alpha)


# --------------------------------------------------------------------------- #
#  Bandwidth selection
# --------------------------------------------------------------------------- #


def auto_bandwidth(X: np.ndarray, edges: np.ndarray, factor: float = 2.0) -> float:
    """
    Heuristic isotropic bandwidth: `factor` times the median edge length.
    A few neighbouring edges then receive appreciable weight at each query.
    """
    X = np.asarray(X, dtype=float)
    lengths = np.linalg.norm(X[edges[:, 1]] - X[edges[:, 0]], axis=1)
    return float(factor * np.median(lengths))


# --------------------------------------------------------------------------- #
#  Core estimator at a single query point
# --------------------------------------------------------------------------- #


def estimate_gradient_at(
    x0: np.ndarray,
    X: np.ndarray,
    edges: np.ndarray,
    d: np.ndarray,
    bandwidth: float,
    intrinsic_dim: int | None = None,
    ridge: float = 0.0,
    weight_floor: float = 1e-12,
    return_diagnostics: bool = False,
):
    """
    Estimate grad(theta) at a single query point x0 via Method A.

    Parameters
    ----------
    x0 : (D,) query point.
    X : (N, D) point coordinates.
    edges : (M, 2) canonical edges (i < j).
    d : (M,) edge measurements wrap(theta_j - theta_i) in radians (oriented i->j).
    bandwidth : kernel scale h (Gaussian on midpoint-to-x0 distance).
    intrinsic_dim : if not None, solve inside the local tangent space of this
        dimension (top eigenvectors of the structure tensor) and lift back.
        Use the manifold dimension (e.g. 1 for a circle, 2 for a surface).
    ridge : Tikhonov regularisation added to the structure tensor as
        ridge * trace(M)/D * I.  Stabilises ill-conditioned neighbourhoods.
    weight_floor : ignore edges whose weight is below this (efficiency only).
    return_diagnostics : also return a dict with M eigenvalues, residual, etc.

    Returns
    -------
    g : (D,) estimated gradient vector at x0  (radians per unit length).
    (optionally) diagnostics : dict.
    """
    x0 = np.asarray(x0, dtype=float)
    X = np.asarray(X, dtype=float)
    D = X.shape[1]

    e = X[edges[:, 1]] - X[edges[:, 0]]          # (M, D) edge vectors
    mid = 0.5 * (X[edges[:, 0]] + X[edges[:, 1]])  # (M, D) midpoints

    # Gaussian kernel on distance from edge midpoint to the query point.
    r2 = np.sum((mid - x0) ** 2, axis=1)
    w = np.exp(-0.5 * r2 / (bandwidth ** 2))

    keep = w > weight_floor
    if np.count_nonzero(keep) < max(D, 1):
        # Not enough local information; fall back to all edges.
        keep = np.ones_like(w, dtype=bool)
    e_k = e[keep]
    d_k = d[keep]
    w_k = w[keep]

    # Structure tensor M = sum_ij w e e^T   and   b = sum_ij w d e
    we = w_k[:, None] * e_k
    M = e_k.T @ we                # (D, D)
    b = we.T @ d_k                # (D,)

    if ridge > 0.0:
        M = M + ridge * (np.trace(M) / max(D, 1)) * np.eye(D)

    if intrinsic_dim is not None and intrinsic_dim < D:
        # Solve inside the local tangent space spanned by the top eigenvectors
        # of the (symmetric PSD) structure tensor M.
        evals, evecs = np.linalg.eigh(M)            # ascending order
        T = evecs[:, -intrinsic_dim:]               # (D, k) tangent basis
        Mt = T.T @ M @ T                            # (k, k)
        bt = T.T @ b                                # (k,)
        ct, *_ = np.linalg.lstsq(Mt, bt, rcond=None)
        g = T @ ct
    else:
        g, *_ = np.linalg.lstsq(M, b, rcond=None)

    if not return_diagnostics:
        return g

    resid = e_k @ g - d_k
    diagnostics = {
        "n_edges_used": int(np.count_nonzero(keep)),
        "structure_tensor": M,
        "eigvals": np.linalg.eigvalsh(M),
        "weighted_residual_rms": float(
            np.sqrt(np.sum(w_k * resid ** 2) / max(np.sum(w_k), 1e-30))
        ),
    }
    return g, diagnostics


# --------------------------------------------------------------------------- #
#  Vector field over many query points
# --------------------------------------------------------------------------- #


def estimate_gradient_field(
    X: np.ndarray,
    theta: np.ndarray | None = None,
    edges: np.ndarray | None = None,
    d: np.ndarray | None = None,
    query_points: np.ndarray | None = None,
    k: int | None = 8,
    radius: float | None = None,
    bandwidth: float | None = None,
    bandwidth_factor: float = 2.0,
    intrinsic_dim: int | None = None,
    ridge: float = 0.0,
):
    """
    Evaluate the Method-A gradient vector field at a set of query points.

    Two ways to supply the data 1-form:
      * pass `theta` (per-vertex circular coordinate in radians); the wrapped
        edge differences are computed internally, OR
      * pass `d` directly (precomputed wrapped differences / harmonic cocycle
        in radians, oriented along `edges`).
    If `edges` is None it is built from X with k / radius.

    Parameters
    ----------
    X : (N, D) point coordinates.
    theta : (N,) circular coordinate in radians (optional if d is given).
    edges : (M, 2) canonical edges i<j (optional; built from X if None).
    d : (M,) edge measurements in radians (optional if theta is given).
    query_points : (Q, D) points at which to evaluate; defaults to X itself.
    k, radius : neighbour-graph parameters (used only if edges is None).
    bandwidth : kernel scale; if None, set by auto_bandwidth.
    bandwidth_factor : factor for auto_bandwidth when bandwidth is None.
    intrinsic_dim : tangent-space dimension for the manifold case.
    ridge : Tikhonov stabilisation passed to the per-point solver.

    Returns
    -------
    G : (Q, D) array of gradient vectors.
    info : dict with the edges, edge measurements d, bandwidth used.
    """
    X = np.asarray(X, dtype=float)

    if edges is None:
        edges = build_neighbor_graph(X, k=k, radius=radius)

    if d is None:
        if theta is None:
            raise ValueError("Provide either theta or precomputed d.")
        d = edge_wrapped_differences(theta, edges)
    d = np.asarray(d, dtype=float)

    if bandwidth is None:
        bandwidth = auto_bandwidth(X, edges, factor=bandwidth_factor)

    if query_points is None:
        query_points = X
    query_points = np.asarray(query_points, dtype=float)

    G = np.empty_like(query_points, dtype=float)
    for q in range(query_points.shape[0]):
        G[q] = estimate_gradient_at(
            query_points[q],
            X,
            edges,
            d,
            bandwidth=bandwidth,
            intrinsic_dim=intrinsic_dim,
            ridge=ridge,
        )

    info = {"edges": edges, "d": d, "bandwidth": bandwidth}
    return G, info
