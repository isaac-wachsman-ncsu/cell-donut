"""Circular coordinates from persistent cohomology (weighted-Laplacian method).

This is the *upstream* step of the pipeline: it turns a point cloud into the
circular coordinate ``theta`` that every other part of :mod:`velocity_ot`
consumes (``adata.obs['circular_coords']``).

Pipeline
--------
1. ``D_eff`` = **effective-resistance** distance on a symmetric kNN graph.
   Effective resistance is density- and topology-aware: it is large between
   points that are close in ambient space but far through the graph (e.g.
   across the hole of a cycle), which makes the ``H1`` class of a limit cycle
   far more prominent than with Euclidean distance.
2. ``H1`` persistent cohomology of ``D_eff`` over ``Z/pZ`` (via ``ripser``),
   taking the most persistent class and its representative cocycle.
3. The cocycle is lifted ``Z/pZ -> Z`` and smoothed by a **weighted** least
   squares problem on the 1-skeleton at a fixed filtration scale. The edge
   weights ``w(uv) = 1 / (deg(u) + deg(v))`` down-weight edges in dense
   regions, so the resulting coordinate advances at a rate that is robust to
   non-uniform sampling density (rather than bunching up where data is dense).

Only ``numpy``/``scipy``/``scikit-learn`` plus ``ripser`` are required.

Example
-------
>>> from velocity_ot import circular_coordinates
>>> theta = circular_coordinates(adata.obsm["X_pca"], knn=100)
>>> adata.obs["circular_coords"] = theta
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

_TWO_PI = 2.0 * np.pi


# --------------------------------------------------------------------------- #
#  Effective resistance  (port of eff-ph `dist_utils.get_eff_res`)
# --------------------------------------------------------------------------- #
def _sknn_graph(X: np.ndarray, k: int, metric: str = "euclidean", weighted: bool = False):
    """Symmetric kNN graph as a COO matrix (unweighted, or distance-weighted)."""
    from sklearn.neighbors import NearestNeighbors

    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    k = int(min(k, n - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    dist, idx = nn.kneighbors(X)          # column 0 is the point itself
    rows = np.repeat(np.arange(n), k)
    cols = idx[:, 1:].reshape(-1)
    data = dist[:, 1:].reshape(-1) if weighted else np.ones(n * k)
    G = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    return G.maximum(G.transpose()).tocoo()   # symmetrise


def _eff_res_connected(A) -> np.ndarray:
    """Effective resistance of a *connected* graph via the Laplacian pseudo-inverse."""
    n = A.shape[0]
    A_dense = A.toarray() if sp.issparse(A) else np.asarray(A, dtype=float)
    L = np.diag(A_dense.sum(0)) - A_dense
    ones = np.ones((n, n)) / n
    Lp = np.linalg.inv(L + ones) - ones         # Moore-Penrose pseudo-inverse
    d = np.diag(Lp).reshape(n, 1)
    return d + d.T - 2.0 * Lp


def effective_resistance_dist(
    X: np.ndarray,
    knn: int = 100,
    corrected: bool = True,
    weighted: bool = False,
    disconnect: bool = True,
    metric: str = "euclidean",
) -> np.ndarray:
    """Effective-resistance distance matrix on the symmetric kNN graph.

    Self-contained port of ``eff_ph.utils.dist_utils.get_eff_res`` (only the
    branch this project uses), depending on numpy/scipy/scikit-learn instead of
    the ``vis_utils``/torch/UMAP stack.

    Args:
        X: Coordinates ``[N, D]`` (e.g. ``adata.obsm['X_pca']``).
        knn: Neighbours of the symmetric kNN graph.
        corrected: Apply the von-Luxburg degree correction, which removes the
            ``1/deg`` term that otherwise dominates effective resistance on
            large graphs and washes out the global (cycle) structure.
        weighted: Use the distance-weighted graph instead of the unweighted one.
        disconnect: Compute each connected component separately (resistance
            between components is set to twice the maximum finite value).
        metric: Metric for the kNN search.

    Returns:
        ``[N, N]`` effective-resistance distance matrix.
    """
    G = _sknn_graph(X, knn, metric=metric, weighted=weighted)
    G.data = 1.0 / G.data                     # conductance = 1 / resistance

    if disconnect:
        n_comp, labels = sp.csgraph.connected_components(G)
        EffR = np.full(G.shape, np.inf)
        Gc = G.tocsr()
        for c in range(n_comp):
            m = labels == c
            comp = np.where(m)[0]
            EffR[np.ix_(comp, comp)] = _eff_res_connected(Gc[comp, :][:, comp])
    else:
        EffR = _eff_res_connected(G)

    finite = np.isfinite(EffR)
    if not finite.all():                      # bridge disconnected components
        EffR[~finite] = EffR[finite].max() * 2.0

    if corrected:                             # von Luxburg fix
        degs = np.asarray(G.sum(axis=1)).reshape(-1, 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            deg_dist = 1.0 / degs + 1.0 / degs.T
        np.fill_diagonal(deg_dist, 0.0)
        EffR = EffR - deg_dist + 2.0 * G.toarray() / (degs * degs.T)

    EffR = np.asarray(EffR, dtype=float)
    np.fill_diagonal(EffR, 0.0)
    return EffR


# --------------------------------------------------------------------------- #
#  Weighted-Laplacian circular coordinates
# --------------------------------------------------------------------------- #
def suggest_thresh(D_eff: np.ndarray, q: float = 0.10) -> float:
    """A Rips cut-off for large ``N``: the ``q``-quantile of pairwise distances.

    The full Rips complex is intractable beyond ~1000 points. Capping the
    filtration keeps the (short-lived, low-scale) ``H1`` class of a cycle while
    discarding the expensive high-scale simplices that only fill the complex in.
    """
    D_eff = np.asarray(D_eff, dtype=float)
    off = D_eff[~np.eye(D_eff.shape[0], dtype=bool)]
    return float(np.quantile(off, q))



def weighted_circular_coords(
    D_eff: np.ndarray,
    prime: int = 47,
    cocycle_idx: int = 0,
    epsilon_mode: str | float = "midpoint",
    thresh: float | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """Density-robust circular coordinates from ``H1`` via a weighted Laplacian.

    Takes the ``cocycle_idx``-th most persistent ``H1`` class of ``D_eff``,
    lifts its ``Z/pZ`` representative cocycle to ``Z``, and smooths it by
    weighted least squares on the 1-skeleton at a fixed filtration scale. Edge
    weights ``1 / (deg(u) + deg(v))`` make the coordinate's rate of advance
    robust to non-uniform density.

    Args:
        D_eff: Precomputed ``[N, N]`` distance matrix (effective resistance).
        prime: Field ``Z/pZ`` for the cohomology; a larger prime makes the lift
            to integer coefficients more reliable.
        cocycle_idx: Which ``H1`` class to use, ranked by persistence (0 = most
            persistent, i.e. the dominant cycle).
        epsilon_mode: Filtration scale of the fixed complex: ``"midpoint"``
            (``(birth + death) / 2``), ``"birth"``, or an explicit float.
        thresh: Cap on the Rips filtration passed to ``ripser``. The full
            complex costs ``O(N^3)`` memory and will exhaust RAM beyond roughly
            a thousand points, so for larger ``N`` set this to a value above the
            cycle's death but well below the diameter (see
            :func:`suggest_thresh`). ``None`` builds the full complex.
        verbose: Print the selected class and complex size.

    Returns:
        ``[N]`` circular coordinates in ``[0, 2*pi)``.
    """
    from ripser import ripser
    from scipy.sparse.linalg import lsqr

    D_eff = np.asarray(D_eff, dtype=float)
    N = D_eff.shape[0]

    # -- 1. persistent cohomology, most persistent H1 class --------------------
    kw = {} if thresh is None else {"thresh": float(thresh)}
    res = ripser(D_eff, distance_matrix=True, maxdim=1, coeff=prime,
                 do_cocycles=True, **kw)
    dgm, cocycles = res["dgms"][1], res["cocycles"][1]
    if len(dgm) == 0:
        raise ValueError("No H1 features found; the data has no detectable cycle.")

    order = np.argsort(dgm[:, 1] - dgm[:, 0])[::-1]      # by persistence, desc
    t = order[cocycle_idx]
    birth, death = dgm[t]
    if not np.isfinite(death):   # class still alive at the end of the filtration
        death = float(thresh) if thresh is not None else D_eff.max()
    if verbose:
        print(f"H1 class {cocycle_idx}: birth={birth:.4f} death={death:.4f} "
              f"persistence={death - birth:.4f}")

    # -- 2. fixed filtration scale --------------------------------------------
    if epsilon_mode == "midpoint":
        eps = 0.5 * (birth + death)
    elif epsilon_mode == "birth":
        eps = float(birth)
    elif isinstance(epsilon_mode, (int, float)):
        eps = float(epsilon_mode)
    else:
        raise ValueError(f"Unknown epsilon_mode: {epsilon_mode!r}")

    # -- 3. 1-skeleton at that scale ------------------------------------------
    adj = D_eff <= eps
    np.fill_diagonal(adj, False)
    deg = adj.sum(1).astype(float)
    rows, cols = np.where(np.triu(adj, 1))                # canonical i < j
    n_edges = rows.size
    if n_edges == 0:
        raise ValueError("Empty 1-skeleton at the chosen epsilon.")
    if verbose:
        print(f"complex at eps={eps:.4f}: {N} vertices, {n_edges} edges")

    # -- 4. density-robust edge weights  w = 1 / (deg_u + deg_v) --------------
    deg_sum = deg[rows] + deg[cols]
    w = np.where(deg_sum > 0, 1.0 / np.maximum(deg_sum, 1e-12), 1.0)

    # -- 5. lift the Z/p cocycle to Z on canonical edge orientations ----------
    # (vectorised: encode (i, j) as i * N + j and look it up by binary search)
    key = rows.astype(np.int64) * N + cols
    order_key = np.argsort(key)
    key_sorted = key[order_key]

    cyc = np.asarray(cocycles[t])
    u, v, val = cyc[:, 0].astype(np.int64), cyc[:, 1].astype(np.int64), cyc[:, 2].astype(np.int64)
    val = np.where(val <= prime // 2, val, val - prime)   # centred lift to Z
    flip = u > v                                          # reorient to i < j
    lo, hi = np.where(flip, v, u), np.where(flip, u, v)
    val = np.where(flip, -val, val)

    q = lo * N + hi
    pos = np.searchsorted(key_sorted, q)
    ok = (pos < key_sorted.size) & (key_sorted[np.clip(pos, 0, key_sorted.size - 1)] == q)
    alpha = np.zeros(n_edges)
    alpha[order_key[pos[ok]]] = val[ok]                   # edges outside the complex are dropped

    # -- 6. coboundary d0:  (d0 f)(u,v) = f(v) - f(u) -------------------------
    e_idx = np.arange(n_edges)
    d0 = sp.csr_matrix(
        (np.concatenate([-np.ones(n_edges), np.ones(n_edges)]),
         (np.concatenate([e_idx, e_idx]), np.concatenate([rows, cols]))),
        shape=(n_edges, N),
    )

    # -- 7. weighted least squares:  min || W (d0 f + alpha) || ---------------
    f = lsqr(sp.diags(w) @ d0, -w * alpha)[0]

    # -- 8. project the 0-cochain onto the circle -----------------------------
    return (f % 1.0) * _TWO_PI


# --------------------------------------------------------------------------- #
#  One-call convenience
# --------------------------------------------------------------------------- #
def circular_coordinates(
    X: np.ndarray,
    knn: int = 100,
    prime: int = 47,
    cocycle_idx: int = 0,
    epsilon_mode: str | float = "midpoint",
    thresh: float | str | None = "auto",
    corrected: bool = True,
    weighted: bool = False,
    disconnect: bool = True,
    metric: str = "euclidean",
    verbose: bool = True,
    return_dist: bool = False,
):
    """Circular coordinate ``theta`` for a cyclic point cloud (full pipeline).

    Effective-resistance distance -> ``H1`` persistent cohomology -> weighted
    Laplacian smoothing. This is the coordinate the rest of the package expects
    in ``adata.obs['circular_coords']``.

    Args:
        X: Coordinates ``[N, D]`` (typically ``adata.obsm['X_pca']``).
        knn: Neighbours for the effective-resistance graph.
        prime: Field ``Z/pZ`` for the persistent cohomology.
        cocycle_idx: Which ``H1`` class to use (0 = most persistent).
        epsilon_mode: ``"midpoint"``, ``"birth"``, or an explicit float.
        thresh: Rips filtration cap. "auto" (default) caps it only when
            N > 1000, where the full complex would exhaust memory; None forces
            the full complex; a float sets it explicitly.
        corrected, weighted, disconnect, metric: Passed to
            :func:`effective_resistance_dist`.
        verbose: Print diagnostics for the selected class.
        return_dist: Also return the ``[N, N]`` effective-resistance matrix
            (useful for plotting the persistence diagram).

    Returns:
        ``theta`` of shape ``[N]`` in ``[0, 2*pi)``, or ``(theta, D_eff)`` when
        ``return_dist`` is ``True``.
    """
    D_eff = effective_resistance_dist(
        X, knn=knn, corrected=corrected, weighted=weighted,
        disconnect=disconnect, metric=metric,
    )
    if thresh == "auto":
        # The full Rips complex is only tractable for small N.
        thresh = None if D_eff.shape[0] <= 1000 else suggest_thresh(D_eff)
        if verbose and thresh is not None:
            print(f"N={D_eff.shape[0]} > 1000: capping the Rips filtration at "
                  f"thresh={thresh:.4g} (pass thresh=None to disable)")
    theta = weighted_circular_coords(
        D_eff, prime=prime, cocycle_idx=cocycle_idx,
        epsilon_mode=epsilon_mode, thresh=thresh, verbose=verbose,
    )
    return (theta, D_eff) if return_dist else theta
