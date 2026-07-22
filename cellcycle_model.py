"""
cellcycle_model.py
==================
Auxiliary functions for generating a synthetic single-cell RNA-seq dataset from a
mechanistic ODE model of the mammalian cell cycle.

The dynamical core is the *skeleton* Cdk-network model of

    Gerard C. & Goldbeter A. (2011) "A skeleton model for the network of
    cyclin-dependent kinases driving the mammalian cell cycle."
    Interface Focus 1:24-35.  doi:10.1098/rsfs.2010.0008

which is itself the reduced backbone of the 39-variable model of

    Gerard C. & Goldbeter A. (2009) "Temporal self-organization of the cyclin/Cdk
    network driving the mammalian cell cycle."
    PNAS 106:21643-21648.  doi:10.1073/pnas.0903827106

We extend the skeleton model with an explicit transcriptional layer (mRNA
species) so that the observable state has the semantics of RNA abundance rather
than kinase activity.  The Antimony source and the derivation live in the
companion notebook, which owns the single authoritative definition of the ODE
system; every function here takes either that string or a compiled RoadRunner.

Requires: tellurium, numpy.

Everything here is deterministic and reproducible given ``seed``.
"""

from __future__ import annotations

import numpy as np
import tellurium as te


# --------------------------------------------------------------------------- #
# 1. The model
# --------------------------------------------------------------------------- #
# Notation follows Gerard & Goldbeter (2011), Table 1 & 2:
#   Md    = cyclin D/Cdk4-6 complex          [uM]
#   E2F   = active transcription factor E2F  [uM]
#   Me    = cyclin E/Cdk2 complex            [uM]
#   Ma    = cyclin A/Cdk2 complex            [uM]
#   Mb    = cyclin B/Cdk1 complex            [uM]
#   Cdc20 = active (phosphorylated) Cdc20    [uM]
# Time is in HOURS throughout.
#
# The four cyclin/Cdk complexes are no longer synthesised directly from their
# transcriptional activator; each is translated from an explicit mRNA pool:
#
#       d[mRNA_X]/dt = alpha_X * TF_X  -  beta_X * [mRNA_X]
#       d[Protein_X]/dt = k_X * [mRNA_X]  -  (degradation, unchanged)
#
# with TF_CCND1 = GF/(Kgf+GF), TF_CCNE1 = TF_CCNA2 = E2F, TF_CCNB1 = Ma,
# exactly mirroring the transcriptional wiring of the published skeleton model.
#
# The four "reporter" transcripts (MCM6, PCNA, PLK1, AURKA) are read-outs only:
# they are driven by E2F or by cyclin B/Cdk1 and feed back on nothing.  They
# raise the ambient dimension of the observed data without perturbing the
# dynamics (skew-product structure), which is what we want when stress-testing a
# manifold / optimal-transport method.

# NOTE: the Antimony model string itself lives in the notebook, which owns the
# single authoritative definition of the ODE system.  Every function here takes
# either that string or an already-compiled RoadRunner instance.


#: Names of the observed (mRNA) features, in the order they appear in ``adata.X``.
MRNA_GENES = [
    "CCND1", "CCNE1", "CCNA2", "CCNB1",
    "MCM6", "PCNA", "PLK1", "AURKA",
]

#: What drives each transcript (stored in ``adata.var``).
GENE_DRIVER = {
    "CCND1": "GF (constitutive)",
    "CCNE1": "E2F",
    "CCNA2": "E2F",
    "CCNB1": "cyclin A/Cdk2",
    "MCM6": "E2F",
    "PCNA": "E2F",
    "PLK1": "cyclin B/Cdk1",
    "AURKA": "cyclin B/Cdk1",
}


# --------------------------------------------------------------------------- #
# 2. Model construction
# --------------------------------------------------------------------------- #
def build_model(antimony: str, GF: float = 1.0,
                atol: float = 1e-10, rtol: float = 1e-10):
    """Compile the Antimony model and configure a tight CVODE integrator.

    Tight tolerances matter here.  ``Kdb = 0.005 uM`` makes cyclin B degradation
    near-zero-order, so the mitotic exit is a stiff, near-vertical drop; loose
    tolerances visibly distort the shape of the limit cycle and therefore the
    ground-truth vector field.
    """
    rr = te.loada(antimony)
    rr.integrator.absolute_tolerance = atol
    rr.integrator.relative_tolerance = rtol
    # CVODE's default cap of 500 internal steps per output point is not enough
    # when output points are far apart (e.g. a long burn-in with few samples).
    rr.integrator.setValue("maximum_num_steps", 200_000)
    rr.GF = float(GF)
    return rr


def state_ids(rr) -> list[str]:
    """Ordered names of the ODE state vector.

    ``getRatesOfChange()`` returns derivatives in exactly this order; the
    notebook asserts that against finite differences.
    """
    return list(rr.getFloatingSpeciesIds())


def mrna_indices(rr) -> np.ndarray:
    """Column indices of the mRNA species inside the full state vector."""
    ids = state_ids(rr)
    return np.array([ids.index(f"mRNA_{g}") for g in MRNA_GENES], dtype=int)


# --------------------------------------------------------------------------- #
# 3. Burn-in and period estimation
# --------------------------------------------------------------------------- #
def burn_in(rr, t_end: float = 200.0, n_points: int = 20_001) -> np.ndarray:
    """Integrate from the initial condition onto the attractor and discard.

    Returns the transient (useful only for the diagnostic that it *has* decayed).
    The RoadRunner instance is left sitting at the final state, so subsequent
    calls to ``rr.simulate(...)`` continue from the attractor.
    """
    return np.asarray(rr.simulate(0.0, t_end, n_points))


def estimate_period(rr, marker: str = "Mb", t_span: float = 300.0,
                    n_points: int = 300_001) -> float:
    """Estimate the limit-cycle period by successive upward mean-crossings.

    Zero-crossing (mean-crossing) detection is used rather than peak picking:
    it is a linear interpolation between two grid points, so it is accurate to
    O(h^2) even when the peak itself is a stiff spike that the output grid
    resolves poorly.

    The RoadRunner state is left where the probe simulation ended, which is
    still on the attractor, so this is safe to call between sampling runs.
    """
    ids = state_ids(rr)
    res = np.asarray(rr.simulate(0.0, t_span, n_points))
    t = res[:, 0]
    x = res[:, 1 + ids.index(marker)]
    level = 0.5 * (x.max() + x.min())

    up = np.where((x[:-1] < level) & (x[1:] >= level))[0]
    if len(up) < 3:
        raise RuntimeError(
            f"Only {len(up)} crossings found for '{marker}' - the system does "
            "not appear to be oscillating. Check that GF is above threshold."
        )
    # linear interpolation for a sub-grid crossing time
    frac = (level - x[up]) / (x[up + 1] - x[up])
    t_cross = t[up] + frac * (t[up + 1] - t[up])
    # drop the first interval, which may still contain residual transient
    return float(np.mean(np.diff(t_cross[1:])))


# --------------------------------------------------------------------------- #
# 4. Sampling the stationary distribution on the limit cycle
# --------------------------------------------------------------------------- #
def sample_limit_cycle(rr, n_cells: int, period: float, n_periods: int = 8,
                       grid_factor: int = 40, seed: int = 0):
    """Draw ``n_cells`` states uniformly in time from the attractor.

    Why uniform-in-time?  An asynchronously cycling, non-growing population at
    steady state occupies each point of the cycle in proportion to the time it
    spends there.  For a deterministic flow x' = f(x) on a closed orbit, that
    invariant (stationary) density is

        rho(s) proportional to 1 / |f(x(s))|          (s = arc length)

    which is exactly the push-forward of the uniform measure on [0, T) under
    t -> x(t).  So sampling uniform times *is* sampling the stationary
    distribution - no importance reweighting needed.  Points therefore pile up
    in the slow parts of the cycle (G1) and thin out across the fast mitotic
    spike; that non-uniform density is a feature of the ground truth, and is
    precisely the confound an OT/TDA method has to disentangle from the flow.

    Sampling spans an integer number of periods so that the empirical phase
    distribution is exactly uniform (no partial-cycle bias).

    Returns
    -------
    t : (n_cells,) float
        Sample times, sorted, measured from the start of the sampling window.
    X_full : (n_cells, n_states) float
        Noiseless state on the limit cycle.
    ids : list[str]
        Names of the state variables (columns of ``X_full``).
    """
    rng = np.random.default_rng(seed)
    t_end = n_periods * period
    n_grid = int(grid_factor * n_cells) + 1

    res = np.asarray(rr.simulate(0.0, t_end, n_grid))
    ids = state_ids(rr)

    # Uniformly at random *without replacement* from a dense uniform-in-time
    # grid == i.i.d. uniform times up to a grid spacing of T*n_periods/n_grid
    # (here ~5e-3 h), which is far below any timescale in the model.
    idx = np.sort(rng.choice(res.shape[0] - 1, size=n_cells, replace=False))
    return res[idx, 0], res[idx, 1:], ids


def exact_rates(rr, X_full: np.ndarray) -> np.ndarray:
    """Evaluate the exact right-hand side f(x) of the ODE at each supplied state.

    This is the ground-truth vector field: no finite differences, no
    interpolation.  We push each state into the compiled model and read
    ``getRatesOfChange()``, whose ordering matches ``getFloatingSpeciesIds()``.
    """
    F = np.empty_like(X_full)
    for i, x in enumerate(X_full):
        rr.model.setFloatingSpeciesConcentrations(x)
        F[i] = rr.getRatesOfChange()
    return F


# --------------------------------------------------------------------------- #
# 5. Observation model
# --------------------------------------------------------------------------- #
def add_gaussian_noise(M: np.ndarray, noise_frac: float = 0.05,
                       isotropic: bool = True, clip_negative: bool = False,
                       seed: int = 0):
    """Add i.i.d. Gaussian measurement noise to the noiseless mRNA matrix.

    Parameters
    ----------
    noise_frac
        Noise scale relative to the spread of the data.
    isotropic
        If True (default) a single sigma is used for every gene, set to
        ``noise_frac`` times the root-mean-square deviation of the whole
        centred matrix.  Isotropy is deliberate: downstream optimal-transport
        and persistent-homology machinery uses Euclidean distances, and
        per-gene sigmas would turn the noise ball into an ellipsoid, silently
        coupling the geometry of the estimate to the arbitrary units of each
        gene.  Set ``isotropic=False`` to get per-gene sigma instead.
    clip_negative
        Left False by default.  Clipping is biologically tidier but it censors
        the Gaussian at zero, which breaks the "limit cycle plus isotropic
        Gaussian tube" model that the recovery method is being tested against.

    Returns
    -------
    X_noisy, sigma
    """
    rng = np.random.default_rng(seed)
    Mc = M - M.mean(axis=0, keepdims=True)
    if isotropic:
        sigma = noise_frac * float(np.sqrt((Mc ** 2).mean()))
        sigma = np.full(M.shape[1], sigma)
    else:
        sigma = noise_frac * Mc.std(axis=0)
    X = M + rng.normal(0.0, 1.0, size=M.shape) * sigma
    if clip_negative:
        X = np.clip(X, 0.0, None)
    return X, sigma


def assign_cycle_phase(phase: np.ndarray, marks: dict) -> np.ndarray:
    """Label each cell G1 / S / G2 / M from its position on the cycle.

    ``marks`` maps the four landmark transitions to phases in [0, 1):
    ``G1/S`` (peak of cyclin E/Cdk2), ``S/G2`` (peak of cyclin A/Cdk2),
    ``G2/M`` (peak of cyclin B/Cdk1) and ``M/G1`` (peak of active Cdc20,
    i.e. mitotic exit).  This is an operational definition tied to the model's
    own landmarks, not a claim about real phase durations.
    """
    order = ["G1/S", "S/G2", "G2/M", "M/G1"]
    labels_after = {"G1/S": "S", "S/G2": "G2", "G2/M": "M", "M/G1": "G1"}

    cuts = np.array([marks[k] for k in order])
    srt = np.argsort(cuts)                       # landmarks in increasing phase
    cuts_s = cuts[srt]
    names_s = [labels_after[order[k]] for k in srt]

    # searchsorted -> index of the landmark immediately *before* this phase;
    # "% 4" wraps the arc that straddles phase 0.
    idx = (np.searchsorted(cuts_s, phase, side="right") - 1) % 4
    return np.array([names_s[k] for k in idx], dtype=object)


# --------------------------------------------------------------------------- #
# 6. Diagnostics
# --------------------------------------------------------------------------- #
def embedding_margin(M_cycle: np.ndarray, frac: float = 0.05) -> float:
    """Ratio (minimum non-local distance) / (diameter) for a closed curve.

    ``M_cycle`` must be one full period, ordered in time.  Points whose phase
    separation is below ``frac`` of a period are excluded (they are supposed to
    be close).  A value clearly above 0 certifies that the projection of the
    limit cycle into these coordinates is an *embedding*: the observed data
    really is a simple closed curve, not a self-intersecting shadow of one.
    This matters because a self-intersection would make the "true" vector field
    multi-valued as a function of the observed coordinates.
    """
    n = len(M_cycle)
    D = np.linalg.norm(M_cycle[:, None, :] - M_cycle[None, :, :], axis=-1)
    i = np.arange(n)
    circ = np.minimum(np.abs(i[:, None] - i[None, :]), n - np.abs(i[:, None] - i[None, :]))
    return float(D[circ > n * frac].min() / D.max())


def gf_scan(antimony: str, gf_values, t_end: float = 2000.0,
            n_points: int = 20_001, frac_tail: float = 0.2) -> np.ndarray:
    """Peak-to-trough amplitude of cyclin B/Cdk1 as a function of growth factor.

    A cheap numerical stand-in for a bifurcation diagram: it localises the
    growth-factor threshold above which the quiescent steady state gives way to
    large-amplitude Cdk oscillations.
    """
    amps = []
    for gf in gf_values:
        rr = build_model(antimony, GF=gf)
        ids = state_ids(rr)
        res = np.asarray(rr.simulate(0.0, t_end, n_points))
        tail = res[res[:, 0] > (1 - frac_tail) * t_end, 1 + ids.index("Mb")]
        amps.append(tail.max() - tail.min())
    return np.array(amps)
