"""Limit-cycle manifold estimation and the on-cycle target velocity.

This module estimates the 1-D limit cycle :math:`\\Gamma(\\theta)` underlying a
cyclic point cloud and the velocity field *on* that cycle implied by
steady-state mass conservation. It provides:

* a direct MSE supervision target for the learned field on the cycle
  (the ``L_lcycle`` loss in :mod:`velocity_ot.solver`), and
* the transverse noise scale :math:`\\sigma` consumed by the noise-matched
  stationarity loss.

Model
-----
Data are assumed to be samples of a 1-D limit cycle plus noise,
``x_i = Gamma(theta_i) + eta_i``. We

1. estimate the mean cycle ``Gamma(theta)`` as a **periodic** cubic spline
   through angularly-binned means of the data (exact ``2*pi`` periodicity,
   robust to empty bins);
2. estimate the stationary angular density ``rho(theta)`` with a circular
   (wrapped-Gaussian) kernel density estimate;
3. set the on-cycle velocity from steady-state mass conservation. For a 1-D
   cycle the stationary flux ``rho(theta) * dtheta/dt`` is constant along the
   loop, so the angular speed is *inversely proportional to the density*,

   .. math::  \\frac{d\\theta}{dt} = \\frac{c}{\\rho(\\theta)}, \\qquad
              v(\\theta) = \\frac{d\\Gamma}{d\\theta}\\,\\frac{c}{\\rho(\\theta)},

   with the single constant ``c`` fixed so that one loop takes time ``T``:
   ``c = (∮ rho dtheta) / T``. This normalisation is always well defined
   (the angular integral of a density is finite and positive);
4. estimate the transverse noise ``sigma`` from the residuals
   ``x_i - Gamma(theta_i)``.

Everything here depends only on ``numpy`` / ``scipy`` so it runs without the
Torch stack, exactly like :mod:`velocity_ot.graph_ot`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LimitCycle:
    """Estimated limit cycle and the target velocity field on it.

    Attributes:
        theta_grid: Angular grid ``[M]`` on ``[0, 2*pi)`` the cycle is sampled at.
        gamma: Cycle points ``Gamma(theta_grid)`` of shape ``[M, D]``.
        tangent: Angular derivative ``dGamma/dtheta`` of shape ``[M, D]``.
        velocity: Target on-cycle velocity ``[M, D]`` (one loop per ``T``).
        speed: Target ambient speed ``||velocity||`` of shape ``[M]``.
        density: Angular density ``rho(theta_grid)`` (floored) of shape ``[M]``.
        sigma: Per-dimension transverse noise std of shape ``[D]``.
        period_time: The loop time the velocity is normalised to (``== T``).
        bandwidth: Kernel bandwidth used for the density estimate.
    """

    theta_grid: np.ndarray
    gamma: np.ndarray
    tangent: np.ndarray
    velocity: np.ndarray
    speed: np.ndarray
    density: np.ndarray
    sigma: np.ndarray
    period_time: float
    bandwidth: float


_TWO_PI = 2.0 * np.pi


def _wrap(x: np.ndarray) -> np.ndarray:
    """Wrap angles (radians) into the principal branch ``(-pi, pi]``."""
    return (np.asarray(x, dtype=float) + np.pi) % _TWO_PI - np.pi


def _auto_bandwidth(theta: np.ndarray, factor: float = 3.0) -> float:
    """Circular KDE bandwidth: ``factor`` x median wrapped angular spacing."""
    st = np.sort(np.mod(np.asarray(theta, dtype=float), _TWO_PI))
    gaps = np.diff(np.concatenate([st, st[:1] + _TWO_PI]))
    med = float(np.median(gaps)) if gaps.size else _TWO_PI / 64.0
    return max(factor * med, 1e-3)


def _fill_circular(centers: np.ndarray, means: np.ndarray) -> np.ndarray:
    """Fill empty (NaN) angular bins by circular linear interpolation."""
    out = means.copy()
    valid = ~np.isnan(means[:, 0])
    if valid.all():
        return out
    if int(valid.sum()) < 2:
        raise ValueError(
            "Too few non-empty angular bins to build the cycle; reduce `n_bins`."
        )
    xp = centers[valid]
    for d in range(means.shape[1]):
        out[:, d] = np.interp(centers, xp, means[valid, d], period=_TWO_PI)
    return out


def _periodic_mean_spline(theta: np.ndarray, X: np.ndarray, n_bins: int):
    """Periodic cubic spline through angularly-binned means of ``X``.

    Returns a vector-valued :class:`scipy.interpolate.CubicSpline` with
    ``bc_type='periodic'`` so ``Gamma(theta + 2*pi) == Gamma(theta)``.
    """
    from scipy.interpolate import CubicSpline

    th = np.mod(theta, _TWO_PI)
    edges = np.linspace(0.0, _TWO_PI, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip((th / _TWO_PI * n_bins).astype(int), 0, n_bins - 1)

    D = X.shape[1]
    means = np.full((n_bins, D), np.nan)
    for k in range(n_bins):
        m = bin_idx == k
        if np.any(m):
            means[k] = X[m].mean(axis=0)
    filled = _fill_circular(centers, means)

    # Close the loop so the spline is exactly periodic.
    nodes = np.concatenate([centers, centers[:1] + _TWO_PI])
    vals = np.concatenate([filled, filled[:1]], axis=0)
    return CubicSpline(nodes, vals, bc_type="periodic", axis=0)


def _circular_kde(theta: np.ndarray, grid: np.ndarray, bandwidth: float) -> np.ndarray:
    """Wrapped-Gaussian KDE of the angular density on ``grid``."""
    diff = _wrap(grid[:, None] - theta[None, :])          # [M, N]
    k = np.exp(-0.5 * (diff / bandwidth) ** 2)
    return k.sum(axis=1) / (theta.shape[0] * bandwidth * np.sqrt(_TWO_PI))


def estimate_limit_cycle(
    X: np.ndarray,
    theta: np.ndarray,
    T: float = 1.0,
    n_grid: int = 256,
    n_bins: int = 64,
    bandwidth: float | None = None,
    density_floor: float = 0.1,
) -> LimitCycle:
    """Estimate the limit cycle ``Gamma(theta)`` and its on-cycle velocity.

    Args:
        X: Coordinates ``[N, D]`` (the fit space).
        theta: Circular coordinate ``[N]`` (radians).
        T: Cycle length; the velocity is normalised so one loop takes this time.
        n_grid: Number of angular samples ``M`` of the returned cycle.
        n_bins: Angular bins used to build the mean-cycle spline.
        bandwidth: Circular-KDE bandwidth (radians); auto-set from the angular
            spacing when ``None``.
        density_floor: Floor applied to the density as a fraction of its median,
            bounding the ``1/rho`` speed in sparse regions.

    Returns:
        A :class:`LimitCycle`.
    """
    X = np.asarray(X, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    n = theta.shape[0]
    n_bins = int(min(int(n_bins), max(4, n // 2)))

    cs = _periodic_mean_spline(theta, X, n_bins)
    grid = np.linspace(0.0, _TWO_PI, n_grid, endpoint=False)
    gamma = np.asarray(cs(grid))                     # [M, D]
    tangent = np.asarray(cs(grid, 1))                # [M, D]  dGamma/dtheta

    bw = _auto_bandwidth(theta) if bandwidth is None else float(bandwidth)
    rho = _circular_kde(theta, grid, bw)
    rho = np.maximum(rho, density_floor * np.median(rho))  # bound 1/rho

    # One loop in time T:  loop_time = ∮ rho/c dtheta = (2*pi*mean(rho))/c = T.
    c = (_TWO_PI * float(np.mean(rho))) / max(T, 1e-12)
    dtheta_dt = c / rho                              # [M]
    velocity = tangent * dtheta_dt[:, None]          # [M, D]
    speed = np.linalg.norm(velocity, axis=1)         # [M]

    # Transverse noise from residuals at the data angles.
    resid = X - np.asarray(cs(np.mod(theta, _TWO_PI)))
    sigma = resid.std(axis=0)                        # [D]

    return LimitCycle(
        theta_grid=grid,
        gamma=gamma.astype(np.float32),
        tangent=tangent.astype(np.float32),
        velocity=velocity.astype(np.float32),
        speed=speed.astype(np.float32),
        density=rho.astype(np.float32),
        sigma=sigma.astype(np.float32),
        period_time=float(T),
        bandwidth=float(bw),
    )
