"""Plotting utilities for fitted velocity fields.

Visual helpers to inspect a trained :class:`velocity_ot.VelocityFieldEstimator`
(or a bare velocity :class:`torch.nn.Module`):

* :func:`plot_velocity_field` — quiver / streamplot of :math:`v_\\phi`.
* :func:`integrate_trajectories` — roll the autonomous field forward from seed
  points and return the trajectories as numpy arrays.
* :func:`plot_trajectories` — draw those trajectories, coloured by time, over
  the data cloud.
* :func:`plot_loss_history` — training-curve diagnostics.

The styling (viridis time-colouring, optional spline smoothing, clean seaborn
axes) is adapted from CytoBridge's ``plot.py`` ``plot_ode`` helper, simplified
for a single stationary distribution and an autonomous field.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

try:  # optional, only used for trajectory smoothing
    from scipy.interpolate import make_interp_spline

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False

from .dynamics import integrate

__all__ = [
    "integrate_trajectories",
    "plot_velocity_field",
    "plot_trajectories",
    "plot_loss_history",
]


# --------------------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------------------- #
def _apply_style() -> None:
    """Apply a clean, light plotting style (best-effort)."""
    try:
        import seaborn as sns

        sns.set_style("white")
    except Exception:  # pragma: no cover
        pass
    plt.rcParams.update(
        {
            "axes.facecolor": "white",
            "axes.edgecolor": "lightgrey",
            "axes.grid": False,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "font.size": 12,
        }
    )


def _as_velocity(model: Any) -> tuple[torch.nn.Module, torch.device]:
    """Resolve ``model`` to ``(velocity_module, device)``.

    Accepts either a :class:`velocity_ot.VelocityFieldEstimator` (any object
    exposing a ``.model`` attribute that is an ``nn.Module``) or a bare velocity
    ``nn.Module``.
    """
    inner = getattr(model, "model", None)
    if isinstance(inner, torch.nn.Module):
        device = getattr(model, "device", next(inner.parameters()).device)
        return inner, torch.device(device)
    if isinstance(model, torch.nn.Module):
        return model, next(model.parameters()).device
    raise TypeError(
        "`model` must be a VelocityFieldEstimator or a torch.nn.Module velocity field."
    )


@torch.no_grad()
def _eval_field(module: torch.nn.Module, device: torch.device, pts: np.ndarray) -> np.ndarray:
    """Evaluate the velocity field at numpy points ``[Q, D]`` -> ``[Q, D]``."""
    was_training = module.training
    module.eval()
    x = torch.as_tensor(np.asarray(pts, dtype=np.float32), device=device)
    v = module(x).cpu().numpy()
    if was_training:
        module.train()
    return v


def _resolve_seeds(
    seeds: np.ndarray | None,
    X: np.ndarray | None,
    n_seeds: int,
    dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Pick seed points for trajectory integration."""
    if seeds is not None:
        return np.asarray(seeds, dtype=np.float32)
    if X is None:
        raise ValueError("Provide either `seeds` or `X` to seed trajectories from.")
    X = np.asarray(X, dtype=np.float32)
    idx = rng.choice(X.shape[0], size=min(n_seeds, X.shape[0]), replace=False)
    return X[idx]


# --------------------------------------------------------------------------- #
#  Trajectory integration
# --------------------------------------------------------------------------- #
def integrate_trajectories(
    model: Any,
    seeds: np.ndarray,
    T: float = 1.0,
    n_steps: int = 100,
    method: str = "rk4",
) -> tuple[np.ndarray, np.ndarray]:
    """Roll the autonomous field forward from ``seeds``.

    Args:
        model: A :class:`velocity_ot.VelocityFieldEstimator` or velocity module.
        seeds: Initial points of shape ``[n, D]``.
        T: End time (integrate over ``[0, T]``).
        n_steps: Number of integration steps.
        method: ``"euler"`` or ``"rk4"``.

    Returns:
        ``(trajectory, times)`` where ``trajectory`` has shape
        ``[n_steps + 1, n, D]`` and ``times`` has shape ``[n_steps + 1]``, both
        as numpy arrays.
    """
    module, device = _as_velocity(model)
    x0 = torch.as_tensor(np.asarray(seeds, dtype=np.float32), device=device)
    if x0.dim() != 2:
        raise ValueError(f"`seeds` must be [n, D], got {tuple(x0.shape)}.")
    with torch.no_grad():
        result = integrate(module, x0, t0=0.0, t1=T, n_steps=n_steps, method=method)
    return result.trajectory.cpu().numpy(), result.times.cpu().numpy()


# --------------------------------------------------------------------------- #
#  Velocity-field plot
# --------------------------------------------------------------------------- #
def plot_velocity_field(
    model: Any,
    X: np.ndarray,
    dims: tuple[int, int] = (0, 1),
    quiver: bool = True,
    streamplot: bool = True,
    n_grid: int = 22,
    margin: float = 0.1,
    ax: plt.Axes | None = None,
    point_color: str = "0.55",
    cmap: str = "viridis",
    title: str | None = "Velocity field",
) -> plt.Axes:
    """Plot the fitted velocity field over the data cloud.

    Draws the data points and, for two-dimensional coordinates, a streamplot on
    a regular grid and/or a quiver of the field sampled at the data points.

    Args:
        model: A :class:`velocity_ot.VelocityFieldEstimator` or velocity module.
        X: Data coordinates ``[N, D]``.
        dims: Which two coordinate axes to display.
        quiver: Draw arrows of the field at the data points.
        streamplot: Draw a streamplot on a grid (2-D fields only).
        n_grid: Grid resolution per axis for the streamplot.
        margin: Fractional padding around the data bounding box.
        ax: Existing axes to draw on; a new figure is created if ``None``.
        point_color: Colour of the background data points.
        cmap: Colormap used to shade streamlines / arrows by speed.
        title: Axes title (or ``None``).

    Returns:
        The matplotlib axes containing the plot.
    """
    module, device = _as_velocity(model)
    X = np.asarray(X, dtype=np.float32)
    i, j = dims

    if ax is None:
        _apply_style()
        _, ax = plt.subplots(figsize=(7, 6), dpi=110)

    ax.scatter(X[:, i], X[:, j], s=12, c=point_color, alpha=0.35, linewidths=0, zorder=1)

    is_2d = X.shape[1] == 2
    if streamplot and is_2d:
        x_min, x_max = X[:, i].min(), X[:, i].max()
        y_min, y_max = X[:, j].min(), X[:, j].max()
        dx, dy = (x_max - x_min) * margin, (y_max - y_min) * margin
        gx = np.linspace(x_min - dx, x_max + dx, n_grid)
        gy = np.linspace(y_min - dy, y_max + dy, n_grid)
        GX, GY = np.meshgrid(gx, gy)
        grid = np.stack([GX.ravel(), GY.ravel()], axis=1)
        V = _eval_field(module, device, grid)
        U = V[:, 0].reshape(GX.shape)
        W = V[:, 1].reshape(GX.shape)
        speed = np.sqrt(U**2 + W**2)
        ax.streamplot(
            GX, GY, U, W, color=speed, cmap=cmap, density=1.2, linewidth=1.0, arrowsize=1.0, zorder=2
        )
    elif streamplot and not is_2d:
        # No meaningful full-field streamplot in >2-D; fall back to quiver.
        quiver = True

    if quiver:
        V = _eval_field(module, device, X)
        ax.quiver(
            X[:, i], X[:, j], V[:, i], V[:, j],
            np.linalg.norm(V, axis=1),
            cmap=cmap, angles="xy", alpha=0.7, width=0.003, zorder=3,
        )

    ax.set_aspect("equal", adjustable="datalim")
    if title:
        ax.set_title(title)
    ax.set_xlabel(f"dim {i}")
    ax.set_ylabel(f"dim {j}")
    return ax


# --------------------------------------------------------------------------- #
#  Trajectory plot
# --------------------------------------------------------------------------- #
def plot_trajectories(
    model: Any,
    seeds: np.ndarray | None = None,
    X: np.ndarray | None = None,
    n_seeds: int = 30,
    T: float = 1.0,
    n_steps: int = 100,
    method: str = "rk4",
    dims: tuple[int, int] = (0, 1),
    smooth: bool = False,
    show_background: bool = True,
    ax: plt.Axes | None = None,
    cmap: str = "viridis",
    linewidth: float = 1.3,
    alpha: float = 0.9,
    seed: int | None = 0,
    title: str | None = "Integrated trajectories",
) -> plt.Axes:
    """Integrate trajectories from seed points and plot them, coloured by time.

    Args:
        model: A :class:`velocity_ot.VelocityFieldEstimator` or velocity module.
        seeds: Explicit seed points ``[n, D]``. If ``None``, ``n_seeds`` points
            are drawn from ``X``.
        X: Data cloud, used both to draw the background and (if ``seeds`` is
            ``None``) to sample seeds.
        n_seeds: Number of seeds to sample when ``seeds`` is ``None``.
        T: Integration horizon (e.g. one cycle).
        n_steps: Number of integration steps.
        method: ``"euler"`` or ``"rk4"``.
        dims: Which two coordinate axes to display.
        smooth: Cubic-spline smoothing of each trajectory (needs SciPy).
        show_background: Scatter the data cloud behind the trajectories.
        ax: Existing axes to draw on; a new figure is created if ``None``.
        cmap: Colormap used to encode time along each trajectory.
        linewidth: Trajectory line width.
        alpha: Trajectory line opacity.
        seed: RNG seed for reproducible seed sampling.
        title: Axes title (or ``None``).

    Returns:
        The matplotlib axes containing the plot.
    """
    module, device = _as_velocity(model)
    rng = np.random.default_rng(seed)
    dim = (X.shape[1] if X is not None else np.asarray(seeds).shape[1])
    seed_pts = _resolve_seeds(seeds, X, n_seeds, dim, rng)

    traj, times = integrate_trajectories(module, seed_pts, T=T, n_steps=n_steps, method=method)
    # traj: [K+1, n, D]
    i, j = dims
    K1, n, _ = traj.shape

    if ax is None:
        _apply_style()
        _, ax = plt.subplots(figsize=(7, 6), dpi=110)

    if show_background and X is not None:
        Xb = np.asarray(X, dtype=np.float32)
        ax.scatter(Xb[:, i], Xb[:, j], s=12, c="0.7", alpha=0.3, linewidths=0, zorder=1)

    norm = plt.Normalize(times.min(), times.max())
    colormap = plt.get_cmap(cmap)

    for p in range(n):
        xs = traj[:, p, i]
        ys = traj[:, p, j]
        ts = times
        if smooth and _HAVE_SCIPY and K1 >= 4:
            t_old = np.linspace(0.0, 1.0, K1)
            t_new = np.linspace(0.0, 1.0, 300)
            xs = make_interp_spline(t_old, xs, k=3)(t_new)
            ys = make_interp_spline(t_old, ys, k=3)(t_new)
            ts = np.linspace(times.min(), times.max(), 300)
        pts = np.stack([xs, ys], axis=1).reshape(-1, 1, 2)
        segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segments, cmap=colormap, norm=norm, zorder=2)
        lc.set_array(0.5 * (ts[:-1] + ts[1:]))
        lc.set_linewidth(linewidth)
        lc.set_alpha(alpha)
        ax.add_collection(lc)

    # Start (o) and end (*) markers.
    ax.scatter(traj[0, :, i], traj[0, :, j], s=28, facecolors="none",
               edgecolors="black", linewidths=0.8, zorder=4, label="start")
    ax.scatter(traj[-1, :, i], traj[-1, :, j], s=45, marker="*",
               c="black", zorder=4, label="end")

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("time")

    ax.set_aspect("equal", adjustable="datalim")
    ax.autoscale_view()
    if title:
        ax.set_title(title)
    ax.set_xlabel(f"dim {i}")
    ax.set_ylabel(f"dim {j}")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    return ax


# --------------------------------------------------------------------------- #
#  Loss curves
# --------------------------------------------------------------------------- #
def plot_loss_history(
    history: Any,
    ax: plt.Axes | None = None,
    title: str | None = "Training losses",
) -> plt.Axes:
    """Plot per-epoch loss curves.

    Args:
        history: A loss-history ``dict`` mapping term name -> list of values, a
            :class:`velocity_ot.VelocityFieldEstimator` (uses ``.history``), or
            an :class:`anndata.AnnData` (uses ``uns["velocity_ot"]["history"]``).
        ax: Existing axes to draw on; a new figure is created if ``None``.
        title: Axes title (or ``None``).

    Returns:
        The matplotlib axes containing the plot.
    """
    if hasattr(history, "history") and isinstance(history.history, dict):
        curves = history.history
    elif hasattr(history, "uns"):
        curves = history.uns["velocity_ot"]["history"]
    elif isinstance(history, dict):
        curves = history
    else:
        raise TypeError("Unrecognised `history` argument.")

    if ax is None:
        _apply_style()
        _, ax = plt.subplots(figsize=(7, 4.5), dpi=110)

    for name, values in curves.items():
        values = np.asarray(values, dtype=float)
        ax.plot(np.arange(len(values)), values, label=name, linewidth=1.6)

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    return ax
