"""velocity_ot: optimal-transport velocity fields for cyclic dynamical systems.

Learn an autonomous velocity field :math:`v_\\phi : \\mathbb{R}^D \\to \\mathbb{R}^D`
whose flow reproduces the stationary distribution of a cyclic system, from a
single-time-point sample stored in an :class:`anndata.AnnData` object.

Quick start
-----------
>>> from velocity_ot import VelocityFieldEstimator
>>> est = VelocityFieldEstimator(intrinsic_dim=1)
>>> est.fit(adata, spatial_key="X_spatial", theta_key="theta", n_epochs=200)
>>> adata.obsm["velocity_field"]   # fitted velocities, shape [N, D]
"""

from __future__ import annotations

from . import circular_gradient, dynamics, losses, models, plotting, solver
from .dynamics import IntegrationResult, ODEIntegrator, integrate
from .plotting import (
    integrate_trajectories,
    plot_loss_history,
    plot_trajectories,
    plot_velocity_field,
)
from .losses import (
    angular_alignment_loss,
    cycle_consistency_loss,
    kinetic_energy_loss,
    sinkhorn_divergence,
    stationarity_loss,
)
from .models import ACTIVATION_FN, VelocityNet
from .solver import VelocityFieldEstimator

__version__ = "0.1.0"

__all__ = [
    # high-level API
    "VelocityFieldEstimator",
    # models
    "VelocityNet",
    "ACTIVATION_FN",
    # dynamics
    "integrate",
    "ODEIntegrator",
    "IntegrationResult",
    # losses
    "sinkhorn_divergence",
    "kinetic_energy_loss",
    "stationarity_loss",
    "angular_alignment_loss",
    "cycle_consistency_loss",
    # plotting
    "integrate_trajectories",
    "plot_velocity_field",
    "plot_trajectories",
    "plot_loss_history",
    # submodules
    "models",
    "dynamics",
    "losses",
    "solver",
    "circular_gradient",
]
