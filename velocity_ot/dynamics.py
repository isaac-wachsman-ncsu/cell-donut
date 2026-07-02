"""ODE integration of an autonomous velocity field.

Given an autonomous field :math:`v_\\phi` and initial coordinates
:math:`x(0) = x_0`, we solve the initial-value problem

.. math::

    \\dot{x}(t) = v_\\phi(x(t)), \\qquad t \\in [t_0, t_1]

with a fixed-step explicit integrator (Euler or classical RK4). Every step is
a composition of differentiable PyTorch operations, so gradients flow from the
trajectory back to the network parameters :math:`\\phi` via backpropagation.

The integrator also returns the velocity evaluated at each trajectory node,
which the kinetic-energy loss consumes without a second forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

_METHODS = ("euler", "rk4")


@dataclass
class IntegrationResult:
    """Container for the output of :func:`integrate`.

    Attributes:
        endpoint: Final coordinates ``x(t_1)`` of shape ``[B, D]``.
        trajectory: Coordinates at every node, shape ``[K + 1, B, D]`` where
            ``K`` is the number of steps. ``trajectory[0]`` is ``x_0`` and
            ``trajectory[-1]`` is ``endpoint``.
        velocities: Velocity field evaluated at every trajectory node,
            ``velocities[k] = v(trajectory[k])``, shape ``[K + 1, B, D]``.
        times: The ``K + 1`` node times, shape ``[K + 1]``.
        dt: The (uniform) step size ``(t_1 - t_0) / K``.
    """

    endpoint: torch.Tensor
    trajectory: torch.Tensor
    velocities: torch.Tensor
    times: torch.Tensor
    dt: float


def _step(
    velocity: nn.Module,
    x: torch.Tensor,
    dt: float,
    method: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance ``x`` by one step and return ``(x_next, v_at_x)``.

    ``v_at_x = v(x)`` is returned so the caller can record the node velocity
    for free (for RK4 it is the ``k1`` slope that is computed anyway).
    """
    k1 = velocity(x)
    if method == "euler":
        x_next = x + dt * k1
    else:  # rk4
        k2 = velocity(x + 0.5 * dt * k1)
        k3 = velocity(x + 0.5 * dt * k2)
        k4 = velocity(x + dt * k3)
        x_next = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return x_next, k1


def integrate(
    velocity: nn.Module,
    x0: torch.Tensor,
    t0: float = 0.0,
    t1: float = 1.0,
    n_steps: int = 20,
    method: str = "rk4",
) -> IntegrationResult:
    """Integrate the autonomous field ``velocity`` from ``t0`` to ``t1``.

    Args:
        velocity: A module implementing ``v(x) -> [B, D]`` (e.g.
            :class:`velocity_ot.models.VelocityNet`).
        x0: Initial coordinates of shape ``[B, D]``.
        t0: Start time.
        t1: End time. May be smaller than a full cycle for partial evolution.
        n_steps: Number of fixed integration steps ``K``.
        method: ``"euler"`` or ``"rk4"``.

    Returns:
        An :class:`IntegrationResult` with the endpoint, full trajectory,
        node velocities, node times and step size.
    """
    if x0.dim() != 2:
        raise ValueError(f"Expected `x0` of shape [B, D], got {tuple(x0.shape)}.")
    if method not in _METHODS:
        raise ValueError(f"Unknown method '{method}'. Choose from {_METHODS}.")
    if n_steps < 1:
        raise ValueError(f"`n_steps` must be >= 1, got {n_steps}.")

    dt = (t1 - t0) / n_steps
    x = x0
    traj = [x]
    vels: list[torch.Tensor] = []

    for _ in range(n_steps):
        x, v_at_x = _step(velocity, x, dt, method)
        vels.append(v_at_x)
        traj.append(x)

    # Velocity at the terminal node (one extra evaluation) so that the
    # `velocities` and `trajectory` stacks are node-aligned and share length.
    vels.append(velocity(x))

    times = torch.linspace(t0, t1, n_steps + 1, device=x0.device, dtype=x0.dtype)
    return IntegrationResult(
        endpoint=x,
        trajectory=torch.stack(traj, dim=0),
        velocities=torch.stack(vels, dim=0),
        times=times,
        dt=float(dt),
    )


class ODEIntegrator:
    """Reusable, configured wrapper around :func:`integrate`.

    Stores the integration hyper-parameters so a training loop can call
    :meth:`__call__` repeatedly with a consistent scheme.

    Args:
        velocity: The velocity module to integrate.
        method: ``"euler"`` or ``"rk4"``.
        n_steps: Default number of steps for a full cycle ``[0, T]``.
        T: The cycle length ``T`` (end time of a full cycle).
    """

    def __init__(
        self,
        velocity: nn.Module,
        method: str = "rk4",
        n_steps: int = 20,
        T: float = 1.0,
    ) -> None:
        if method not in _METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from {_METHODS}.")
        self.velocity = velocity
        self.method = method
        self.n_steps = int(n_steps)
        self.T = float(T)

    def __call__(
        self,
        x0: torch.Tensor,
        t_end: float | None = None,
        n_steps: int | None = None,
    ) -> IntegrationResult:
        """Integrate from ``0`` to ``t_end`` (default: one full cycle ``T``).

        When ``t_end`` is a fraction of ``T`` the number of steps is scaled
        proportionally (rounded up) so the step size stays roughly constant,
        which keeps partial and full-cycle integrations consistent.

        Args:
            x0: Initial coordinates ``[B, D]``.
            t_end: End time. Defaults to the configured cycle length ``T``.
            n_steps: Override the number of steps. If ``None`` it is derived
                from ``t_end`` and the configured ``n_steps`` per full cycle.

        Returns:
            An :class:`IntegrationResult`.
        """
        t_end = self.T if t_end is None else float(t_end)
        if n_steps is None:
            frac = abs(t_end) / self.T if self.T != 0 else 1.0
            n_steps = max(1, int(round(self.n_steps * frac)))
        return integrate(
            self.velocity,
            x0,
            t0=0.0,
            t1=t_end,
            n_steps=n_steps,
            method=self.method,
        )
