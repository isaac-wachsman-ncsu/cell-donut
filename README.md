# velocity_ot

Learn an **autonomous velocity field** `v_φ : ℝ^D → ℝ^D` whose flow matches an
underlying **cyclic dynamical system**, from a single snapshot of that system's
**stationary distribution**. Built on PyTorch and [POT](https://pythonot.github.io/)
(Python Optimal Transport), with first-class [`anndata`](https://anndata.readthedocs.io/)
integration.

The data are assumed to be i.i.d. samples from the stationary (time-invariant)
distribution of a system whose state cycles with period `T`. We recover the
per-point velocity that generates that cycle.

## Installation
Using conda is the preferred way to install dependencies for this package.
To install the velocity_ot package, navigate to your desired installation location in a terminal and run the following commands:

```bash
git clone https://github.com/isaac-wachsman-ncsu/cell-donut.git
conda create --name cell-donut python=3.10.20
conda activate cell-donut
pip install -r requirements.txt
```

Drop the `velocity_ot/` package on your `PYTHONPATH` and `import velocity_ot`.

## Quick start

```python
import velocity_ot as vo

est = vo.VelocityFieldEstimator(intrinsic_dim=1)   # 1 -> data live on a circle
est.fit(adata, spatial_key="X_spatial", theta_key="theta", n_epochs=200)

adata.obsm["velocity_field"]   # fitted velocities, shape [N, D]
est.predict(new_points)        # evaluate v_φ at arbitrary out-of-sample points
```

## AnnData contract

| Role | Location | Shape |
|------|----------|-------|
| Coordinates `x` (input) | `adata.obsm[spatial_key]` | `[N, D]` |
| Circular coordinate `θ` (input, radians) | `adata.obsm[theta_key]` | `[N]` or `[N, 1]` |
| Fitted velocity field (output) | `adata.obsm[velocity_key]` | `[N, D]` |
| Estimated `∇θ` (output) | `adata.obsm[grad_theta_out_key]` | `[N, D]` |
| Loss history + config (output) | `adata.uns["velocity_ot"]` | dict |

## The objective

The field minimises a weighted sum of four terms:

```
L = λ₁·L_KE + λ₂·L_OT_global + λ₃·L_align + λ₄·L_OT_sub
```

- **`L_KE` — kinetic energy.** The mean action `∫₀ᵀ ½‖v‖² dt` accumulated as
  points flow through one cycle (trapezoidal rule over the ODE trajectory).
  Regularises toward the gentlest flow.
- **`L_OT_global` — stationarity.** Sinkhorn divergence between the initial
  cloud `X₀` and the cloud evolved over one full cycle `X_T`. Since the
  distribution is stationary, `X₀ ≈ X_T`.
- **`L_align` — angular alignment.** `1 − cos(v_φ(x), ∇θ(x))`, averaged over
  points. Fixes the *rotational direction*; scale-invariant. `∇θ` is estimated
  from the circular coordinate by `circular_gradient.py` (local weighted
  least-squares on wrapped edge differences).
- **`L_OT_sub` — temporal anchor.** Optional. For a labelled sub-population with
  known targets at an intermediate time `t'`, the Sinkhorn divergence between
  the model-evolved sub-population and those targets. Ties latent time to
  physical cycle time and thereby fixes the field's *speed*.

Alignment and global-OT together pin down direction and shape but leave the
speed loosely determined; supply a sub-population (`sub_source`, `sub_target`,
`sub_time`) to anchor the physical speed.

## Plotting

`velocity_ot.plotting` visualises a fitted estimator (or a bare velocity
module):

```python
import velocity_ot as vo

vo.plot_loss_history(est)                                  # training curves
vo.plot_velocity_field(est, adata.obsm["X_spatial"])       # streamplot / quiver
vo.plot_trajectories(est, X=adata.obsm["X_spatial"], T=1.0, n_steps=120)

# raw trajectory array for your own analysis: [n_steps+1, n_seeds, D]
traj, times = vo.integrate_trajectories(est, seeds, T=1.0, n_steps=100)
```

Trajectories are integrated forward through the autonomous field and coloured
by time (start ○ → end ★). The styling is adapted from CytoBridge's `plot.py`.

## Demo notebook

`velocity_ot_demo.ipynb` runs the whole pipeline — load data, fit, inspect the
field, integrate and plot trajectories — on a synthetic rotating ring, with a
clearly marked cell for plugging in your own `AnnData`.

## Module layout

- `models.py` — `VelocityNet`, a configurable MLP `ℝ^D → ℝ^D` (arbitrary `D`,
  choice of activation, optional layer-norm and residuals).
- `dynamics.py` — `integrate` / `ODEIntegrator`, fixed-step Euler or RK4 that is
  fully differentiable and also returns node velocities for the KE term.
- `losses.py` — the four loss terms and a numerically robust, differentiable
  `sinkhorn_divergence` (log-domain solver via POT's PyTorch backend).
- `solver.py` — `VelocityFieldEstimator`, the `.fit(adata, ...)` training loop
  (Adam/AdamW), plus a runnable example under `if __name__ == "__main__"`.
- `plotting.py` — velocity-field, trajectory and loss-curve plots.
- `circular_gradient.py` — the provided `∇θ` estimator (reused unchanged).

## Notes on numerics

Entropic OT is computed with POT's **log-stabilised** Sinkhorn
(`method="sinkhorn_log"`), and cost matrices are rescaled to an `O(1)` scale so
the regularisation `reg` behaves consistently. This avoids the primal-domain
underflow that otherwise makes small-`reg` Sinkhorn silently return zero.

Everything is **device-agnostic**: pass `device="cuda"` (or leave `device=None`
to auto-select) and all tensors and modules move accordingly.

## Run the example

```bash
python -m velocity_ot.solver
```

Fits a rotating-ring toy for 5 epochs and prints that the learned field is
rotational (`cos(v, +θ) ≈ 1`).
