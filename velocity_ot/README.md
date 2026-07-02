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

```bash
pip install torch POT anndata scipy numpy
```

Drop the `velocity_ot/` package on your `PYTHONPATH` and `import velocity_ot`.

## Quick start

```python
import velocity_ot as vo

est = vo.VelocityFieldEstimator(intrinsic_dim=1)   # 1 -> data live on a circle
est.fit(adata, n_epochs=200)                        # reads adata.X + adata.obs['circular_coords']

adata.obsm["velocity_field"]   # fitted velocities, shape [N, D]
est.predict(new_points)        # evaluate v_φ at arbitrary out-of-sample points
```

## AnnData contract

| Role | Location | Shape |
|------|----------|-------|
| Coordinates `x` (input) | `adata.X` (default; or `obsm[spatial_key]`) | `[N, D]` |
| Circular coordinate `θ` (input, radians) | `adata.obs['circular_coords']` (or `obsm`) | `[N]` |
| Fitted velocity field (output) | `adata.obsm[velocity_key]` | `[N, D]` |
| Estimated `∇θ` (output) | `adata.obsm[grad_theta_out_key]` | `[N, D]` |
| Loss history + config (output) | `adata.uns["velocity_ot"]` | dict |

Spatial coordinates are read from `adata.X` by default (pass `spatial_key="..."`
to read from `obsm` instead); the circular coordinate is read from
`adata.obs['circular_coords']` by default (`theta_key=` to change the name).

## The objective

The field minimises a weighted sum of four terms:

```
L = λ₁·L_KE + λ₂·L_stationarity + λ₃·L_align + λ₄·L_OT_sub
```

- **`L_KE` — kinetic energy.** The mean action `∫₀ᵀ ½‖v‖² dt` accumulated as
  points flow through one cycle (trapezoidal rule over the ODE trajectory).
  Regularises toward the gentlest flow.
- **`L_stationarity` — time-marginal matches the data.** Seed a *localized* arc
  of the circular coordinate, flow it around the cycle, pool the trajectory
  positions sampled uniformly in time, and take the Sinkhorn divergence to the
  entire data distribution. This forces the flow's own stationary (time-average)
  density to equal `p_data`. Seeding from a localized clump is essential — a
  random subset already looks like `p_data` and would be matched at zero speed.
- **`L_align` — angular alignment.** `1 − cos(v_φ(x), ∇θ(x))`, averaged over
  points. Fixes the *direction* of the field; scale-invariant. `∇θ` is estimated
  from the circular coordinate by `circular_gradient.py`.
- **`L_OT_sub` — cycle consistency.** Sinkhorn divergence between a subset and
  the *same* subset evolved over one full cycle, `S(Φ_T(x₀), x₀)`. Source = sink;
  enforces "go around once in `T = 1`" (an integer number of loops).

How speed is pinned: `L_stationarity` (localized seed) forces the flow to cover
the whole cycle, i.e. **at least** one loop; `L_KE` trims to the *minimum* speed
that still covers it (one loop); `L_OT_sub` keeps the period at an integer number
of loops. Together they fix a single loop per cycle. On a rotationally
*symmetric* ring `L_OT_sub` has little signal, but `L_stationarity` still pins
the speed.

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
