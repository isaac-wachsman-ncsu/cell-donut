"""High-level estimator that fits an OT velocity field to an ``AnnData`` object.

:class:`VelocityFieldEstimator` wires the pieces together:

* builds a :class:`~velocity_ot.models.VelocityNet`,
* estimates :math:`\\nabla\\theta` from the circular coordinate with
  :mod:`velocity_ot.circular_gradient`,
* integrates the field with :mod:`velocity_ot.dynamics`,
* optimises the composite loss from :mod:`velocity_ot.losses`, and
* writes the fitted velocities back to ``adata.obsm``.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from . import losses as L
from .circular_gradient import estimate_gradient_field
from .dynamics import ODEIntegrator
from .models import VelocityNet

ArrayLike = np.ndarray | torch.Tensor | str


def _to_2d_array(value: np.ndarray | torch.Tensor, name: str) -> np.ndarray:
    """Coerce to a float32 ``[N, D]`` numpy array with validation."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"`{name}` must be 2-D [N, D], got shape {arr.shape}.")
    return arr


class VelocityFieldEstimator:
    """Learn an autonomous OT velocity field matching a cyclic dynamical system.

    The estimator assumes the observed points are an i.i.d. sample from the
    *stationary* distribution of a system whose state cycles with period ``T``.
    It learns :math:`v_\\phi` so that (i) the flow's time-marginal density
    matches the data (stationarity), (ii) the flow follows the circular
    coordinate (angular alignment), (iii) the motion is as gentle as possible
    (kinetic energy), and (iv) a subset returns to itself after one full cycle
    (``OT_sub`` cycle-consistency), which pins the period to ``T``.

    Args:
        hidden_dims: Hidden-layer widths of the velocity MLP.
        activation: Activation name (see
            :data:`velocity_ot.models.ACTIVATION_FN`).
        layer_norm: Whether to layer-normalise hidden activations.
        residual: Whether to use residual connections.
        n_steps: Integration steps per full cycle.
        method: ODE integrator, ``"euler"`` or ``"rk4"``.
        T: Cycle length (normalised time horizon).
        lambda_ke: Weight :math:`\\lambda_1` of the kinetic-energy term.
        lambda_stationarity: Weight :math:`\\lambda_2` of the stationarity
            (time-marginal vs. data) term.
        lambda_align: Weight :math:`\\lambda_3` of the alignment term.
        lambda_ot_sub: Weight :math:`\\lambda_4` of the cycle-consistency
            (``OT_sub``: subset returns to itself after one cycle) term.
        sinkhorn_reg: Entropic regularisation for all OT terms.
        sinkhorn_iter: Maximum Sinkhorn iterations.
        stationarity_n_points: Cap on the number of points per side of the
            stationarity Sinkhorn divergence (pooled trajectory nodes and the
            data are each subsampled to at most this many points).
        stationarity_seed_frac: Fraction of the points forming the localized arc
            (contiguous in the circular coordinate) that seeds the stationarity
            term. Must be small enough that the clump does not already resemble
            ``p_data`` (so reproducing it requires flowing around the cycle).
        normalize_cost: Rescale OT cost matrices to an ``O(1)`` scale.
        lr: Learning rate.
        weight_decay: Optimiser weight decay.
        optimizer: ``"adam"`` or ``"adamw"``.
        knn: Neighbours for the circular-gradient reconstruction graph.
        intrinsic_dim: Intrinsic manifold dimension for the gradient estimate
            (e.g. ``1`` for a circle); ``None`` solves in the ambient space.
        bandwidth: Kernel bandwidth for the gradient estimate; ``None`` = auto.
        device: Torch device string; ``None`` picks CUDA when available.
        seed: Optional RNG seed for reproducibility.
        verbose: Whether to print progress during :meth:`fit`.

    Attributes:
        model: The fitted :class:`~velocity_ot.models.VelocityNet` (after
            :meth:`fit`).
        history: Dict of per-epoch loss curves (after :meth:`fit`).
    """

    def __init__(
        self,
        hidden_dims: Sequence[int] = (128, 128, 128),
        activation: str = "silu",
        layer_norm: bool = True,
        residual: bool = False,
        n_steps: int = 20,
        method: str = "rk4",
        T: float = 1.0,
        lambda_ke: float = 0.01,
        lambda_stationarity: float = 1.0,
        lambda_align: float = 1.0,
        lambda_ot_sub: float = 1.0,
        sinkhorn_reg: float = 0.05,
        sinkhorn_iter: int = 200,
        stationarity_n_points: int = 256,
        stationarity_seed_frac: float = 0.2,
        normalize_cost: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer: str = "adam",
        knn: int = 10,
        intrinsic_dim: int | None = None,
        bandwidth: float | None = None,
        device: str | None = None,
        seed: int | None = None,
        verbose: bool = True,
    ) -> None:
        self.hidden_dims = tuple(hidden_dims)
        self.activation = activation
        self.layer_norm = layer_norm
        self.residual = residual

        self.n_steps = int(n_steps)
        self.method = method
        self.T = float(T)

        self.lambda_ke = float(lambda_ke)
        self.lambda_stationarity = float(lambda_stationarity)
        self.lambda_align = float(lambda_align)
        self.lambda_ot_sub = float(lambda_ot_sub)

        self.sinkhorn_reg = float(sinkhorn_reg)
        self.sinkhorn_iter = int(sinkhorn_iter)
        self.stationarity_n_points = int(stationarity_n_points)
        self.stationarity_seed_frac = float(stationarity_seed_frac)
        self.normalize_cost = bool(normalize_cost)

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.optimizer = optimizer

        self.knn = int(knn)
        self.intrinsic_dim = intrinsic_dim
        self.bandwidth = bandwidth

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed = seed
        self.verbose = verbose

        self.model: VelocityNet | None = None
        self.history: dict[str, list[float]] = {}
        self.dim_: int | None = None

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_spatial(adata, spatial_key: str | None) -> np.ndarray:
        """Extract spatial coordinates ``[N, D]``.

        ``spatial_key=None`` reads ``adata.X`` (densifying if sparse);
        otherwise ``adata.obsm[spatial_key]``.
        """
        if spatial_key is None:
            X = adata.X
            if hasattr(X, "toarray"):  # scipy sparse
                X = X.toarray()
        else:
            if spatial_key not in adata.obsm:
                raise KeyError(f"Key '{spatial_key}' not found in adata.obsm.")
            X = adata.obsm[spatial_key]
        return _to_2d_array(X, "spatial")

    @staticmethod
    def _get_theta(adata, theta_key: str) -> np.ndarray:
        """Extract the circular coordinate ``[N]`` (radians).

        Looks in ``adata.obs`` first, then ``adata.obsm``.
        """
        if theta_key in adata.obs:
            theta = np.asarray(adata.obs[theta_key].to_numpy(), dtype=np.float32)
        elif theta_key in adata.obsm:
            theta = np.asarray(adata.obsm[theta_key], dtype=np.float32)
        else:
            raise KeyError(
                f"Circular coordinate '{theta_key}' not found in adata.obs or adata.obsm."
            )
        return theta.reshape(-1)

    def _resolve_obsm(self, adata, value: ArrayLike, name: str) -> np.ndarray:
        """Resolve an argument that is either an array or an ``obsm`` key."""
        if isinstance(value, str):
            if value not in adata.obsm:
                raise KeyError(f"Key '{value}' not found in adata.obsm for `{name}`.")
            return _to_2d_array(adata.obsm[value], name)
        return _to_2d_array(value, name)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        assert self.model is not None
        opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
        if self.optimizer not in opt_cls:
            raise ValueError(f"Unknown optimizer '{self.optimizer}'.")
        return opt_cls[self.optimizer](
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def _sinkhorn_kwargs(self) -> dict[str, Any]:
        return dict(
            reg=self.sinkhorn_reg,
            n_iter=self.sinkhorn_iter,
            normalize_cost=self.normalize_cost,
        )

    # ------------------------------------------------------------------ #
    #  Fitting
    # ------------------------------------------------------------------ #
    def fit(
        self,
        adata,
        spatial_key: str | None = None,
        theta_key: str = "circular_coords",
        n_epochs: int = 200,
        batch_size: int | None = None,
        grad_theta_key: str | None = None,
        velocity_key: str = "velocity_field",
        grad_theta_out_key: str | None = "grad_theta",
    ) -> "VelocityFieldEstimator":
        """Fit the velocity field to the data in ``adata``.

        Args:
            adata: An :class:`anndata.AnnData` object.
            spatial_key: Where to read coordinates ``[N, D]``. ``None`` (default)
                reads ``adata.X``; a string reads ``adata.obsm[spatial_key]``.
            theta_key: Name of the circular coordinate (radians). Read from
                ``adata.obs`` first, then ``adata.obsm``. Defaults to
                ``"circular_coords"`` (i.e. ``adata.obs['circular_coords']``).
            n_epochs: Number of training epochs (full shuffled passes).
            batch_size: Mini-batch size for the OT terms. ``None`` uses
                ``min(N, 256)``.
            grad_theta_key: If given, read a precomputed :math:`\\nabla\\theta`
                field from ``adata.obsm`` instead of estimating it.
            velocity_key: Output key in ``adata.obsm`` for fitted velocities.
            grad_theta_out_key: If not ``None``, also store the (estimated or
                supplied) :math:`\\nabla\\theta` field under this ``obsm`` key.

        Returns:
            ``self``, with :attr:`model` and :attr:`history` populated, and the
            fitted velocity field written to ``adata.obsm[velocity_key]``.
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        # ---- 1. extract data -------------------------------------------------
        X_np = self._get_spatial(adata, spatial_key)
        theta_np = self._get_theta(adata, theta_key)
        n, d = X_np.shape
        if theta_np.shape[0] != n:
            raise ValueError(
                f"theta has {theta_np.shape[0]} entries but there are {n} points."
            )
        self.dim_ = d

        # ---- 2. circular-coordinate gradient (fixed target for alignment) ----
        if grad_theta_key is not None:
            grad_np = self._resolve_obsm(adata, grad_theta_key, "grad_theta")
        else:
            grad_np, _ = estimate_gradient_field(
                X_np,
                theta=theta_np,
                k=self.knn,
                bandwidth=self.bandwidth,
                intrinsic_dim=self.intrinsic_dim,
            )
            grad_np = grad_np.astype(np.float32)
        if grad_np.shape != X_np.shape:
            raise ValueError(
                f"grad_theta shape {grad_np.shape} does not match data {X_np.shape}."
            )

        X = torch.as_tensor(X_np, device=self.device)
        G = torch.as_tensor(grad_np, device=self.device)

        # ---- 3. model + optimiser --------------------------------------------
        self.model = VelocityNet(
            dim=d,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            layer_norm=self.layer_norm,
            residual=self.residual,
        ).to(self.device)
        integrator = ODEIntegrator(self.model, method=self.method, n_steps=self.n_steps, T=self.T)
        opt = self._build_optimizer()
        sk = self._sinkhorn_kwargs()

        if batch_size is None:
            batch_size = min(n, 256)

        # Localized seed for the stationarity term: a contiguous arc in the
        # circular coordinate (a clump that must flow around to reproduce data).
        theta_order = torch.as_tensor(np.argsort(theta_np), device=self.device)
        seed_w = max(8, min(n, int(round(self.stationarity_seed_frac * n))))

        self.history = {"total": [], "ke": [], "stationarity": [], "align": [], "ot_sub": []}

        # ---- 4. training loop ------------------------------------------------
        self.model.train()
        for epoch in range(n_epochs):
            perm = torch.randperm(n, device=self.device)
            epoch_stats = {k: 0.0 for k in self.history}
            n_batches = 0

            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                x0 = X[idx]
                g0 = G[idx]

                # One integration over [0, T] feeds KE, alignment and OT_sub.
                result = integrator(x0, t_end=self.T)

                loss_ke = L.kinetic_energy_loss(result.velocities, result.dt)
                loss_align = L.angular_alignment_loss(self.model(x0), g0)
                loss_ot_sub = L.cycle_consistency_loss(x0, result.endpoint, **sk)

                # Stationarity: flow a localized arc around the cycle and match
                # the pooled (uniform-in-time) trajectory to the full data.
                s = int(torch.randint(0, n, (1,)).item())
                arc = theta_order[(s + torch.arange(seed_w, device=self.device)) % n]
                loc_result = integrator(X[arc], t_end=self.T)
                loss_stationarity = L.stationarity_loss(
                    loc_result.trajectory, X, n_points=self.stationarity_n_points, **sk
                )

                loss = (
                    self.lambda_ke * loss_ke
                    + self.lambda_stationarity * loss_stationarity
                    + self.lambda_align * loss_align
                    + self.lambda_ot_sub * loss_ot_sub
                )

                opt.zero_grad()
                loss.backward()
                opt.step()

                epoch_stats["total"] += float(loss.detach())
                epoch_stats["ke"] += float(loss_ke.detach())
                epoch_stats["stationarity"] += float(loss_stationarity.detach())
                epoch_stats["align"] += float(loss_align.detach())
                epoch_stats["ot_sub"] += float(loss_ot_sub.detach())
                n_batches += 1

            for k in self.history:
                self.history[k].append(epoch_stats[k] / max(n_batches, 1))

            if self.verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
                h = {k: self.history[k][-1] for k in self.history}
                print(
                    f"[epoch {epoch:4d}] total={h['total']:.4f}  "
                    f"KE={h['ke']:.4f}  stationarity={h['stationarity']:.4f}  "
                    f"align={h['align']:.4f}  OT_sub={h['ot_sub']:.4f}"
                )

        # ---- 5. write outputs back to AnnData --------------------------------
        adata.obsm[velocity_key] = self.predict(X_np)
        if grad_theta_out_key is not None:
            adata.obsm[grad_theta_out_key] = grad_np
        adata.uns["velocity_ot"] = {
            "history": {k: list(v) for k, v in self.history.items()},
            "config": {
                "hidden_dims": list(self.hidden_dims),
                "activation": self.activation,
                "method": self.method,
                "n_steps": self.n_steps,
                "T": self.T,
                "lambdas": {
                    "ke": self.lambda_ke,
                    "stationarity": self.lambda_stationarity,
                    "align": self.lambda_align,
                    "ot_sub": self.lambda_ot_sub,
                },
                "sinkhorn_reg": self.sinkhorn_reg,
            },
        }
        return self

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        """Evaluate the fitted velocity field at arbitrary points.

        Args:
            X: Query coordinates ``[Q, D]``.

        Returns:
            Velocity vectors ``[Q, D]`` as a numpy array.
        """
        if self.model is None:
            raise RuntimeError("Call `fit` before `predict`.")
        was_training = self.model.training
        self.model.eval()
        x = torch.as_tensor(_to_2d_array(X, "X"), device=self.device)
        v = self.model(x).cpu().numpy()
        if was_training:
            self.model.train()
        return v


# ======================================================================= #
#  Example usage
# ======================================================================= #
if __name__ == "__main__":
    import anndata as ad

    rng = np.random.default_rng(0)

    # ---- synthetic 2-D stationary sample of a rotating ring ----------------
    # Non-uniform angular density so the OT_sub term genuinely constrains the
    # period (a uniform ring is invariant under any rotation).
    n = 500
    theta = np.sort(rng.beta(2.0, 5.0, size=n) * 2.0 * np.pi)  # clumped angles
    radius = 1.0 + 0.03 * rng.standard_normal(n)
    coords = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)

    # Spatial coordinates live in adata.X; circular coordinate in adata.obs.
    adata = ad.AnnData(X=coords.astype(np.float32))
    adata.obs["circular_coords"] = theta.astype(np.float32)

    # ---- fit for a quick 5-epoch smoke test --------------------------------
    estimator = VelocityFieldEstimator(
        hidden_dims=(64, 64),
        activation="silu",
        n_steps=10,
        method="rk4",
        lambda_ke=0.01,
        lambda_stationarity=1.0,
        lambda_align=1.0,
        lambda_ot_sub=1.0,
        sinkhorn_reg=0.05,
        lr=3e-3,
        intrinsic_dim=1,   # data lie on a 1-D circle
        knn=12,
        device="cpu",
        seed=0,
    )
    estimator.fit(adata, n_epochs=5)  # reads adata.X and adata.obs['circular_coords']

    # ---- inspect results ---------------------------------------------------
    v = adata.obsm["velocity_field"]
    print("\nFitted velocity field written to adata.obsm['velocity_field'], "
          f"shape {v.shape}")
    speed = np.linalg.norm(v, axis=1)
    tangent = np.stack([-np.sin(theta), np.cos(theta)], axis=1)
    cos = (v * tangent).sum(1) / (speed * np.linalg.norm(tangent, axis=1) + 1e-8)
    print(f"mean speed              : {speed.mean():.3f}")
    print(f"mean cos(v, +theta dir) : {cos.mean():.3f}  (1.0 = perfectly rotational)")

    # Out-of-sample evaluation on a fresh grid point.
    print("velocity at (1, 0):", estimator.predict(np.array([[1.0, 0.0]]))[0])
