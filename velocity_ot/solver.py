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

import copy
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from . import graph_ot
from . import losses as L
from .circular_gradient import estimate_gradient_field
from .dynamics import ODEIntegrator
from .models import VelocityNet

ArrayLike = np.ndarray | torch.Tensor | str


@dataclass
class Stage:
    """One training stage: which losses are active, how strongly, and for how long.

    Any weight left as ``None`` inherits the estimator's corresponding
    ``lambda_*``; ``lr=None`` inherits the estimator's ``lr``. A loss whose
    resolved weight is ``0`` is skipped entirely (so an align-only stage does no
    ODE integration or Sinkhorn, making it fast).

    Attributes:
        name: Label shown in the progress bar / logs.
        epochs: Number of epochs for this stage.
        lr: Learning rate for this stage (``None`` -> estimator ``lr``).
        lambda_ke, lambda_stationarity, lambda_align, lambda_ot_sub: Per-stage
            loss weights (``None`` -> the estimator's value).
    """

    name: str = "train"
    epochs: int = 100
    lr: float | None = None
    lambda_ke: float | None = None
    lambda_stationarity: float | None = None
    lambda_align: float | None = None
    lambda_ot_sub: float | None = None


def load_stages(spec) -> list[Stage] | None:
    """Normalise a stage spec into a ``list[Stage]`` (or ``None``).

    ``spec`` may be ``None`` (single default stage), a list of :class:`Stage`
    or dicts, or a path to a YAML file with a top-level ``stages:`` list.
    """
    if spec is None:
        return None
    if isinstance(spec, str):
        import yaml

        with open(spec) as f:
            doc = yaml.safe_load(f)
        spec = doc["stages"] if isinstance(doc, dict) and "stages" in doc else doc
    stages = []
    for s in spec:
        if isinstance(s, Stage):
            stages.append(s)
        elif isinstance(s, dict):
            stages.append(Stage(**s))
        else:
            raise TypeError(f"Each stage must be a Stage or dict, got {type(s)}.")
    return stages


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
        stationarity_burnin_loops: Number of cycles the localized seed is flowed
            *without gradients* before sampling, so it settles onto the field's
            attractor (limit cycle). This makes the stationarity marginal reflect
            the true stationary distribution rather than the initial transient,
            and truncating the gradient here avoids ill-conditioned backprop
            through the long transient.
        stationarity_sample_loops: Number of cycles (with gradients) pooled after
            burn-in to form the stationarity marginal; ``>= 1`` covers the cycle.
        normalize_cost: Rescale OT cost matrices to an ``O(1)`` scale.
        lr: Learning rate.
        weight_decay: Optimiser weight decay.
        optimizer: ``"adam"`` or ``"adamw"``.
        knn: Neighbours for the circular-gradient reconstruction graph.
        intrinsic_dim: Intrinsic manifold dimension for the gradient estimate
            (e.g. ``1`` for a circle); ``None`` solves in the ambient space.
        bandwidth: Kernel bandwidth for the gradient estimate; ``None`` = auto.
        n_components: If set, the input coordinates are reduced with PCA to this
            many dimensions *before* fitting, and the velocity field is learned
            in that PCA space. ``None`` fits directly in the input space. The
            fitted PCA is stored and reused for plotting projections.
        pca_whiten: Whether the fit-time PCA whitens components.
        init: Run the graph-OT initialisation stage before fine-tuning (helps
            learning in high dimensions). ``True`` by default.
        init_epochs: Epochs for the initialisation stage.
        lambda_init: Weight of the initialisation MSE ``||v(x) - u||^2``.
        init_knn: Neighbours for the effective-resistance graph (defaults to
            ``knn``).
        init_reg: Entropic regularisation for the graph-OT transition plan.
        init_angle_delta: Forward-step barrier width ``delta`` (radians);
            auto-set from the graph when ``None``.
        init_angle_weight: Weight of the angular barrier in the graph-OT cost
            (``C + init_angle_weight * Phi``; negative reproduces ``C - Phi``).
        init_phi_max: Wall value for forbidden/near-boundary angular steps.
        init_max_cells: Subsample cap for the (dense) graph-OT computation; when
            there are more cells, a random subset seeds the init targets.
        device: Torch device string; ``None`` picks CUDA when available.
        seed: Optional RNG seed for reproducibility.
        verbose: Whether to print progress during :meth:`fit`.

    Attributes:
        model: The fitted :class:`~velocity_ot.models.VelocityNet` (after
            :meth:`fit`).
        history: Dict of per-epoch loss curves (after :meth:`fit`).
        pca_: The fitted fit-time PCA (or ``None`` if ``n_components`` was unset).
        fit_key_: ``obsm`` key holding the fit-space coordinates.
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
        stationarity_burnin_loops: float = 3.0,
        stationarity_sample_loops: float = 1.0,
        normalize_cost: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer: str = "adam",
        knn: int = 10,
        intrinsic_dim: int | None = None,
        bandwidth: float | None = None,
        n_components: int | None = None,
        pca_whiten: bool = False,
        init: bool = True,
        init_epochs: int = 100,
        lambda_init: float = 1.0,
        init_knn: int | None = None,
        init_reg: float = 0.05,
        init_angle_delta: float | None = None,
        init_angle_weight: float = 1.0,
        init_phi_max: float = 50.0,
        init_max_cells: int = 2000,
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
        self.stationarity_burnin_loops = float(stationarity_burnin_loops)
        self.stationarity_sample_loops = float(stationarity_sample_loops)
        self.normalize_cost = bool(normalize_cost)

        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.optimizer = optimizer

        self.knn = int(knn)
        self.intrinsic_dim = intrinsic_dim
        self.bandwidth = bandwidth
        self.n_components = n_components
        self.pca_whiten = bool(pca_whiten)

        self.init = bool(init)
        self.init_epochs = int(init_epochs)
        self.lambda_init = float(lambda_init)
        self.init_knn = init_knn
        self.init_reg = float(init_reg)
        self.init_angle_delta = init_angle_delta
        self.init_angle_weight = float(init_angle_weight)
        self.init_phi_max = float(init_phi_max)
        self.init_max_cells = int(init_max_cells)

        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed = seed
        self.verbose = verbose

        self.model: VelocityNet | None = None
        self.history: dict[str, list[float]] = {}
        self.init_history: dict[str, list[float]] = {}
        self.stage_history_: list = []
        self.dim_: int | None = None
        self.best_epoch_: int | None = None
        self.best_loss_: float | None = None
        # Transforms saved at fit time and reused for plotting.
        self.pca_ = None                 # fit-space PCA (raw -> fit space), or None
        self.fit_key_: str | None = None  # obsm key holding the fit-space coords
        self._plot_reducers: dict = {}    # cache of {(basis, k): fitted reducer}

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

    def _reduce(self, X_raw: np.ndarray, pca=None):
        """Reduce raw coordinates to the fit space via PCA.

        Returns ``(X_fit, pca_or_None)``. If ``n_components`` is unset or not
        smaller than the input dimension, no reduction is applied.
        """
        d = X_raw.shape[1]
        if self.n_components is None or self.n_components >= d:
            return X_raw.astype(np.float32), None
        if pca is None:
            from sklearn.decomposition import PCA

            pca = PCA(
                n_components=int(self.n_components),
                whiten=self.pca_whiten,
                random_state=self.seed if self.seed is not None else 0,
            ).fit(X_raw)
        return np.asarray(pca.transform(X_raw), dtype=np.float32), pca

    def transform(self, X_raw: np.ndarray | torch.Tensor) -> np.ndarray:
        """Map raw input coordinates into the fit space (applies the fit PCA)."""
        X = _to_2d_array(
            X_raw.detach().cpu().numpy() if isinstance(X_raw, torch.Tensor) else X_raw, "X"
        )
        if self.pca_ is None:
            return X.astype(np.float32)
        return np.asarray(self.pca_.transform(X), dtype=np.float32)

    def fit_space_coords(self, adata, spatial_key: str | None = None) -> np.ndarray:
        """Fit-space coordinates for ``adata`` (cached in ``obsm[fit_key_]``)."""
        if self.fit_key_ is not None and self.fit_key_ in adata.obsm:
            return _to_2d_array(adata.obsm[self.fit_key_], "fit_coords")
        return self.transform(self._get_spatial(adata, spatial_key))

    def plotting_reducer(self, basis: str, X_fit: np.ndarray, n_components: int = 2, reducer=None):
        """Return (and cache) a fitted 2-/3-D reducer for projecting the fit space.

        ``basis`` is ``"pca"`` or ``"umap"``. A pre-fitted ``reducer`` (anything
        with ``.transform``) is used as-is; otherwise one is fitted on ``X_fit``
        and cached on the estimator so repeated plots reuse it.
        """
        if reducer is not None:
            return reducer
        key = (basis, int(n_components))
        if key in self._plot_reducers:
            return self._plot_reducers[key]
        rs = self.seed if self.seed is not None else 0
        if basis == "pca":
            from sklearn.decomposition import PCA

            reducer = PCA(n_components=n_components, random_state=rs).fit(X_fit)
        elif basis == "umap":
            try:
                import umap
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "basis='umap' requires the 'umap-learn' package (pip install umap-learn)."
                ) from e
            reducer = umap.UMAP(n_components=n_components, random_state=rs).fit(X_fit)
        else:
            raise ValueError(f"Unknown basis '{basis}'. Choose 'pca' or 'umap'.")
        self._plot_reducers[key] = reducer
        return reducer

    def _build_optimizer(self, lr: float | None = None) -> torch.optim.Optimizer:
        assert self.model is not None
        opt_cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
        if self.optimizer not in opt_cls:
            raise ValueError(f"Unknown optimizer '{self.optimizer}'.")
        return opt_cls[self.optimizer](
            self.model.parameters(),
            lr=self.lr if lr is None else lr,
            weight_decay=self.weight_decay,
        )

    def _sinkhorn_kwargs(self) -> dict[str, Any]:
        return dict(
            reg=self.sinkhorn_reg,
            n_iter=self.sinkhorn_iter,
            normalize_cost=self.normalize_cost,
        )

    def _init_stage(
        self,
        X: torch.Tensor,
        theta_np: np.ndarray,
        integrator: ODEIntegrator,
        sk: dict,
        batch_size: int,
        eff_res_fn=None,
    ) -> np.ndarray:
        """Graph-OT initialisation of the velocity network.

        Builds displacement targets ``u`` from an entropic transition plan
        (effective-resistance cost + angular barrier) and trains the network on
        ``lambda_init * ||v(x) - u||^2 + lambda_ot_sub * OT_sub`` for
        ``init_epochs``. Operates on a random subsample when there are more than
        ``init_max_cells`` cells. Returns ``u`` (aligned to the subset used).
        """
        assert self.model is not None
        n = X.shape[0]
        knn = self.init_knn if self.init_knn is not None else self.knn

        # Subsample for the dense graph-OT when necessary.
        if n > self.init_max_cells:
            gen = torch.Generator(device="cpu")
            if self.seed is not None:
                gen.manual_seed(self.seed)
            sub = torch.randperm(n, generator=gen)[: self.init_max_cells]
            idx_sub = sub.to(X.device)
        else:
            idx_sub = torch.arange(n, device=X.device)

        X_sub = X[idx_sub]
        theta_sub = theta_np[idx_sub.cpu().numpy()]

        u_np, info = graph_ot.graph_ot_init_targets(
            X_sub.detach().cpu().numpy(),
            theta_sub,
            knn=knn,
            delta=self.init_angle_delta,
            angle_weight=self.init_angle_weight,
            reg=self.init_reg,
            phi_max=self.init_phi_max,
            eff_res_fn=eff_res_fn,
        )
        U = torch.as_tensor(u_np, device=X.device)

        opt = self._build_optimizer()
        m = X_sub.shape[0]
        bs = min(batch_size, m)
        self.init_history = {"total": [], "mse": [], "ot_sub": []}

        self.model.train()
        for epoch in range(self.init_epochs):
            perm = torch.randperm(m, device=X.device)
            stats = {k: 0.0 for k in self.init_history}
            nb = 0
            for start in range(0, m, bs):
                b = perm[start : start + bs]
                xb = X_sub[b]
                loss_mse = ((self.model(xb) - U[b]) ** 2).sum(-1).mean()
                # OT_sub keeps the flow periodic and prevents a degenerate field.
                endpoint = integrator(xb, t_end=self.T).endpoint
                loss_ot_sub = L.cycle_consistency_loss(xb, endpoint, **sk)
                loss = self.lambda_init * loss_mse + self.lambda_ot_sub * loss_ot_sub

                opt.zero_grad()
                loss.backward()
                opt.step()

                stats["total"] += float(loss.detach())
                stats["mse"] += float(loss_mse.detach())
                stats["ot_sub"] += float(loss_ot_sub.detach())
                nb += 1
            for k in self.init_history:
                self.init_history[k].append(stats[k] / max(nb, 1))
            if self.verbose and (
                epoch % max(1, self.init_epochs // 5) == 0 or epoch == self.init_epochs - 1
            ):
                h = {k: self.init_history[k][-1] for k in self.init_history}
                print(
                    f"[init  {epoch:4d}] total={h['total']:.4f}  "
                    f"mse={h['mse']:.4f}  OT_sub={h['ot_sub']:.4f}  (delta={info['delta']:.3f})"
                )
        return u_np

    def _progress(self, n_epochs: int, desc: str):
        """Return a progress iterator over epochs (tqdm bar if available)."""
        if not self.verbose:
            return range(n_epochs)
        try:
            from tqdm.auto import tqdm

            return tqdm(range(n_epochs), desc=desc, leave=True)
        except Exception:  # pragma: no cover
            return range(n_epochs)

    def _run_stage(self, stage, X, G, theta_order, seed_w, integrator, sk, n, batch_size):
        """Run one training stage; return ``(history, best_info)``.

        Only the losses with non-zero (resolved) weight are computed, so a stage
        that activates a subset of terms is correspondingly cheaper. Tracks the
        lowest-total-loss weights within the stage and restores them at the end.
        """
        w_ke = self.lambda_ke if stage.lambda_ke is None else stage.lambda_ke
        w_stat = self.lambda_stationarity if stage.lambda_stationarity is None else stage.lambda_stationarity
        w_align = self.lambda_align if stage.lambda_align is None else stage.lambda_align
        w_sub = self.lambda_ot_sub if stage.lambda_ot_sub is None else stage.lambda_ot_sub
        lr = self.lr if stage.lr is None else stage.lr
        need_traj = (w_ke > 0.0) or (w_sub > 0.0)  # the [0, T] rollout

        opt = self._build_optimizer(lr=lr)
        hist = {k: [] for k in ("total", "ke", "stationarity", "align", "ot_sub")}
        best_state, best_loss, best_epoch = None, float("inf"), None

        bar = self._progress(stage.epochs, f"[{stage.name}]")
        use_bar = hasattr(bar, "set_postfix")
        self.model.train()
        for epoch in bar:
            perm = torch.randperm(n, device=self.device)
            es = {k: 0.0 for k in hist}
            nb = 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                x0 = X[idx]
                zero = torch.zeros((), device=self.device)
                l_ke = l_stat = l_align = l_sub = zero

                if need_traj:
                    result = integrator(x0, t_end=self.T)
                    if w_ke > 0.0:
                        l_ke = L.kinetic_energy_loss(result.velocities, result.dt)
                    if w_sub > 0.0:
                        l_sub = L.cycle_consistency_loss(x0, result.endpoint, **sk)
                if w_align > 0.0:
                    l_align = L.angular_alignment_loss(self.model(x0), G[idx])
                if w_stat > 0.0:
                    # Flow a localized arc onto the attractor (burn-in, no grad),
                    # then match a short grad-carrying sampling rollout to data.
                    s = int(torch.randint(0, n, (1,)).item())
                    arc = theta_order[(s + torch.arange(seed_w, device=self.device)) % n]
                    x_seed = X[arc]
                    if self.stationarity_burnin_loops > 0:
                        with torch.no_grad():
                            x_seed = integrator(x_seed, t_end=(epoch // 10 + 1) * self.T).endpoint
                    loc = integrator(x_seed, t_end=self.stationarity_sample_loops * self.T)
                    l_stat = L.stationarity_loss(
                        loc.trajectory, X, n_points=self.stationarity_n_points, **sk
                    )

                loss = w_ke * l_ke + w_stat * l_stat + w_align * l_align + w_sub * l_sub
                opt.zero_grad()
                loss.backward()
                opt.step()

                es["total"] += float(loss.detach())
                es["ke"] += float(l_ke.detach())
                es["stationarity"] += float(l_stat.detach())
                es["align"] += float(l_align.detach())
                es["ot_sub"] += float(l_sub.detach())
                nb += 1

            for k in hist:
                hist[k].append(es[k] / max(nb, 1))
            if hist["total"][-1] < best_loss:
                best_loss = float(hist["total"][-1])
                best_epoch = int(epoch)
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

            if use_bar:
                active = {"tot": hist["total"][-1]}
                if w_ke > 0: active["ke"] = hist["ke"][-1]
                if w_stat > 0: active["stat"] = hist["stationarity"][-1]
                if w_align > 0: active["align"] = hist["align"][-1]
                if w_sub > 0: active["sub"] = hist["ot_sub"][-1]
                bar.set_postfix({k: f"{v:.3f}" for k, v in active.items()})

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return hist, {"best_epoch": best_epoch, "best_loss": best_loss,
                      "weights": {"ke": w_ke, "stationarity": w_stat,
                                  "align": w_align, "ot_sub": w_sub}, "lr": lr}

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
        stages=None,
        grad_theta_key: str | None = None,
        velocity_key: str = "velocity_field",
        grad_theta_out_key: str | None = "grad_theta",
        pca=None,
        fit_key: str = "X_velocity_fit",
        eff_res_fn=None,
    ) -> "VelocityFieldEstimator":
        """Fit the velocity field to the data in ``adata``.

        Args:
            adata: An :class:`anndata.AnnData` object.
            spatial_key: Where to read coordinates ``[N, D]``. ``None`` (default)
                reads ``adata.X``; a string reads ``adata.obsm[spatial_key]``.
            theta_key: Name of the circular coordinate (radians). Read from
                ``adata.obs`` first, then ``adata.obsm``. Defaults to
                ``"circular_coords"`` (i.e. ``adata.obs['circular_coords']``).
            n_epochs: Epochs for the default single stage (used only when
                ``stages`` is ``None``).
            batch_size: Mini-batch size for the OT terms. ``None`` uses
                ``min(N, 256)``.
            stages: Staged-training spec — ``None`` (one default stage using the
                estimator's ``lambda_*``/``lr`` for ``n_epochs``), a list of
                :class:`Stage`/dicts, or a path to a YAML file with a top-level
                ``stages:`` list. Each stage sets its own active losses, weights,
                learning rate and epochs, and runs in order.
            grad_theta_key: If given, read a precomputed :math:`\\nabla\\theta`
                field from ``adata.obsm`` instead of estimating it.
            velocity_key: Output key in ``adata.obsm`` for fitted velocities
                (in the fit space).
            grad_theta_out_key: If not ``None``, also store the (estimated or
                supplied) :math:`\\nabla\\theta` field under this ``obsm`` key.
            pca: Optional pre-fitted PCA (any object with ``.transform``) to load
                instead of fitting a new one; only used when ``n_components`` is
                set. Handy for reusing a cached reducer across runs.
            fit_key: ``obsm`` key under which the fit-space coordinates are saved
                (reused by the plotting projections).
            eff_res_fn: Effective-resistance source for the init stage: a
                precomputed ``[N, N]`` cost matrix, a callable ``fn(X, knn)``
                (e.g. ``dist_utils.get_eff_res``), or ``None`` (use
                ``dist_utils.get_eff_res`` if importable, else the built-in).

        Returns:
            ``self``, with :attr:`model` and :attr:`history` populated, and the
            fitted velocity field written to ``adata.obsm[velocity_key]``.
        """
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)

        # ---- 1. extract data -------------------------------------------------
        X_raw = self._get_spatial(adata, spatial_key)
        theta_np = self._get_theta(adata, theta_key)
        n = X_raw.shape[0]
        if theta_np.shape[0] != n:
            raise ValueError(
                f"theta has {theta_np.shape[0]} entries but there are {n} points."
            )

        # ---- 1b. optional PCA reduction into the fit space -------------------
        X_np, self.pca_ = self._reduce(X_raw, pca)
        self.fit_key_ = fit_key
        adata.obsm[fit_key] = X_np
        self._plot_reducers = {}  # invalidate any cached plotting reducers
        d = X_np.shape[1]
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
        sk = self._sinkhorn_kwargs()

        if batch_size is None:
            batch_size = min(n, 256)

        # ---- 3b. graph-OT initialisation stage -------------------------------
        self.init_history = {}
        if self.init and self.init_epochs > 0:
            self._init_stage(X, theta_np, integrator, sk, batch_size, eff_res_fn=eff_res_fn)
            adata.obsm["velocity_init"] = self.predict(X_np)
            self.model.train()

        # Localized seed for the stationarity term: a contiguous arc in the
        # circular coordinate (a clump that must flow around to reproduce data).
        theta_order = torch.as_tensor(np.argsort(theta_np), device=self.device)
        seed_w = max(8, min(n, int(round(self.stationarity_seed_frac * n))))

        # ---- 4. staged training ----------------------------------------------
        stage_list = load_stages(stages)
        if stage_list is None:  # backward-compatible single stage
            stage_list = [Stage(name="train", epochs=n_epochs)]

        self.history = {"total": [], "ke": [], "stationarity": [], "align": [], "ot_sub": []}
        self.stage_history_ = []
        self.best_loss_ = None
        self.best_epoch_ = None
        offset = 0
        for stage in stage_list:
            if stage.epochs <= 0:
                continue
            hist, binfo = self._run_stage(
                stage, X, G, theta_order, seed_w, integrator, sk, n, batch_size
            )
            for k in self.history:
                self.history[k].extend(hist[k])
            binfo["name"] = stage.name
            binfo["epochs"] = stage.epochs
            binfo["global_best_epoch"] = (
                offset + binfo["best_epoch"] if binfo["best_epoch"] is not None else None
            )
            self.stage_history_.append(binfo)
            offset += stage.epochs
            # Final returned model = best of the last stage that ran.
            self.best_loss_ = binfo["best_loss"]
            self.best_epoch_ = binfo["global_best_epoch"]
            if self.verbose:
                print(
                    f"[{stage.name}] best epoch {binfo['best_epoch']} "
                    f"(total={binfo['best_loss']:.4f}) restored"
                )

        # ---- 5. write outputs back to AnnData --------------------------------
        adata.obsm[velocity_key] = self.predict(X_np)
        if grad_theta_out_key is not None:
            adata.obsm[grad_theta_out_key] = grad_np
        adata.uns["velocity_ot"] = {
            "history": {k: list(v) for k, v in self.history.items()},
            "init_history": {k: list(v) for k, v in self.init_history.items()},
            "stages": self.stage_history_,
            "best_epoch": self.best_epoch_,
            "best_loss": self.best_loss_,
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
                "n_components": self.n_components,
                "fit_key": self.fit_key_,
                "fit_dim": self.dim_,
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
