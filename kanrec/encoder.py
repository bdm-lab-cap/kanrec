"""
KAN Numerical Encoder — core contribution of KAN-REC.

For each numerical field j, learns a continuous mapping:
    φⱼ : ℝ → ℝᵈ
implemented as an independent EfficientKAN edge per field, sharing input xⱼ.

Unlike AutoDis (KDD 2021), this encoder:
  - Requires NO discretisation (no bucket boundaries, no H hyperparameter)
  - Is smooth and differentiable everywhere
  - Supports optional monotonicity constraints per field (partial — see note
    on `monotone_fields` below)
  - Produces directly plottable curves (see get_spline_curves)

Correccion aplicada en la revision critica
------------------------------------------------------------
efficient-kan's B-spline grid defaults to ``[-1, 1]``. Outside that range
every basis function is exactly zero, so the spline term of the encoder
vanishes and ``forward`` degenerates to ``base_weight · SiLU(x)`` — a plain
linear layer, not a spline. Combined with unnormalised inputs (I6–I13 in
Criteo reach the thousands), this meant the "KAN encoder" was doing almost
no spline work on most of the data.

The fix is `calibrate()`: it runs the vendored `KAN.forward(x, update_grid=True)`
on a representative batch, which adapts each field's knot sequence to that
field's *own* empirical distribution (a blend of adaptive quantile-based and
uniform placement — see `efficient_kan.KANLinear.update_grid`). Call it once,
right after building the model and before training, on a batch drawn from
the *normalised* training data.

Known limitation (deferred): the `monotone_fields` constraint is currently
applied as `cumsum(relu(·))` over the *embedding* axis, which does not impose
monotonicity in `x`. A correct fix requires reparametrising the spline
coefficients themselves (they can be forced non-decreasing along the knot
axis, using the variation-diminishing property of B-splines) and dropping or
constraining the base (SiLU) path, which is not monotonic. This is left for
a dedicated interpretability pass; until then, `monotone_fields` should be
treated as inactive and is not used by the model factory.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .vendor.efficient_kan import KAN


class KANNumericalEncoder(nn.Module):
    """
    Args:
        num_fields:       Number of numerical fields (e.g. 10 after MLlib selection).
        embedding_dim:    Output embedding dimension per field.
        grid_size:        B-spline grid points (ablation: 5, 10, 20).
        spline_order:     B-spline order (ablation: 3, 5).
        monotone_fields:  Reserved for a future monotone reparametrisation.
                          Currently unused — see module docstring.
    """

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int = 16,
        grid_size: int = 10,
        spline_order: int = 3,
        monotone_fields: list[int] | None = None,
        input_clip: float = 10.0,
    ):
        super().__init__()
        self.num_fields = num_fields
        #: Winsorizado de la entrada, en desviaciones tipicas (ver forward()).
        self.input_clip = input_clip
        self.embedding_dim = embedding_dim
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.monotone_fields = set(monotone_fields or [])
        self._calibrated = False

        # One independent KAN per numerical field: R^1 -> R^d.
        # grid_range starts at the library default [-1, 1]; call calibrate()
        # before training to adapt it to each field's real distribution.
        self.field_kans = nn.ModuleList([
            KAN(
                layers_hidden=[1, embedding_dim],
                grid_size=grid_size,
                spline_order=spline_order,
            )
            for _ in range(num_fields)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_fields] — normalised numerical values
        Returns:
            embeddings: [batch, num_fields, embedding_dim]
        """
        # Winsorizado de la entrada (verificado empiricamente): tras StandardScaler, Criteo conserva outliers de hasta ~690
        # desviaciones tipicas en I6/I12. La ruta base del KAN es
        # base_weight * SiLU(x), y SiLU(690) ~= 690, asi que un solo outlier
        # arrastra el embedding a magnitud ~1e2-1e3. Con grid_size>=10 eso
        # basta para que los logits desborden float32 en GPU y todas las
        # predicciones salgan inf (AUC 0.5, logloss inf). Medido: sin clip
        # |emb|max = 6.9e2; con clip = 1.0e1, dos ordenes de magnitud menos.
        #
        # +-10 sigmas conserva el 99.99% de los datos intacto (una normal
        # supera 10 sigmas con probabilidad ~1e-23): solo se recortan los
        # outliers patologicos, que es exactamente la practica estandar de
        # winsorizado en CTR. No afecta a los baselines: viven fuera de
        # este encoder.
        x = x.clamp(-self.input_clip, self.input_clip)

        embeddings = []
        for j, kan in enumerate(self.field_kans):
            xj = x[:, j : j + 1]           # [batch, 1]
            ej = kan(xj)                   # [batch, embedding_dim]
            embeddings.append(ej.unsqueeze(1))  # [batch, 1, embedding_dim]

        return torch.cat(embeddings, dim=1)     # [batch, num_fields, embedding_dim]

    @torch.no_grad()
    def calibrate(self, x_sample: torch.Tensor) -> None:
        """
        Adapts every field's B-spline grid to its own empirical distribution.

        Must be called once, before training, on a representative batch of
        *normalised* data (a few thousand rows concatenated across batches
        is enough). Uses efficient-kan's own `update_grid`, so the adaptation
        is the officially supported KAN grid-refinement mechanism, not a
        heuristic on top of it.

        La calibracion se ejecuta SIEMPRE EN CPU (causa raiz verificada):
        `update_grid` llama a `curve2coeff`, que resuelve
        `torch.linalg.lstsq(A, B)` donde A son las bases B-spline. En Criteo
        muchos campos concentran sus valores en pocos nudos, asi que A es
        rank-deficient (medido: rango 3 de 13 columnas). En CPU, lstsq usa
        por defecto el driver `gelsd` (basado en SVD), que maneja rango
        deficiente sin problema. En CUDA solo existe `gels`, que EXIGE rango
        completo y devuelve NaN. Ese NaN contaminaba spline_weight y hacia
        que todas las predicciones salieran inf (AUC 0.5, logloss inf) en
        GPU, mientras el mismo codigo funcionaba en CPU. Explica tambien por
        que grid_size=5 sobrevivia (8 columnas en A, menos deficiencia de
        rango) y 10/20 no (13 y 23 columnas).

        Calibrar en CPU es barato: es una sola pasada previa al
        entrenamiento, no afecta al bucle de training, que sigue en GPU.

        Args:
            x_sample: [n_rows, num_fields] normalised numerical values.
        """
        if x_sample.size(1) != self.num_fields:
            raise ValueError(
                f"calibrate() expected {self.num_fields} columns, got {x_sample.size(1)}"
            )

        device = next(self.parameters()).device
        self.to("cpu")
        x_cpu = x_sample.detach().to("cpu")
        try:
            for j, kan in enumerate(self.field_kans):
                xj = x_cpu[:, j : j + 1]
                kan(xj, update_grid=True)
                # Salvaguarda: si aun asi algun peso saliera no finito, se
                # restaura una inicializacion valida en vez de propagar NaN.
                layer = kan.layers[0]
                if not torch.isfinite(layer.spline_weight).all():
                    import warnings
                    warnings.warn(
                        f"calibrate(): spline_weight no finito en el campo {j} "
                        f"tras update_grid; se reinicializa ese campo.",
                        stacklevel=2,
                    )
                    layer.reset_parameters()
        finally:
            self.to(device)
        self._calibrated = True

    def field_range(self, field_idx: int) -> tuple[float, float]:
        """
        Returns the (low, high) span actually covered by field `field_idx`'s
        calibrated grid — i.e. the region where the spline term is active.

        Falls back to the library default [-1, 1] with a warning if
        `calibrate()` has not been called; evaluating or plotting outside
        the calibrated span is exactly the mistake that produced misleading
        symbolic fits before this fix (see module docstring).
        """
        layer = self.field_kans[field_idx].layers[0]
        order = layer.spline_order
        core = layer.grid[0, order:-order] if order > 0 else layer.grid[0]
        return core[0].item(), core[-1].item()

    def is_calibrated(self, field_idx: int, tol: float = 1e-6) -> bool:
        """
        True if this field's grid has moved away from the library default
        ``[-1, 1]``, i.e. calibrate() has run at some point.

        Checks the actual grid buffer rather than a Python flag, because
        ``grid`` is a registered PyTorch buffer and therefore IS restored by
        ``load_state_dict()``, whereas a plain attribute like
        ``self._calibrated`` is NOT. Using the flag made every model loaded
        from a checkpoint report itself as uncalibrated and emit a spurious
        warning, even though its grid was correctly restored.
        """
        low, high = self.field_range(field_idx)
        return not (abs(low + 1.0) < tol and abs(high - 1.0) < tol)

    def get_spline_curves(
        self,
        field_idx: int,
        n_points: int = 300,
        margin: float = 0.10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the learned curve phi_j over the range the spline was
        actually calibrated on (not a fixed [-3, 3] window — evaluating
        outside the grid only shows the SiLU base path, which produced the
        spurious "exp dominates every field" finding in the first version
        of this thesis).

        Returns:
            x_grid:   [n_points]                 — input values
            y_curves: [n_points, embedding_dim]  — encoder outputs
        """
        if not self.is_calibrated(field_idx):
            import warnings
            warnings.warn(
                "get_spline_curves() called on a field whose grid is still "
                "the library default [-1, 1]: call calibrate() first, or load "
                "a checkpoint trained after calibration. Curves computed now "
                "reflect the base SiLU path outside that range, not the "
                "learned spline.",
                stacklevel=2,
            )
        low, high = self.field_range(field_idx)
        span = high - low
        # El grid debe crearse en el MISMO device que el modelo: torch.linspace
        # devuelve CPU por defecto, y eso rompia con el modelo en GPU
        # ("Expected all tensors to be on the same device"). Se detecta en
        # tiempo de ejecucion en vez de asumir CPU.
        device = self.field_kans[field_idx].layers[0].base_weight.device
        x_grid = torch.linspace(low - margin * span, high + margin * span,
                                n_points, device=device)
        with torch.no_grad():
            y_curves = self.field_kans[field_idx](x_grid.unsqueeze(1))
        # Se devuelven en CPU: todo lo que consume estas curvas (ajuste
        # simbolico con scipy, graficas con matplotlib, metricas con numpy)
        # trabaja en CPU, y asi el llamante no tiene que acordarse de mover.
        return x_grid.cpu(), y_curves.cpu()

    def get_edge_norms(self) -> list[float]:
        """
        Returns the L1 norm of the *spline* weights for each field (used for
        pruning candidates). Fixed to actually reach `spline_weight`: it
        lives on the inner `KANLinear` layer (`kan.layers[0].spline_weight`),
        not on the `KAN` wrapper itself, so the previous `hasattr(kan,
        "spline_weight")` check was always False and this always fell back
        to summing every parameter — including the base (linear) path,
        which is exactly what pruning is supposed to look past.

        Note: this norm still does not by itself tell you which fields
        matter for the *prediction* — see the ablation-by-substitution
        experiment (faithfulness step) for a criterion grounded in AUC.
        """
        return [
            sum(layer.spline_weight.abs().sum().item() for layer in kan.layers)
            for kan in self.field_kans
        ]

    def entropy_regularization_loss(
        self, regularize_activation: float = 1.0, regularize_entropy: float = 1.0
    ) -> torch.Tensor:
        """
        Sparsity-promoting regularisation over spline weights (KAN 2.0),
        delegated to efficient-kan's own implementation, which lives on the
        inner `KANLinear` layers and already combines an L1 term with an
        entropy term. Summed across fields.

        This replaces the version that lived in KANRecModel, which checked
        `hasattr(kan, "spline_weight")` on the wrong object and therefore
        always contributed exactly 0.0 regardless of the model's actual
        spline weights (see tests/test_encoder.py::test_entropy_regularization_is_positive).
        """
        return sum(
            kan.regularization_loss(regularize_activation, regularize_entropy)
            for kan in self.field_kans
        )
