"""
CTRModel: shared architecture for every encoder compared in this thesis
(raw normalisation, AutoDis, KAN-REC), plus KANRecModel as the concrete
KAN-REC instantiation.

Single source of truth (auditoría de tribunal — hallazgo B5 y B6)
--------------------------------------------------------------------
Earlier, `KANRecModel` here used a `KANInteractionLayer` (a second KAN),
while every training notebook (04, 07, 08, 09) independently redefined
`KANRecModel` with a plain MLP interaction and trained *that* version —
the one the memoria documents. So the installable package was not the
model that produced the thesis's results, and `experiments/train.py`'s
`--encoder` flag changed only the checkpoint's file name: it always built
a KANRecModel regardless of the requested encoder (see baselines.py).

Now there is exactly one interaction backbone (`InteractionMLP`) and one
prediction head, shared by all three encoders via `CTRModel`. The only
thing that differs between raw / AutoDis / KAN-REC is the numerical
encoder plugged in — which is exactly the variable the thesis is about.
Notebooks and `experiments/train.py` should build models through
`kanrec.baselines.build_model(...)`, never redefine them locally.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import KANNumericalEncoder


class InteractionMLP(nn.Module):
    """
    Feature-interaction backbone shared by every encoder in the comparison.
    Kept identical across encoders so that any difference in downstream
    metrics is attributable to the encoder, not to the backbone.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))


class CTRModel(nn.Module):
    """
    Full CTR model: [numerical_encoder | categorical embeddings] ->
    InteractionMLP -> sigmoid head.

    Args:
        numerical_encoder:  Any module with
                             forward(x_num) -> [batch, num_numerical, embedding_dim].
                             This is the only thing that changes between the
                             raw / AutoDis / KAN-REC comparison.
        cat_cardinalities:  Vocabulary sizes for each categorical field.
        num_numerical:      Number of numerical fields (must match the encoder).
        embedding_dim:      Shared embedding dimension.
    """

    def __init__(
        self,
        numerical_encoder: nn.Module,
        cat_cardinalities: list[int],
        num_numerical: int,
        embedding_dim: int = 16,
    ):
        super().__init__()
        self.num_numerical = num_numerical
        self.num_categorical = len(cat_cardinalities)
        self.embedding_dim = embedding_dim

        self.numerical_encoder = numerical_encoder

        # NOTE (deferred, hallazgo B9): padding_idx=0 zeroes out whichever
        # category Spark's StringIndexer assigned index 0 to — the *most
        # frequent* category, not a padding token. Affects all encoders
        # equally, so comparisons stay fair; left for a later pass.
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, embedding_dim, padding_idx=0)
            for card in cat_cardinalities
        ])

        total_fields = num_numerical + len(cat_cardinalities)
        self.interaction = InteractionMLP(
            input_dim=total_fields * embedding_dim,
            hidden_dim=256,
            output_dim=64,
        )

        self.head = nn.Sequential(
            nn.Linear(self.interaction.output_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_num: [batch, num_numerical]   — normalised numerical features
            x_cat: [batch, num_categorical] — integer categorical IDs
        Returns:
            y_pred: [batch, 1]
        """
        num_emb = self.numerical_encoder(x_num)                     # [B, N_num, D]
        cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_emb = torch.stack(cat_embs, dim=1)                      # [B, N_cat, D]
        all_emb = torch.cat([num_emb, cat_emb], dim=1)              # [B, N_total, D]
        return self.head(self.interaction(all_emb))


class KANRecModel(CTRModel):
    """
    KAN-REC: CTRModel with a KANNumericalEncoder plugged in.

    Note on grid calibration: the numerical encoder's B-spline grid starts
    at the library default [-1, 1] and does *not* adapt itself. Call
    `model.calibrate(x_num_sample)` once, right after construction and
    before training, on a batch of normalised numerical data — see
    KANNumericalEncoder.calibrate for why this matters (hallazgo A3).
    """

    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: list[int],
        embedding_dim: int = 16,
        kan_grid_size: int = 10,
        kan_spline_order: int = 3,
    ):
        numerical_encoder = KANNumericalEncoder(
            num_fields=num_numerical,
            embedding_dim=embedding_dim,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
        )
        super().__init__(
            numerical_encoder=numerical_encoder,
            cat_cardinalities=cat_cardinalities,
            num_numerical=num_numerical,
            embedding_dim=embedding_dim,
        )
        # Entropy regularisation weight (promotes spline sparsity -> more
        # faithful symbolic extraction). See encoder.entropy_regularization_loss
        # for the fix to the bug that made this always contribute zero
        # (hallazgo A6).
        #
        # 1e-5, not 1e-3 (verificado empiricamente): at 1e-3 the penalty
        # crushes the spline path. Measured on a synthetic sin(2.5x) signal,
        # |base_weight| / |spline_weight| after 40 epochs was 63x at 1e-3
        # versus 16x at 1e-5 -- i.e. the regulariser was suppressing the
        # very component this thesis is about. Ironically the A6 bug (which
        # made the term always exactly 0.0) was hiding this.
        self.entropy_reg_weight = 1e-5

    def parameter_groups(self, base_lr: float = 1e-3, spline_lr_mult: float = 25.0) -> list[dict]:
        """
        Optimiser parameter groups giving the spline coefficients their own,
        larger learning rate.

        Why this exists (hallazgo detectado al ejecutar 05_symbolic_extraction
        sobre datos reales): efficient-kan initialises `spline_weight` with
        noise of amplitude `scale_noise / grid_size` (~0.01), while
        `base_weight` gets full Kaiming init (~0.5). The spline therefore
        starts ~50x smaller and, under a single shared learning rate, never
        catches up: after 40 epochs it was still 13-63x smaller, and the
        learned phi curves stayed essentially straight lines regardless of
        the true shape of the data.

        The symptom was unmistakable: symbolic extraction returned `linear`
        for all 10 surviving fields with R2 ~= 0.998, eight of them sharing
        the identical value 0.9981 -- the encoder was blind to the data.

        With a 25x spline learning rate the encoder becomes shape-adaptive,
        measured on synthetic signals over 3 seeds (R2 of a linear fit to
        the learned curve -- low means "genuinely curved", high means
        "straight"):

            true signal   shared lr     spline lr x25
            sin(2.5x)     0.734         0.094   <- correctly curves
            x^2           0.732         0.028   <- correctly curves
            2x (linear)   0.737         0.876   <- correctly stays straight

        Note the control: under the shared lr the encoder produced R2~0.73
        no matter what the underlying signal was. It is the *adaptivity*,
        not merely a lower number, that shows the spline is now working.

        Usage:
            optimizer = torch.optim.Adam(model.parameter_groups(lr), weight_decay=1e-5)
        """
        spline_params, other_params = [], []
        for name, param in self.named_parameters():
            if "spline_weight" in name or "spline_scaler" in name:
                spline_params.append(param)
            else:
                other_params.append(param)
        return [
            {"params": other_params, "lr": base_lr},
            {"params": spline_params, "lr": base_lr * spline_lr_mult},
        ]

    def calibrate(self, x_num_sample: torch.Tensor) -> None:
        """Adapts the numerical encoder's B-spline grids. See KANNumericalEncoder.calibrate."""
        self.numerical_encoder.calibrate(x_num_sample)

    def entropy_regularization_loss(self) -> torch.Tensor:
        """
        Sparsity-promoting regularisation over the numerical encoder's
        spline weights (KAN 2.0). Always returned exactly 0.0 before this
        fix — see KANNumericalEncoder.entropy_regularization_loss.
        """
        device = next(self.parameters()).device
        reg = self.numerical_encoder.entropy_regularization_loss()
        return self.entropy_reg_weight * reg.to(device)
