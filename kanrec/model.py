"""
KANRecModel: full model combining KAN numerical encoder,
standard categorical embeddings, KAN interaction layer,
and a sigmoid prediction head.
"""
import torch
import torch.nn as nn

try:
    from efficient_kan import KAN
except ImportError:
    raise ImportError("pip install efficient-kan")

from .encoder import KANNumericalEncoder


class KANInteractionLayer(nn.Module):
    """
    Shallow KAN over the concatenated embedding vector.
    Kept to 1 hidden layer to maintain symbolic tractability.
    """

    def __init__(self, input_dim: int, grid_size: int = 5):
        super().__init__()
        hidden_dim = max(input_dim // 2, 32)
        output_dim = max(input_dim // 4, 16)
        self.kan = KAN(
            layers_hidden=[input_dim, hidden_dim, output_dim],
            grid_size=grid_size,
            spline_order=3,
        )
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.view(x.size(0), -1)
        return self.kan(flat)


class KANRecModel(nn.Module):
    """
    Full KAN-REC model.

    Args:
        num_numerical:      Number of numerical fields (post-MLlib selection).
        cat_cardinalities:  List of vocabulary sizes for each categorical field.
        embedding_dim:      Shared embedding dimension.
        kan_grid_size:      Grid size for numerical encoder.
        kan_spline_order:   Spline order for numerical encoder.
        monotone_fields:    Field indices where monotonicity is enforced.
    """

    def __init__(
        self,
        num_numerical: int,
        cat_cardinalities: list[int],
        embedding_dim: int = 16,
        kan_grid_size: int = 10,
        kan_spline_order: int = 3,
        monotone_fields: list[int] | None = None,
    ):
        super().__init__()
        self.num_numerical   = num_numerical
        self.num_categorical = len(cat_cardinalities)
        self.embedding_dim   = embedding_dim

        # 1. KAN encoder for numerical fields
        self.numerical_encoder = KANNumericalEncoder(
            num_fields=num_numerical,
            embedding_dim=embedding_dim,
            grid_size=kan_grid_size,
            spline_order=kan_spline_order,
            monotone_fields=monotone_fields,
        )

        # 2. Standard lookup embeddings for categorical fields
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card + 1, embedding_dim, padding_idx=0)
            for card in cat_cardinalities
        ])

        # 3. Shallow KAN interaction
        total_fields = num_numerical + len(cat_cardinalities)
        self.interaction = KANInteractionLayer(
            input_dim=total_fields * embedding_dim,
            grid_size=5,
        )

        # 4. Prediction head
        self.head = nn.Sequential(
            nn.Linear(self.interaction.output_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Entropy regularisation weight (promotes spline sparsity → better symbolic extraction)
        self.entropy_reg_weight = 1e-3

    def forward(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x_num: [batch, num_numerical]   — normalised numerical features
            x_cat: [batch, num_categorical] — integer categorical IDs
        Returns:
            y_pred: [batch, 1]
        """
        num_emb = self.numerical_encoder(x_num)                    # [B, N_num, D]
        cat_embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_emb  = torch.stack(cat_embs, dim=1)                    # [B, N_cat, D]
        all_emb  = torch.cat([num_emb, cat_emb], dim=1)            # [B, N_total, D]
        return self.head(self.interaction(all_emb))

    def entropy_regularization_loss(self) -> torch.Tensor:
        """
        Entropy over spline weight distributions — encourages sparsity,
        which is required for stable symbolic extraction (KAN 2.0).
        """
        reg = torch.tensor(0.0, device=next(self.parameters()).device)
        for kan in self.numerical_encoder.field_kans:
            if hasattr(kan, "spline_weight"):
                w      = kan.spline_weight.abs()
                w_norm = w / (w.sum() + 1e-8)
                entropy = -(w_norm * (w_norm + 1e-8).log()).sum()
                reg = reg + entropy
        return self.entropy_reg_weight * reg
