"""
KAN Numerical Encoder — core contribution of KAN-REC.

For each numerical field j, learns a continuous mapping:
    φⱼ : ℝ → ℝᵈ
implemented as d independent EfficientKAN edges sharing input xⱼ.

Unlike AutoDis (KDD 2021), this encoder:
  - Requires NO discretisation (no bucket boundaries, no H hyperparameter)
  - Is smooth and differentiable everywhere
  - Supports optional monotonicity constraints per field
  - Produces directly plottable curves (see get_spline_curves)
"""
import torch
import torch.nn as nn

try:
    from efficient_kan import KAN
except ImportError:
    raise ImportError(
        "Install EfficientKAN: pip install efficient-kan\n"
        "See: https://github.com/blealtan/efficient-kan"
    )


class KANNumericalEncoder(nn.Module):
    """
    Args:
        num_fields:       Number of numerical fields (e.g. 10 after MLlib selection).
        embedding_dim:    Output embedding dimension per field.
        grid_size:        B-spline grid points (ablation: 5, 10, 20).
        spline_order:     B-spline order (ablation: 3, 5).
        monotone_fields:  Set of field indices where monotonicity is enforced.
    """

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int = 16,
        grid_size: int = 10,
        spline_order: int = 3,
        monotone_fields: list[int] | None = None,
    ):
        super().__init__()
        self.num_fields = num_fields
        self.embedding_dim = embedding_dim
        self.monotone_fields = set(monotone_fields or [])

        # One independent KAN per numerical field: ℝ¹ → ℝᵈ
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
            x: [batch, num_fields]  — normalised numerical values
        Returns:
            embeddings: [batch, num_fields, embedding_dim]
        """
        embeddings = []
        for j, kan in enumerate(self.field_kans):
            xj = x[:, j : j + 1]          # [batch, 1]
            ej = kan(xj)                   # [batch, embedding_dim]
            if j in self.monotone_fields:
                # Monotonicity by construction: cumsum of ReLU activations
                ej = torch.cumsum(torch.relu(ej), dim=-1)
            embeddings.append(ej.unsqueeze(1))  # [batch, 1, embedding_dim]

        return torch.cat(embeddings, dim=1)    # [batch, num_fields, embedding_dim]

    def get_spline_curves(
        self, field_idx: int, n_points: int = 300
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates the learned curve φⱼ on a fine grid.
        Used for visualisation and symbolic extraction.

        Returns:
            x_grid:   [n_points]                — input values
            y_curves: [n_points, embedding_dim] — encoder outputs
        """
        x_grid = torch.linspace(-3.0, 3.0, n_points).unsqueeze(1)
        with torch.no_grad():
            y_curves = self.field_kans[field_idx](x_grid)
        return x_grid.squeeze(), y_curves

    def get_edge_norms(self) -> list[float]:
        """Returns the L1 norm of spline weights for each field (used for pruning)."""
        norms = []
        for kan in self.field_kans:
            if hasattr(kan, "spline_weight"):
                norm = kan.spline_weight.abs().sum().item()
            else:
                norm = sum(p.abs().sum().item() for p in kan.parameters())
            norms.append(norm)
        return norms
