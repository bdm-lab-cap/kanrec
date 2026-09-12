"""
Baseline numerical encoders and the model factory that builds every model
compared in this thesis (raw normalisation, AutoDis, KAN-REC) around the
same InteractionMLP backbone and prediction head.

Correccion aplicada en la revision critica
--------------------------------------------------------------
The previous AutoDis implementation applied a sigmoid to the per-bucket
logits *before* the softmax:

    w = softmax(cat([sigmoid(dist), ones], dim=1), dim=-1)

Sigmoid squashes every logit into (0, 1), so the largest possible ratio
between two softmax weights is at most e^1 ≈ 2.7 — the attention over
buckets came out nearly uniform no matter what the input was, and the
numerical value barely reached the embedding. Measured log-loss for that
version (0.6942) was *worse* than the constant predictor p=0.5 (0.6931):
the baseline was not trained, in the sense that it was not doing anything
its inputs could have prevented.

This version follows the AutoDis paper's actual scheme: an unbounded
linear projection to per-bucket logits, a LeakyReLU + linear skip
connection (the paper's trick to keep gradients flowing to the projection),
and a learnable per-field temperature — then softmax, with no bounding
nonlinearity in between.
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder import KANNumericalEncoder
from .model import CTRModel, KANRecModel


class RawNumericalEncoder(nn.Module):
    """
    Baseline: each already-normalised numerical field is linearly projected
    to embedding_dim. This is the "raw normalisation" baseline from the
    audit — no discretisation, no spline, just a per-field linear map.
    """

    def __init__(self, num_fields: int, embedding_dim: int = 16):
        super().__init__()
        self.num_fields = num_fields
        self.proj = nn.ModuleList([nn.Linear(1, embedding_dim) for _ in range(num_fields)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embs = [self.proj[j](x[:, j : j + 1]).unsqueeze(1) for j in range(self.num_fields)]
        return torch.cat(embs, dim=1)


class AutoDisNumericalEncoder(nn.Module):
    """
    AutoDis (Guo et al., KDD 2021): differentiable soft discretisation of a
    continuous field into `num_buckets` learned meta-embeddings, combined
    by an attention distribution over the buckets.

    Args:
        num_fields:     Number of numerical fields.
        embedding_dim:  Output embedding dimension per field.
        num_buckets:    Number of soft buckets per field (paper's H).
        temperature:    Initial softmax temperature (learned per field,
                         kept positive via softplus).
    """

    def __init__(
        self,
        num_fields: int,
        embedding_dim: int = 16,
        num_buckets: int = 16,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_fields = num_fields
        self.num_buckets = num_buckets

        # h = LeakyReLU(W_h x + b_h)  — per-field projection to bucket logits
        self.project = nn.ModuleList(
            [nn.Linear(1, num_buckets) for _ in range(num_fields)]
        )
        self.leaky_relu = nn.LeakyReLU(0.1)

        # h' = W_d h + b_d  — the paper's skip connection, kept as a residual
        # so gradients reach `project` even where LeakyReLU saturates.
        self.skip = nn.ModuleList(
            [nn.Linear(num_buckets, num_buckets) for _ in range(num_fields)]
        )

        self.meta_embeddings = nn.ParameterList([
            nn.Parameter(torch.randn(num_buckets, embedding_dim) * 0.1)
            for _ in range(num_fields)
        ])

        # Learnable per-field temperature, reparametrised through softplus
        # so it always stays positive; initialised so softplus(raw) ~= temperature.
        init_raw = math.log(math.expm1(temperature)) if temperature > 0 else 0.0
        self.log_temperature = nn.Parameter(torch.full((num_fields,), init_raw))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embs = []
        for j in range(self.num_fields):
            xj = x[:, j : j + 1]                         # [B, 1]
            h = self.leaky_relu(self.project[j](xj))     # [B, num_buckets]
            h = h + self.skip[j](h)                       # unbounded logits, no sigmoid
            tau = F.softplus(self.log_temperature[j]) + 1e-3
            attn = F.softmax(h / tau, dim=-1)             # [B, num_buckets]
            e = attn @ self.meta_embeddings[j]             # [B, embedding_dim]
            embs.append(e.unsqueeze(1))
        return torch.cat(embs, dim=1)

    def attention_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Mean entropy of the bucket attention, in nats, averaged over fields
        and the batch. Useful as a regression test against the old bug:
        a near-uniform distribution over `num_buckets` buckets has entropy
        close to log(num_buckets); a distribution that has actually learned
        to discriminate should sit measurably below that ceiling.
        """
        entropies = []
        with torch.no_grad():
            for j in range(self.num_fields):
                xj = x[:, j : j + 1]
                h = self.leaky_relu(self.project[j](xj))
                h = h + self.skip[j](h)
                tau = F.softplus(self.log_temperature[j]) + 1e-3
                attn = F.softmax(h / tau, dim=-1)
                entropies.append(-(attn * (attn + 1e-12).log()).sum(dim=-1).mean())
        return torch.stack(entropies).mean()


EncoderName = Literal["raw", "autodis", "kan-bspline"]


def build_model(
    encoder: EncoderName,
    num_numerical: int,
    cat_cardinalities: list[int],
    embedding_dim: int = 16,
    kan_grid_size: int = 10,
    kan_spline_order: int = 3,
    autodis_num_buckets: int = 16,
    autodis_temperature: float = 1.0,
) -> nn.Module:
    """
    Builds the model for a named encoder, all sharing the same
    InteractionMLP backbone and head via CTRModel (la revision critica).

    This is what fixes `experiments/train.py --encoder ...`: previously the
    flag only changed the checkpoint's file name because every branch
    built a KANRecModel regardless. Now it actually selects the numerical
    encoder, so `run_all.sh`'s three runs are three different models
    instead of the same model saved under three names.

    Args:
        encoder: one of "raw", "autodis", "kan-bspline".
    """
    if encoder == "kan-bspline":
        return KANRecModel(
            num_numerical=num_numerical,
            cat_cardinalities=cat_cardinalities,
            embedding_dim=embedding_dim,
            kan_grid_size=kan_grid_size,
            kan_spline_order=kan_spline_order,
        )
    if encoder == "raw":
        numerical_encoder = RawNumericalEncoder(num_numerical, embedding_dim)
    elif encoder == "autodis":
        numerical_encoder = AutoDisNumericalEncoder(
            num_numerical, embedding_dim, autodis_num_buckets, autodis_temperature
        )
    else:
        raise ValueError(f"Unknown encoder '{encoder}'. Expected raw, autodis or kan-bspline.")

    return CTRModel(
        numerical_encoder=numerical_encoder,
        cat_cardinalities=cat_cardinalities,
        num_numerical=num_numerical,
        embedding_dim=embedding_dim,
    )
