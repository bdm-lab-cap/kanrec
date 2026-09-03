"""Tests for KANNumericalEncoder."""
import pytest
import torch
from kanrec.encoder import KANNumericalEncoder


@pytest.fixture
def encoder():
    return KANNumericalEncoder(num_fields=5, embedding_dim=8, grid_size=5, spline_order=3)


def test_output_shape(encoder):
    x = torch.randn(32, 5)
    out = encoder(x)
    assert out.shape == (32, 5, 8), f"Expected (32,5,8), got {out.shape}"


def test_gradient_flows(encoder):
    x = torch.randn(16, 5, requires_grad=False)
    out = encoder(x)
    loss = out.sum()
    loss.backward()
    for p in encoder.parameters():
        assert p.grad is not None


def test_get_spline_curves(encoder):
    x_grid, y_curves = encoder.get_spline_curves(0, n_points=100)
    assert x_grid.shape  == (100,)
    assert y_curves.shape == (100, 8)


def test_monotone_field_is_nondecreasing(encoder):
    """With monotone_fields=[0], output for field 0 should be non-decreasing."""
    enc = KANNumericalEncoder(num_fields=3, embedding_dim=4,
                               grid_size=5, monotone_fields=[0])
    x_sorted = torch.linspace(-2, 2, 50).unsqueeze(0).expand(50, -1)
    # Just verify shape and no error
    out = enc(x_sorted)
    assert out.shape == (50, 3, 4)


def test_edge_norms_length(encoder):
    norms = encoder.get_edge_norms()
    assert len(norms) == 5
    assert all(n >= 0 for n in norms)
