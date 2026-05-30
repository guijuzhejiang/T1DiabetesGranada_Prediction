"""U4 单测 - xLSTMRegressor."""
from __future__ import annotations

import pytest
import torch

from t1d_granada.model import xLSTMRegressor, _slstm_indices


def test_slstm_indices():
    assert _slstm_indices(4, 1.0) == []
    assert _slstm_indices(4, 0.0) == [0, 1, 2, 3]
    assert _slstm_indices(4, 0.5) == [1, 3]
    assert _slstm_indices(2, 0.5) == [1]


def test_forward_shape_pure_mlstm():
    model = xLSTMRegressor(
        d_seq=6, d_static=12, embedding_dim=64, num_blocks=2, mlstm_ratio=1.0,
        mlp_hidden=64, dropout=0.1, conv_kernel_size=2, context_length=4,
    )
    seq = torch.randn(2, 4, 6)
    static = torch.randn(2, 12)
    out = model(seq, static)
    assert out.shape == (2,)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_forward_pure_slstm():
    model = xLSTMRegressor(
        d_seq=6, d_static=12, embedding_dim=64, num_blocks=2, mlstm_ratio=0.0,
        mlp_hidden=64, dropout=0.0, conv_kernel_size=2, context_length=4,
    )
    out = model(torch.randn(2, 4, 6), torch.randn(2, 12))
    assert out.shape == (2,)


def test_forward_mixed():
    model = xLSTMRegressor(
        d_seq=6, d_static=12, embedding_dim=64, num_blocks=4, mlstm_ratio=0.5,
        mlp_hidden=64, dropout=0.0, conv_kernel_size=2, context_length=4,
    )
    out = model(torch.randn(2, 4, 6), torch.randn(2, 12))
    assert out.shape == (2,)


def test_backward_grads_nonzero():
    model = xLSTMRegressor(
        d_seq=6, d_static=12, embedding_dim=64, num_blocks=2, mlstm_ratio=1.0,
        mlp_hidden=64, dropout=0.0, conv_kernel_size=2, context_length=4,
    )
    out = model(torch.randn(4, 4, 6), torch.randn(4, 12))
    loss = out.sum()
    loss.backward()
    n_with_grad = 0
    n_total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        n_total += 1
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            n_with_grad += 1
    assert n_total > 0
    # at least 80% of parameters should receive non-zero grad
    assert n_with_grad / n_total > 0.8, f"only {n_with_grad}/{n_total} params have nonzero grad"


@pytest.mark.parametrize("num_blocks", [2, 6])
def test_block_depth(num_blocks):
    model = xLSTMRegressor(
        d_seq=6, d_static=12, embedding_dim=64, num_blocks=num_blocks, mlstm_ratio=1.0,
        mlp_hidden=64, dropout=0.0, conv_kernel_size=2, context_length=4,
    )
    out = model(torch.randn(2, 4, 6), torch.randn(2, 12))
    assert out.shape == (2,)


@pytest.mark.parametrize("k", [2, 3, 4])
def test_conv_kernel_size(k):
    model = xLSTMRegressor(
        d_seq=6, d_static=12, embedding_dim=64, num_blocks=2, mlstm_ratio=1.0,
        mlp_hidden=64, dropout=0.0, conv_kernel_size=k, context_length=4,
    )
    out = model(torch.randn(2, 4, 6), torch.randn(2, 12))
    assert out.shape == (2,)


@pytest.mark.parametrize("static_embedding_dim", [None, 0, 16, 32, 64])
def test_static_embedding_forward(static_embedding_dim):
    """static encoder 各种取值都应该能正常前向到 (B,) 标量。"""
    model = xLSTMRegressor(
        d_seq=6, d_static=12, embedding_dim=64, num_blocks=2, mlstm_ratio=1.0,
        mlp_hidden=64, dropout=0.0, conv_kernel_size=2, context_length=4,
        static_embedding_dim=static_embedding_dim,
    )
    out = model(torch.randn(2, 4, 6), torch.randn(2, 12))
    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_static_encoder_disabled_vs_enabled_differ():
    """关掉 encoder 与启用 encoder 应当是不同的 module 结构。"""
    m_off = xLSTMRegressor(
        d_seq=6, d_static=12, embedding_dim=64, num_blocks=2, mlstm_ratio=1.0,
        mlp_hidden=64, dropout=0.0, conv_kernel_size=2, context_length=4,
        static_embedding_dim=0,
    )
    m_on = xLSTMRegressor(
        d_seq=6, d_static=12, embedding_dim=64, num_blocks=2, mlstm_ratio=1.0,
        mlp_hidden=64, dropout=0.0, conv_kernel_size=2, context_length=4,
        static_embedding_dim=32,
    )
    assert m_off.static_proj is None
    assert m_on.static_proj is not None
    # head 第一层 in_features 应反映 encoder 是否启用
    assert m_off.head[0].in_features == 64 + 12      # raw concat
    assert m_on.head[0].in_features == 64 + 32       # encoded


def test_static_encoder_default_is_32():
    """构造器默认 static_embedding_dim=32(用户指定的默认)。"""
    model = xLSTMRegressor(
        d_seq=6, d_static=12, embedding_dim=64, num_blocks=2, mlstm_ratio=1.0,
        mlp_hidden=64, dropout=0.0, conv_kernel_size=2, context_length=4,
    )
    assert model.static_embedding_dim == 32
    assert model.static_proj is not None
    assert model.head[0].in_features == 64 + 32


def test_static_encoder_grads_flow():
    """启用 encoder 时,梯度应当能流到 static_proj 的参数上。"""
    model = xLSTMRegressor(
        d_seq=6, d_static=12, embedding_dim=64, num_blocks=2, mlstm_ratio=1.0,
        mlp_hidden=64, dropout=0.0, conv_kernel_size=2, context_length=4,
        static_embedding_dim=32,
    )
    out = model(torch.randn(4, 4, 6), torch.randn(4, 12))
    out.sum().backward()
    static_proj_grads = [p.grad for p in model.static_proj.parameters() if p.requires_grad]
    assert all(g is not None and g.abs().sum().item() > 0 for g in static_proj_grads)
