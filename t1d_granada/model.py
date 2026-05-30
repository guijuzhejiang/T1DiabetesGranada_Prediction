"""xLSTMRegressor: 输入投影 → xLSTMBlockStack → 取最后步 → concat(static_emb) → MLP → 标量。

设计要点 (R5):
- 输入投影 = Linear(d_seq → embedding_dim) + LayerNorm. 单变量 + sin/cos 量纲差距大, LN 必要.
- 主干 xLSTMBlockStack 通过 mlstm_ratio ∈ {0, 0.5, 1.0} 调整 sLSTM 占比.
  · ratio=1.0 → 纯 mLSTM (slstm_at=[])
  · ratio=0.0 → 纯 sLSTM (slstm_at=[0..num_blocks-1])
  · ratio=0.5 → 交错放置 (奇数 index 是 sLSTM)
- sLSTM backend 固定为 'vanilla' (纯 torch op): GPU 与 CPU 都跑得通, SHAP 也兼容.
- 静态特征 encoder (可选, static_embedding_dim 控制):
  · None / 0 → 走"裸 concat"老路径 (head 第一层直接吃 d_static 维)
  · int > 0 → 先 Linear → LayerNorm → GELU → Dropout 投到 static_embedding_dim,
    让 12 维静态特征间能预交互, 再与 seq 末步 concat.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from xlstm import (
    FeedForwardConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
)


def _slstm_indices(num_blocks: int, ratio: float) -> list[int]:
    """Compute which block indices are sLSTM given the mLSTM ratio.

    ratio = fraction of blocks that are mLSTM. 1.0 → all mLSTM (slstm_at=[]).
    0.0 → all sLSTM. 0.5 → alternate, sLSTM at odd indices.
    """
    if ratio >= 1.0 - 1e-6:
        return []
    if ratio <= 1e-6:
        return list(range(num_blocks))
    # mid: place sLSTM at odd indices (1, 3, ...)
    return [i for i in range(num_blocks) if i % 2 == 1]


class xLSTMRegressor(nn.Module):
    """Sequence-to-scalar regression head over an xLSTM trunk."""

    def __init__(
        self,
        d_seq: int,
        d_static: int,
        *,
        embedding_dim: int = 128,
        num_blocks: int = 4,
        mlstm_ratio: float = 1.0,
        mlp_hidden: int = 128,
        dropout: float = 0.1,
        conv_kernel_size: int = 4,
        context_length: int = 4,
        num_heads: int = 4,
        slstm_backend: str = "vanilla",
        static_embedding_dim: int | None = 32,
    ):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim={embedding_dim} must be divisible by num_heads={num_heads}"
            )

        self.d_seq = d_seq
        self.d_static = d_static
        self.embedding_dim = embedding_dim
        self.static_embedding_dim = static_embedding_dim

        # Input projection
        self.input_proj = nn.Linear(d_seq, embedding_dim)
        self.input_norm = nn.LayerNorm(embedding_dim)

        # Static encoder (可选). None / 0 → 走 raw concat
        if static_embedding_dim and static_embedding_dim > 0:
            self.static_proj: nn.Module | None = nn.Sequential(
                nn.Linear(d_static, static_embedding_dim),
                nn.LayerNorm(static_embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            head_static_dim = static_embedding_dim
        else:
            self.static_proj = None
            head_static_dim = d_static

        # xLSTM trunk
        slstm_at = _slstm_indices(num_blocks, mlstm_ratio)
        mlstm_block = mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                num_heads=num_heads,
                conv1d_kernel_size=conv_kernel_size,
                dropout=dropout,
            )
        )
        slstm_block = (
            sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    num_heads=num_heads,
                    conv1d_kernel_size=conv_kernel_size,
                    backend=slstm_backend,
                    dropout=dropout,
                ),
                feedforward=FeedForwardConfig(dropout=dropout),
            )
            if slstm_at
            else None
        )
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mlstm_block,
            slstm_block=slstm_block,
            context_length=context_length,
            num_blocks=num_blocks,
            embedding_dim=embedding_dim,
            slstm_at=slstm_at,
            dropout=dropout,
        )
        self.trunk = xLSTMBlockStack(cfg)

        # Head MLP: (embedding_dim + head_static_dim) → mlp_hidden → 1
        self.head = nn.Sequential(
            nn.Linear(embedding_dim + head_static_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        """seq: (B, T, d_seq), static: (B, d_static) → (B,) scalar."""
        x = self.input_proj(seq)            # (B, T, E)
        x = self.input_norm(x)
        x = self.trunk(x)                   # (B, T, E)
        last = x[:, -1, :]                  # (B, E)
        s = self.static_proj(static) if self.static_proj is not None else static
        h = torch.cat([last, s], dim=-1)    # (B, E + head_static_dim)
        out = self.head(h).squeeze(-1)      # (B,)
        return out
