"""SHAP 计算 + 4 类图 (R10).

策略 (resolves plan F1):
- 用 `shap.GradientExplainer(model, [bg_seq, bg_static])` 列表形式; SHAP 原生支持多输入 nn.Module
- 模型在 SHAP 调用前 `eval()`; 所有张量在同一 device
- backend='vanilla' 的 sLSTM 在 GradientExplainer 下可微 (resolves plan D1)

返回:
- shap_seq: (N, T, D_seq)
- shap_static: (N, D_static)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class _SHAPWrapper(nn.Module):
    """SHAP's GradientExplainer slices `outputs[:, idx]`, so the wrapped model must
    return a (B, n_out) tensor, not (B,). Our regressor squeezes the last dim --
    here we re-add it so SHAP works."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        out = self.inner(seq, static)
        if out.ndim == 1:
            out = out.unsqueeze(-1)
        return out


def compute_shap(
    model: nn.Module,
    bg_seq: np.ndarray,
    bg_static: np.ndarray,
    fg_seq: np.ndarray,
    fg_static: np.ndarray,
    *,
    device: torch.device | None = None,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values for a multi-input PyTorch regressor.

    bg_*: background tensors (typically 256).
    fg_*: foreground tensors to explain (typically 1024).
    """
    import shap

    if device is None:
        device = next(model.parameters()).device

    model = model.eval()
    wrapped = _SHAPWrapper(model).to(device).eval()

    bg_seq_t = torch.from_numpy(bg_seq.astype(np.float32)).to(device)
    bg_static_t = torch.from_numpy(bg_static.astype(np.float32)).to(device)
    fg_seq_t = torch.from_numpy(fg_seq.astype(np.float32)).to(device)
    fg_static_t = torch.from_numpy(fg_static.astype(np.float32)).to(device)

    explainer = shap.GradientExplainer(wrapped, [bg_seq_t, bg_static_t])
    # Process foreground in chunks to manage memory
    seq_parts: list[np.ndarray] = []
    static_parts: list[np.ndarray] = []
    n = fg_seq_t.shape[0]
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        chunk_shap = explainer.shap_values([fg_seq_t[s:e], fg_static_t[s:e]])
        # GradientExplainer returns list of arrays for multi-input. With our (B, 1)
        # wrapper output it appends a trailing n_outputs=1 dim -- squeeze it.
        seq_part = np.asarray(chunk_shap[0])
        static_part = np.asarray(chunk_shap[1])
        if seq_part.ndim == 4 and seq_part.shape[-1] == 1:
            seq_part = seq_part.squeeze(-1)
        if static_part.ndim == 3 and static_part.shape[-1] == 1:
            static_part = static_part.squeeze(-1)
        seq_parts.append(seq_part)
        static_parts.append(static_part)
    return np.concatenate(seq_parts, axis=0), np.concatenate(static_parts, axis=0)


def write_summary_csv(
    shap_seq: np.ndarray, shap_static: np.ndarray,
    seq_feature_names: list[str], static_feature_names: list[str],
    out_path: Path,
) -> None:
    """Aggregate |SHAP| across samples (and across time for seq features) and rank."""
    seq_imp = np.abs(shap_seq).mean(axis=(0, 1))   # (D_seq,)
    static_imp = np.abs(shap_static).mean(axis=0)  # (D_static,)
    names = list(seq_feature_names) + list(static_feature_names)
    vals = np.concatenate([seq_imp, static_imp])
    order = np.argsort(-vals)
    rows = []
    for rank, idx in enumerate(order, start=1):
        rows.append((names[idx], float(vals[idx]), rank))
    import csv
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_name", "mean_abs_shap", "rank"])
        for r in rows:
            w.writerow(r)


def plot_feature_importance(
    shap_seq: np.ndarray, shap_static: np.ndarray,
    seq_feature_names: list[str], static_feature_names: list[str],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    seq_imp = np.abs(shap_seq).mean(axis=(0, 1))
    static_imp = np.abs(shap_static).mean(axis=0)
    names = list(seq_feature_names) + list(static_feature_names)
    vals = np.concatenate([seq_imp, static_imp])
    order = np.argsort(vals)  # ascending so largest at top in barh
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(names))))
    ax.barh([names[i] for i in order], vals[order])
    ax.set_xlabel("mean(|SHAP|)")
    ax.set_title("Feature importance (seq pooled across timesteps + static)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_timestep_importance(shap_seq: np.ndarray, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    # average over samples and feature channels → per-time importance
    t_imp = np.abs(shap_seq).mean(axis=(0, 2))
    T = t_imp.size
    labels = [f"t-{T - 1 - i}" if i < T - 1 else "t-0" for i in range(T)]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, t_imp)
    ax.set_ylabel("mean(|SHAP|)")
    ax.set_title("Time-step importance")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_time_feature_heatmap(
    shap_seq: np.ndarray, seq_feature_names: list[str], out_path: Path
) -> None:
    import matplotlib.pyplot as plt

    grid = np.abs(shap_seq).mean(axis=0)  # (T, D_seq)
    T, D = grid.shape
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, aspect="auto", cmap="viridis")
    ax.set_xticks(range(D))
    ax.set_xticklabels(seq_feature_names, rotation=45, ha="right")
    ax.set_yticks(range(T))
    ax.set_yticklabels([f"t-{T - 1 - i}" if i < T - 1 else "t-0" for i in range(T)])
    ax.set_title("Time × feature mean(|SHAP|)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_force_samples(
    shap_seq: np.ndarray, shap_static: np.ndarray,
    fg_seq: np.ndarray, fg_static: np.ndarray, predictions: np.ndarray,
    seq_feature_names: list[str], static_feature_names: list[str],
    out_dir: Path, n_samples: int = 5,
) -> None:
    """Pick samples (highest, lowest, 3 medians) and draw a waterfall-style chart."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    n = predictions.shape[0]
    if n == 0:
        return
    order = np.argsort(predictions)
    if n >= 5:
        mid = n // 2
        picks = [order[-1], order[0], order[mid - 1], order[mid], order[mid + 1]]
    else:
        # Fewer than 5 foreground samples: plot every one (capped by n_samples).
        picks = list(order)
    candidates = list(dict.fromkeys(int(i) for i in picks))[:n_samples]

    feat_names_full = (
        [f"{name}@t-{shap_seq.shape[1]-1-t}" if t < shap_seq.shape[1]-1 else f"{name}@t-0"
         for t in range(shap_seq.shape[1]) for name in seq_feature_names]
        + list(static_feature_names)
    )
    for ix in candidates:
        seq_flat = shap_seq[ix].ravel()  # (T * D_seq,)
        static_flat = shap_static[ix].ravel()
        vals = np.concatenate([seq_flat, static_flat])
        order_v = np.argsort(np.abs(vals))[-15:]  # top 15
        labels = [feat_names_full[i] for i in order_v]
        contribs = vals[order_v]
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = ["tab:red" if v > 0 else "tab:blue" for v in contribs]
        ax.barh(labels, contribs, color=colors)
        ax.set_title(f"Sample {ix}: pred={predictions[ix]:.1f} mg/dL — top SHAP")
        ax.set_xlabel("SHAP value")
        ax.axvline(0, color="black", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{ix:05d}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
