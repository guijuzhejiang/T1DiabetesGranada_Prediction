"""U8 单测 - SHAP 计算 + 4 类图绘制."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from t1d_granada.model import xLSTMRegressor
from t1d_granada.shap_analysis import (
    compute_shap,
    plot_feature_importance,
    plot_force_samples,
    plot_time_feature_heatmap,
    plot_timestep_importance,
    write_summary_csv,
)


@pytest.fixture
def mini_model_and_data():
    model = xLSTMRegressor(
        d_seq=6, d_static=12, embedding_dim=32, num_blocks=2, mlstm_ratio=1.0,
        mlp_hidden=32, dropout=0.0, conv_kernel_size=2, context_length=4,
    ).eval()
    rng = np.random.RandomState(0)
    bg_seq = rng.randn(16, 4, 6).astype(np.float32)
    bg_static = rng.randn(16, 12).astype(np.float32)
    fg_seq = rng.randn(32, 4, 6).astype(np.float32)
    fg_static = rng.randn(32, 12).astype(np.float32)
    return model, bg_seq, bg_static, fg_seq, fg_static


def test_compute_shap_shapes(mini_model_and_data):
    model, bg_seq, bg_static, fg_seq, fg_static = mini_model_and_data
    shap_seq, shap_static = compute_shap(
        model, bg_seq, bg_static, fg_seq, fg_static,
        device=torch.device("cpu"), batch_size=8,
    )
    assert shap_seq.shape == (32, 4, 6)
    assert shap_static.shape == (32, 12)
    assert not np.isnan(shap_seq).any()
    assert not np.isnan(shap_static).any()


def test_plot_feature_importance_creates_png(mini_model_and_data, tmp_path):
    model, bg_seq, bg_static, fg_seq, fg_static = mini_model_and_data
    shap_seq, shap_static = compute_shap(
        model, bg_seq, bg_static, fg_seq, fg_static,
        device=torch.device("cpu"), batch_size=8,
    )
    seq_names = ["bg_z", "bg_diff", "tod_sin", "tod_cos", "dow_sin", "dow_cos"]
    static_names = ["Sex_M", "Age"] + [f"stat_{i}" for i in range(10)]
    out = tmp_path / "fi.png"
    plot_feature_importance(shap_seq, shap_static, seq_names, static_names, out)
    assert out.exists()
    assert out.stat().st_size > 1000


def test_summary_csv_columns(mini_model_and_data, tmp_path):
    model, bg_seq, bg_static, fg_seq, fg_static = mini_model_and_data
    shap_seq, shap_static = compute_shap(
        model, bg_seq, bg_static, fg_seq, fg_static,
        device=torch.device("cpu"), batch_size=8,
    )
    seq_names = ["bg_z", "bg_diff", "tod_sin", "tod_cos", "dow_sin", "dow_cos"]
    static_names = ["Sex_M", "Age"] + [f"stat_{i}" for i in range(10)]
    out = tmp_path / "summary.csv"
    write_summary_csv(shap_seq, shap_static, seq_names, static_names, out)
    import csv
    with open(out) as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["feature_name", "mean_abs_shap", "rank"]
    assert len(rows) - 1 == len(seq_names) + len(static_names) == 18
    # ranks monotonic
    ranks = [int(r[2]) for r in rows[1:]]
    assert ranks == list(range(1, len(ranks) + 1))


def test_all_plots_pipeline(mini_model_and_data, tmp_path):
    model, bg_seq, bg_static, fg_seq, fg_static = mini_model_and_data
    shap_seq, shap_static = compute_shap(
        model, bg_seq, bg_static, fg_seq, fg_static,
        device=torch.device("cpu"), batch_size=8,
    )
    seq_names = ["bg_z", "bg_diff", "tod_sin", "tod_cos", "dow_sin", "dow_cos"]
    static_names = ["Sex_M", "Age"] + [f"stat_{i}" for i in range(10)]
    plot_feature_importance(shap_seq, shap_static, seq_names, static_names, tmp_path / "fi.png")
    plot_timestep_importance(shap_seq, tmp_path / "ts.png")
    plot_time_feature_heatmap(shap_seq, seq_names, tmp_path / "tf.png")
    # predictions for force plots
    with torch.no_grad():
        preds = model(torch.from_numpy(fg_seq), torch.from_numpy(fg_static)).numpy()
    plot_force_samples(shap_seq, shap_static, fg_seq, fg_static, preds,
                       seq_names, static_names, tmp_path / "force", n_samples=3)
    for f in ["fi.png", "ts.png", "tf.png"]:
        assert (tmp_path / f).exists()
    force_files = list((tmp_path / "force").glob("*.png"))
    assert 1 <= len(force_files) <= 3


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_plot_force_samples_small_n(tmp_path, n):
    """Edge case: foregrounds smaller than the highest+lowest+3-medians template.

    Regression guard for the previous `order[1:n_samples - 2]` slicing logic, which
    silently produced overlapping/empty selections for n in {2, 3, 4}.
    """
    rng = np.random.RandomState(0)
    T, D_seq, D_static = 4, 6, 12
    shap_seq = rng.randn(n, T, D_seq).astype(np.float32)
    shap_static = rng.randn(n, D_static).astype(np.float32)
    fg_seq = rng.randn(n, T, D_seq).astype(np.float32)
    fg_static = rng.randn(n, D_static).astype(np.float32)
    preds = np.linspace(80.0, 200.0, n).astype(np.float32)
    seq_names = [f"s{i}" for i in range(D_seq)]
    static_names = [f"x{i}" for i in range(D_static)]

    out_dir = tmp_path / f"force_n{n}"
    plot_force_samples(
        shap_seq, shap_static, fg_seq, fg_static, preds,
        seq_names, static_names, out_dir, n_samples=5,
    )

    files = sorted(out_dir.glob("*.png"))
    if n < 5:
        # Every distinct sample should be plotted exactly once.
        assert len(files) == n, f"n={n}: expected {n} plots, got {len(files)}"
    else:
        # Highest, lowest, and 3 medians — all distinct for n=5.
        assert len(files) == 5

    # Filenames embed the sample index; confirm no collisions and all in [0, n).
    indices = {int(f.stem.split("_")[-1]) for f in files}
    assert len(indices) == len(files)
    assert all(0 <= ix < n for ix in indices)


def test_plot_force_samples_n_zero_returns(tmp_path):
    """n=0 must short-circuit and produce no files (and no division-by-zero)."""
    out_dir = tmp_path / "force_empty"
    plot_force_samples(
        np.empty((0, 4, 6), dtype=np.float32),
        np.empty((0, 12), dtype=np.float32),
        np.empty((0, 4, 6), dtype=np.float32),
        np.empty((0, 12), dtype=np.float32),
        np.empty((0,), dtype=np.float32),
        [f"s{i}" for i in range(6)],
        [f"x{i}" for i in range(12)],
        out_dir, n_samples=5,
    )
    assert not out_dir.exists() or not list(out_dir.glob("*.png"))


def test_d_seq_one_still_works(mini_model_and_data):
    """D_seq=1 (all flags off) edge case."""
    model = xLSTMRegressor(
        d_seq=1, d_static=2, embedding_dim=32, num_blocks=2, mlstm_ratio=1.0,
        mlp_hidden=32, dropout=0.0, conv_kernel_size=2, context_length=4,
    ).eval()
    rng = np.random.RandomState(0)
    bg_seq = rng.randn(8, 4, 1).astype(np.float32)
    bg_static = rng.randn(8, 2).astype(np.float32)
    fg_seq = rng.randn(8, 4, 1).astype(np.float32)
    fg_static = rng.randn(8, 2).astype(np.float32)
    shap_seq, shap_static = compute_shap(
        model, bg_seq, bg_static, fg_seq, fg_static,
        device=torch.device("cpu"), batch_size=4,
    )
    assert shap_seq.shape == (8, 4, 1)
    assert shap_static.shape == (8, 2)
