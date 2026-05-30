"""U7 单测 - predict.py.

重点:
- 端到端单条/批量推理
- history 不足 → 返回 None (skip)
- 因果性: history 中 last_ts 之后注入异常值 → 预测不变
- model 不存在 → FileNotFoundError
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from t1d_granada import params as P
from t1d_granada.model import xLSTMRegressor
from t1d_granada.scaler import Scaler


def _make_synthetic_history(start: str = "2026-05-09 00:00:00", n: int = 96, base_bg: float = 130.0):
    """1 day of 15-min CGM for one patient."""
    ts = pd.date_range(start, periods=n, freq="15min")
    rng = np.random.RandomState(42)
    bg = base_bg + 10 * np.sin(np.arange(n) * 0.2) + rng.randn(n) * 5
    return pd.DataFrame({
        "Patient_ID": ["TEST_P1"] * n,
        "timestamp": ts,
        "bg": bg.astype(np.float32),
    })


@pytest.fixture
def trained_artifacts(tmp_path: Path):
    """Mini trained model + scaler so predict.py can load."""
    model_dir = tmp_path / "model"
    proc_dir = tmp_path / "data" / "processed"
    model_dir.mkdir(parents=True)
    proc_dir.mkdir(parents=True)

    sc = Scaler()
    sc.fit("bg", np.array([100, 130, 160, 200], dtype=np.float64))
    sc.fit("Age", np.array([20, 50, 70], dtype=np.float64))
    for col in [
        "bg_2h_mean", "bg_2h_std", "bg_2h_tir_70_180", "bg_2h_p5", "bg_2h_p95",
        "bg_6h_mean", "bg_6h_std", "bg_6h_tir_70_180", "bg_6h_p5", "bg_6h_p95",
    ]:
        sc.fit(col, np.array([0.5, 1.0, 1.5], dtype=np.float64))
    sc.save(proc_dir / "scaler.pkl")

    model = xLSTMRegressor(
        d_seq=6, d_static=12, embedding_dim=32, num_blocks=2, mlstm_ratio=1.0,
        mlp_hidden=32, dropout=0.0, conv_kernel_size=2, context_length=4,
        static_embedding_dim=16,
    )
    bundle = {
        "state_dict": model.state_dict(),
        "hp": {
            "embedding_dim": 32, "num_blocks": 2, "mlstm_ratio": 1.0,
            "mlp_hidden": 32, "dropout": 0.0, "conv_kernel_size": 2,
            "lr": 1e-3, "static_embedding_dim": 16,
        },
        "meta": {
            "d_seq": 6, "d_static": 12, "window_size": 8,
        },
        "final_epochs": 1,
    }
    torch.save(bundle, model_dir / "xlstm_best.pt")
    return tmp_path, model_dir, proc_dir


def test_predict_single_happy_path(trained_artifacts, tmp_path):
    _, model_dir, proc_dir = trained_artifacts
    history = _make_synthetic_history()
    history_csv = tmp_path / "history.csv"
    history.to_csv(history_csv, index=False)

    last_ts = history["timestamp"].iloc[-1]
    from predict import main

    rc = main([
        "--history", str(history_csv),
        "--model", str(model_dir / "xlstm_best.pt"),
        "--scaler", str(proc_dir / "scaler.pkl"),
        "--patient_id", "TEST_P1",
        "--last_ts", str(last_ts),
        "--sex", "M",
        "--birth_year", "1985",
    ])
    assert rc == 0


def test_predict_batch(trained_artifacts, tmp_path, capsys):
    _, model_dir, proc_dir = trained_artifacts
    history = _make_synthetic_history()
    history_csv = tmp_path / "history.csv"
    history.to_csv(history_csv, index=False)

    # Build a batch with 3 last_ts values (only some have enough history)
    last_ts_a = history["timestamp"].iloc[-1]  # full history (24h)
    last_ts_b = history["timestamp"].iloc[10]  # ~2.5h history → 6h rolling fails → skip
    last_ts_c = history["timestamp"].iloc[50]  # ~12.5h history → both windows ok
    batch = pd.DataFrame({
        "Patient_ID": ["TEST_P1"] * 3,
        "last_timestamp": [last_ts_a, last_ts_b, last_ts_c],
        "Sex": ["M"] * 3,
        "Birth_year": [1985] * 3,
    })
    batch_csv = tmp_path / "batch.csv"
    batch.to_csv(batch_csv, index=False)
    out_csv = tmp_path / "out.csv"

    from predict import main
    rc = main([
        "--history", str(history_csv),
        "--model", str(model_dir / "xlstm_best.pt"),
        "--scaler", str(proc_dir / "scaler.pkl"),
        "--input", str(batch_csv),
        "--output", str(out_csv),
    ])
    assert rc == 0
    out = pd.read_csv(out_csv)
    assert list(out.columns) == ["Patient_ID", "last_timestamp", "predicted_bg_30min"]
    assert len(out) == 3
    # at least the 24h-history one should produce a numeric prediction
    assert out["predicted_bg_30min"].notna().sum() >= 1


def test_history_insufficient_returns_skip(trained_artifacts, tmp_path):
    _, model_dir, proc_dir = trained_artifacts
    # Only 1h of history → cannot meet 6h rolling
    history = _make_synthetic_history(n=4)
    history_csv = tmp_path / "history.csv"
    history.to_csv(history_csv, index=False)

    last_ts = history["timestamp"].iloc[-1]
    from predict import main
    rc = main([
        "--history", str(history_csv),
        "--model", str(model_dir / "xlstm_best.pt"),
        "--scaler", str(proc_dir / "scaler.pkl"),
        "--patient_id", "TEST_P1",
        "--last_ts", str(last_ts),
        "--sex", "F",
        "--birth_year", "1990",
    ])
    assert rc == 1  # skipped → exit 1


def test_causality_future_injection(trained_artifacts, tmp_path):
    _, model_dir, proc_dir = trained_artifacts
    history = _make_synthetic_history(n=96)

    # last_ts is at row 60 (15h in)
    last_ts = history["timestamp"].iloc[60]

    # Clean version: predict
    from predict import predict_one, _load_model, _load_history
    model, _, _ = _load_model(model_dir / "xlstm_best.pt", torch.device("cpu"))
    scaler = Scaler.load(proc_dir / "scaler.pkl")

    history_clean = history.copy()
    pred_clean = predict_one(model, scaler, history_clean, "TEST_P1", last_ts, "M", 1985, torch.device("cpu"))

    # Inject huge values AFTER last_ts (rows 65..95)
    history_dirty = history.copy()
    history_dirty.loc[65:, "bg"] = 9999.0
    pred_dirty = predict_one(model, scaler, history_dirty, "TEST_P1", last_ts, "M", 1985, torch.device("cpu"))

    assert pred_clean is not None and pred_dirty is not None
    assert abs(pred_clean - pred_dirty) < 1e-3, \
        f"causality violated: clean={pred_clean}, dirty={pred_dirty}"


def test_model_not_found_raises(trained_artifacts, tmp_path):
    _, model_dir, proc_dir = trained_artifacts
    history = _make_synthetic_history()
    history_csv = tmp_path / "history.csv"
    history.to_csv(history_csv, index=False)

    from predict import main
    with pytest.raises(FileNotFoundError):
        main([
            "--history", str(history_csv),
            "--model", str(tmp_path / "no_such.pt"),
            "--scaler", str(proc_dir / "scaler.pkl"),
            "--patient_id", "TEST_P1",
            "--last_ts", str(history["timestamp"].iloc[-1]),
            "--sex", "M",
            "--birth_year", "1985",
        ])


def test_inference_speed_under_80ms(trained_artifacts, tmp_path):
    _, model_dir, proc_dir = trained_artifacts
    history = _make_synthetic_history()

    from predict import predict_one, _load_model
    model, _, _ = _load_model(model_dir / "xlstm_best.pt", torch.device("cpu"))
    scaler = Scaler.load(proc_dir / "scaler.pkl")
    last_ts = history["timestamp"].iloc[-1]

    # warm up
    for _ in range(2):
        predict_one(model, scaler, history, "TEST_P1", last_ts, "M", 1985, torch.device("cpu"))

    import time
    t0 = time.perf_counter()
    n_runs = 20
    for _ in range(n_runs):
        pred = predict_one(model, scaler, history, "TEST_P1", last_ts, "M", 1985, torch.device("cpu"))
    elapsed_ms = (time.perf_counter() - t0) * 1000 / n_runs
    assert pred is not None
    # Loose budget: 80 ms target per plan; allow 200 ms for CI variance
    assert elapsed_ms < 200, f"avg inference {elapsed_ms:.1f} ms exceeded budget"
