"""U6 smoke test: 用 mini npy 验证 train_optuna.py + train.py 两阶段端到端。"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from t1d_granada import params as P
from t1d_granada.scaler import Scaler


@pytest.fixture
def mini_dataset(tmp_path: Path, monkeypatch) -> Path:
    """Create a mini processed dir with realistic shapes; redirect settings paths.

    npy + scaler + meta 都落到 WINDOW_SIZE_<T>/ 子目录, 与生产代码 (utils.processed_data_dir)
    返回的路径保持一致。
    """
    T, D_seq, D_static = 8, 6, 12
    proc_root = tmp_path / "data" / "processed"
    proc = proc_root / f"WINDOW_SIZE_{T}"
    proc.mkdir(parents=True)
    model_dir = tmp_path / "model"
    tuning_dir = tmp_path / "hyperparameter_tuning"
    mlruns = tmp_path / "mlruns"

    rng = np.random.RandomState(0)
    n_train, n_val, n_test = 96, 32, 32
    raw_bg = rng.randn(1000) * 50 + 150  # mg/dL-ish
    sc = Scaler()
    sc.fit("bg", raw_bg)
    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        np.save(proc / f"{split}_seq.npy", rng.randn(n, T, D_seq).astype(np.float32))
        np.save(proc / f"{split}_static.npy", rng.randn(n, D_static).astype(np.float32))
        # target = simple func of seq[:,:,0] mean → learnable signal
        seq = np.load(proc / f"{split}_seq.npy")
        tgt = (seq[:, :, 0].mean(axis=1) + 0.1 * rng.randn(n)).astype(np.float32)
        np.save(proc / f"{split}_target.npy", tgt)
    sc.save(proc / "scaler.pkl")
    meta = {
        "window_size": T, "d_seq": D_seq, "d_static": D_static,
        "counts": {"train": n_train, "val": n_val, "test": n_test},
    }
    (proc / "meta.json").write_text(json.dumps(meta))

    # patch load_settings + override params for fast test
    from t1d_granada import utils as utils_mod

    def fake_settings():
        return {
            # 给 root 路径,utils.processed_data_dir() 会再追加 WINDOW_SIZE_<T>/ 子目录
            "PROCESSED_DATA_DIR": str(proc_root),
            "MODEL_DIR": str(model_dir),
            "HYPERPARAMETER_TUNING_DIR": str(tuning_dir),
            "MLFLOW_TRACKING_URI": str(mlruns),
        }

    monkeypatch.setattr(utils_mod, "load_settings", fake_settings)
    monkeypatch.setattr(P, "MAX_EPOCHS", 2)
    monkeypatch.setattr(P, "PATIENCE", 5)
    monkeypatch.setattr(P, "NUM_WORKERS", 0)
    monkeypatch.setattr(P, "N_TRIALS", 2)
    monkeypatch.setattr(P, "FINAL_FIT_EPOCH_MULT", 1.0)
    # tighter Optuna search space for speed
    monkeypatch.setattr(P, "BATCH_SIZE", 16)
    monkeypatch.setattr(P, "OPTUNA_SEARCH_SPACE", {
        "embedding_dim": {"type": "categorical", "choices": [32]},
        "num_blocks": {"type": "int", "low": 2, "high": 2},
        "mlstm_ratio": {"type": "categorical", "choices": [1.0]},
        "mlp_hidden": {"type": "categorical", "choices": [32]},
        "dropout": {"type": "categorical", "choices": [0.0]},
        "conv_kernel_size": {"type": "categorical", "choices": [2]},
        "lr": {"type": "categorical", "choices": [1e-3]},
        "static_embedding_dim": {"type": "categorical", "choices": [16]},
    })
    return tmp_path


def _force_cpu(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    if torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_train_optuna_then_train_smoke(mini_dataset, monkeypatch):
    """train_optuna.py 写出 best_params + best_trial_meta + study.pkl;
    train.py 读它们,跑最终 fit,产出 xlstm_best.pt + test 指标。"""
    _force_cpu(monkeypatch)

    from train_optuna import main as optuna_main

    rc = optuna_main()
    assert rc == 0

    tuning_dir = mini_dataset / "hyperparameter_tuning"
    best_params_path = tuning_dir / "best_params.json"
    trial_meta_path = tuning_dir / "best_trial_meta.json"
    study_path = tuning_dir / "study.pkl"
    assert best_params_path.exists()
    assert trial_meta_path.exists()
    assert study_path.exists()

    bp = json.loads(best_params_path.read_text())
    expected = {"embedding_dim", "num_blocks", "mlstm_ratio", "mlp_hidden", "dropout",
                "conv_kernel_size", "lr", "static_embedding_dim"}
    assert set(bp.keys()) == expected

    tm = json.loads(trial_meta_path.read_text())
    assert {"trial_number", "best_val_rmse", "last_epoch"}.issubset(tm.keys())
    assert isinstance(tm["last_epoch"], int)

    # SQLite db 应当被默认创建
    db_path = tuning_dir / "xlstm_search.db"
    assert db_path.exists() and db_path.stat().st_size > 0, \
        f"expected sqlite study db at {db_path}"

    # train_optuna 不应再产出 model/xlstm_best.pt
    assert not (mini_dataset / "model" / "xlstm_best.pt").exists()

    # 现在跑 train.py
    from train import main as train_main

    rc = train_main([])
    assert rc == 0
    model_path = mini_dataset / "model" / "xlstm_best.pt"
    assert model_path.exists()
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    assert set(bundle.keys()) == {
        "state_dict", "hp", "meta", "max_epochs", "stopped_at_epoch", "best_val_rmse",
    }
    assert bundle["hp"] == bp
    assert isinstance(bundle["stopped_at_epoch"], int)
    assert bundle["stopped_at_epoch"] < bundle["max_epochs"]


def test_train_optuna_missing_processed_dir_errors(tmp_path, monkeypatch):
    from t1d_granada import utils as utils_mod

    def fake_settings():
        return {
            "PROCESSED_DATA_DIR": str(tmp_path / "nonexistent"),
            "MODEL_DIR": str(tmp_path / "model"),
            "HYPERPARAMETER_TUNING_DIR": str(tmp_path / "tuning"),
            "MLFLOW_TRACKING_URI": str(tmp_path / "mlruns"),
        }

    monkeypatch.setattr(utils_mod, "load_settings", fake_settings)
    monkeypatch.setattr(P, "N_TRIALS", 1)
    from train_optuna import main

    rc = main()
    assert rc == 1


def test_train_missing_best_params_falls_back_to_cli(mini_dataset, monkeypatch):
    """没有 best_params.json 时,train.py 应当回退到 argparse 的默认/CLI 超参,而不是报错退出。"""
    _force_cpu(monkeypatch)
    # 把 DEFAULT_HP 缩到与 mini_dataset 匹配的小尺寸
    monkeypatch.setattr(P, "DEFAULT_HP", {
        "embedding_dim":        32,
        "num_blocks":           2,
        "mlstm_ratio":          1.0,
        "mlp_hidden":           32,
        "dropout":              0.0,
        "conv_kernel_size":     2,
        "lr":                   1e-3,
        "static_embedding_dim": 16,
    })

    tuning_dir = mini_dataset / "hyperparameter_tuning"
    assert not (tuning_dir / "best_params.json").exists(), \
        "fixture 不应预先创建 best_params.json"

    from train import main as train_main

    # 走默认值 → 回退路径
    rc = train_main([])
    assert rc == 0
    model_path = mini_dataset / "model" / "xlstm_best.pt"
    assert model_path.exists()
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    assert bundle["hp"]["embedding_dim"] == 32  # 验证用的是 fallback DEFAULT_HP

    # 再验证 CLI 显式覆盖也生效
    model_path.unlink()
    rc = train_main(["--embedding-dim", "64", "--lr", "5e-4"])
    assert rc == 0
    bundle = torch.load(model_path, map_location="cpu", weights_only=False)
    assert bundle["hp"]["embedding_dim"] == 64
    assert bundle["hp"]["lr"] == 5e-4
