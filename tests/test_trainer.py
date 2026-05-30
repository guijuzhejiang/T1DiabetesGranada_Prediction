"""U5 单测 - trainer.train_one_config."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from t1d_granada.model import xLSTMRegressor
from t1d_granada.scaler import Scaler
from t1d_granada.trainer import train_one_config


def _mini_loaders(
    n_train: int = 64, n_val: int = 32, T: int = 4, D_seq: int = 6, D_static: int = 12, batch: int = 16
) -> tuple[DataLoader, DataLoader, Scaler]:
    rng = np.random.RandomState(0)
    raw_bg = rng.randn(1000) * 50 + 150  # mg/dL-ish
    sc = Scaler()
    sc.fit("bg", raw_bg)

    def _ds(n):
        seq = torch.randn(n, T, D_seq)
        static = torch.randn(n, D_static)
        # target = mean of seq[:, :, 0] + small noise → learnable
        tgt = seq[:, :, 0].mean(dim=1) + 0.1 * torch.randn(n)
        return TensorDataset(seq, static, tgt)

    train = DataLoader(_ds(n_train), batch_size=batch, shuffle=True)
    val = DataLoader(_ds(n_val), batch_size=batch)
    return train, val, sc


def _mini_model() -> xLSTMRegressor:
    return xLSTMRegressor(
        d_seq=6, d_static=12, embedding_dim=32, num_blocks=2, mlstm_ratio=1.0,
        mlp_hidden=32, dropout=0.0, conv_kernel_size=2, context_length=4, num_heads=4,
    )


def test_train_one_config_runs():
    train, val, sc = _mini_loaders()
    model = _mini_model()
    out = train_one_config(
        model, train, val, sc,
        max_epochs=3, patience=10, lr=1e-3, weight_decay=0,
        device=torch.device("cpu"), log_to_mlflow=False,
    )
    assert out["best_val_rmse"] > 0
    assert "best_state_dict" in out
    assert out["last_epoch"] == 2
    assert len(out["history"]) == 3
    keys = set(out["history"][0].keys())
    assert {"epoch", "train_loss", "val_rmse", "val_mae", "val_r2", "lr", "epoch_time"} <= keys


def test_early_stopping_triggers():
    train, val, sc = _mini_loaders()
    model = _mini_model()
    # lr=0 → params never update, val_rmse identical each epoch → strict-< never improves
    out = train_one_config(
        model, train, val, sc,
        max_epochs=20, patience=2, lr=0.0, weight_decay=0,
        device=torch.device("cpu"), log_to_mlflow=False,
    )
    # epoch 0 records best; epochs 1 & 2 don't improve → stop after epoch 2
    assert out["last_epoch"] <= 3, f"early stopping failed, ran {out['last_epoch']} epochs"


def test_empty_train_loader_errors():
    _, val, sc = _mini_loaders()
    empty_train = DataLoader(TensorDataset(torch.empty(0, 4, 6), torch.empty(0, 12), torch.empty(0)), batch_size=4)
    model = _mini_model()
    with pytest.raises(ValueError):
        train_one_config(
            model, empty_train, val, sc,
            max_epochs=1, patience=1, lr=1e-3, weight_decay=0,
            device=torch.device("cpu"), log_to_mlflow=False,
        )


def test_optuna_pruning_raises():
    optuna = pytest.importorskip("optuna")
    train, val, sc = _mini_loaders()
    model = _mini_model()

    class StubTrial:
        def __init__(self):
            self.calls = 0

        def report(self, value, step):
            self.calls += 1

        def should_prune(self):
            return self.calls >= 2  # prune on 2nd epoch

    trial = StubTrial()
    with pytest.raises(optuna.TrialPruned):
        train_one_config(
            model, train, val, sc,
            max_epochs=10, patience=10, lr=1e-3, weight_decay=0,
            device=torch.device("cpu"), log_to_mlflow=False,
            optuna_trial=trial,
        )


def test_state_dict_returned_even_when_no_improvement():
    train, val, sc = _mini_loaders()
    model = _mini_model()
    out = train_one_config(
        model, train, val, sc,
        max_epochs=1, patience=1, lr=1e-3, weight_decay=0,
        device=torch.device("cpu"), log_to_mlflow=False,
    )
    # First epoch always sets best_state
    assert out["best_state_dict"] is not None
    state = out["best_state_dict"]
    assert any("input_proj" in k for k in state.keys())
