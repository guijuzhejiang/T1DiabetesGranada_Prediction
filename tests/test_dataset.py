"""U3 单测 - GlucoseWindowDataset / make_loaders / normalize_bg / denormalize_bg."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from t1d_granada.dataset import (
    GlucoseWindowDataset,
    make_loaders,
    normalize_bg,
    denormalize_bg,
)
from t1d_granada.scaler import Scaler


@pytest.fixture
def mini_processed_dir(tmp_path: Path) -> Path:
    n_train, n_val, n_test = 32, 8, 8
    T, D_seq, D_static = 4, 6, 12
    rng = np.random.RandomState(0)
    for split, n in [("train", n_train), ("val", n_val), ("test", n_test)]:
        np.save(tmp_path / f"{split}_seq.npy", rng.randn(n, T, D_seq).astype(np.float32))
        np.save(tmp_path / f"{split}_static.npy", rng.randn(n, D_static).astype(np.float32))
        np.save(tmp_path / f"{split}_target.npy", rng.randn(n).astype(np.float32))
    return tmp_path


def test_dataset_length(mini_processed_dir):
    ds = GlucoseWindowDataset(
        mini_processed_dir / "train_seq.npy",
        mini_processed_dir / "train_static.npy",
        mini_processed_dir / "train_target.npy",
    )
    assert len(ds) == 32


def test_dataset_getitem_shapes(mini_processed_dir):
    ds = GlucoseWindowDataset(
        mini_processed_dir / "train_seq.npy",
        mini_processed_dir / "train_static.npy",
        mini_processed_dir / "train_target.npy",
    )
    seq, static, tgt = ds[0]
    assert seq.shape == (4, 6) and seq.dtype == torch.float32
    assert static.shape == (12,) and static.dtype == torch.float32
    assert tgt.shape == () and tgt.dtype == torch.float32


def test_make_loaders_batch_shapes(mini_processed_dir):
    train, val, test = make_loaders(mini_processed_dir, batch_size=4, num_workers=0, pin_memory=False)
    seq, static, tgt = next(iter(train))
    assert seq.shape == (4, 4, 6)
    assert static.shape == (4, 12)
    assert tgt.shape == (4,)
    # train shuffle, val/test no shuffle: check first val batch is deterministic
    val_seq_a, _, _ = next(iter(val))
    val_seq_b, _, _ = next(iter(val))
    torch.testing.assert_close(val_seq_a, val_seq_b)


def test_normalize_inverse_roundtrip():
    sc = Scaler()
    sc.fit("bg", np.array([100, 200, 300], dtype=np.float64))
    x = np.array([110, 250, 320], dtype=np.float32)
    z = normalize_bg(x, sc)
    x2 = denormalize_bg(z, sc)
    np.testing.assert_allclose(x, x2, atol=1e-4)


def test_normalize_torch_tensor():
    sc = Scaler()
    sc.fit("bg", np.array([100, 200, 300], dtype=np.float64))
    z = torch.tensor([0.0, 1.0, -1.0])
    out = denormalize_bg(z, sc)
    assert isinstance(out, torch.Tensor)
    # Mean 200, std=100 (population) or std=100*sqrt(N/(N-1)) under sample: numpy.std default ddof=0
    # so m=200, s=81.65; out = z*81.65 + 200
    expected = z * sc.params["bg"][1] + sc.params["bg"][0]
    torch.testing.assert_close(out, expected)
