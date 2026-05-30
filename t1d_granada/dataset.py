"""PyTorch Dataset / DataLoader 工厂. 通过 np.memmap 高效加载 U2 物化的 npy。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from t1d_granada.scaler import Scaler


class GlucoseWindowDataset(Dataset):
    """每条样本返回 (seq, static, target) 三元组。

    seq: (T, D_seq) float32
    static: (D_static,) float32
    target: () float32 (标量)

    使用 mmap 避免一次性把 npy 拉进内存; DataLoader worker 会自行复制需要的 batch。
    """

    def __init__(self, seq_path: str | Path, static_path: str | Path, target_path: str | Path):
        # mmap_mode='r' is read-only, so workers can share without copy-on-write surprises.
        self.seq = np.load(str(seq_path), mmap_mode="r")
        self.static = np.load(str(static_path), mmap_mode="r")
        self.target = np.load(str(target_path), mmap_mode="r")
        if not (len(self.seq) == len(self.static) == len(self.target)):
            raise ValueError(
                f"npy length mismatch: seq={len(self.seq)}, "
                f"static={len(self.static)}, target={len(self.target)}"
            )

    def __len__(self) -> int:
        return int(self.seq.shape[0])

    def __getitem__(self, ix: int):
        # Wrap with np.array(...) so the result is detached from the memmap and torch
        # can take ownership without `RuntimeError: cannot resize storage`.
        seq = torch.from_numpy(np.array(self.seq[ix], dtype=np.float32))
        static = torch.from_numpy(np.array(self.static[ix], dtype=np.float32))
        target = torch.tensor(float(self.target[ix]), dtype=torch.float32)
        return seq, static, target


def make_loaders(
    processed_dir: str | Path,
    *,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
    processed_dir = Path(processed_dir)
    loaders = []
    for split, shuffle in [("train", True), ("val", False), ("test", False)]:
        ds = GlucoseWindowDataset(
            processed_dir / f"{split}_seq.npy",
            processed_dir / f"{split}_static.npy",
            processed_dir / f"{split}_target.npy",
        )
        loaders.append(
            DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=num_workers > 0,
                drop_last=False,
            )
        )
    return tuple(loaders)


def normalize_bg(x: np.ndarray | torch.Tensor, scaler: Scaler) -> np.ndarray | torch.Tensor:
    m, s = scaler.params["bg"]
    if isinstance(x, torch.Tensor):
        return (x - m) / s
    return (np.asarray(x, dtype=np.float32) - np.float32(m)) / np.float32(s)


def denormalize_bg(z: np.ndarray | torch.Tensor, scaler: Scaler) -> np.ndarray | torch.Tensor:
    m, s = scaler.params["bg"]
    if isinstance(z, torch.Tensor):
        return z * s + m
    return np.asarray(z, dtype=np.float32) * np.float32(s) + np.float32(m)
