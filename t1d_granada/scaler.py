"""每个特征独立 z-score 的简单 scaler。

仅在训练集上 `fit`, 推理 / val / test 走 `transform`. NaN-safe (fit 时忽略 NaN).
持久化为 pickle, 含一个 dict[name -> (mean, std)].
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


class Scaler:
    def __init__(self) -> None:
        self.params: dict[str, tuple[float, float]] = {}

    def fit(self, name: str, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64).ravel()
        x = x[np.isfinite(x)]
        if x.size == 0:
            raise ValueError(f"cannot fit scaler `{name}` on empty / all-NaN array")
        m = float(x.mean())
        s = float(x.std())
        self.params[name] = (m, s if s > 1e-8 else 1.0)

    def transform(self, name: str, x: np.ndarray) -> np.ndarray:
        m, s = self.params[name]
        return (np.asarray(x, dtype=np.float32) - np.float32(m)) / np.float32(s)

    def inverse(self, name: str, z: np.ndarray) -> np.ndarray:
        m, s = self.params[name]
        return np.asarray(z, dtype=np.float32) * np.float32(s) + np.float32(m)

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self.params, f)

    @classmethod
    def load(cls, path: str | Path) -> "Scaler":
        obj = cls()
        with open(path, "rb") as f:
            obj.params = pickle.load(f)
        return obj
