"""多窗口血糖统计 (R4 / R4b).

严格因果: `closed='left'` 表示窗口为 [t - window, t), 不含 t 本身。
对每位患者独立计算, 对齐到每个 window 的 last_input_ts。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _tir_aggregator(low: float, high: float):
    def _tir(x):
        if x.size == 0:
            return np.nan
        return float(((x >= low) & (x <= high)).mean())

    return _tir


def stat_columns(window_hours: list[int], stats: list[str]) -> list[str]:
    cols: list[str] = []
    for w in window_hours:
        for s in stats:
            cols.append(f"bg_{w}h_{s}")
    return cols


def compute_static_stats_for_patient(
    timestamps: np.ndarray,
    bg: np.ndarray,
    *,
    window_hours: list[int],
    stats: list[str],
    sample_interval_min: int,
    tir_low: float,
    tir_high: float,
) -> pd.DataFrame:
    """For one patient, compute rolling stats anchored at each timestamp.

    Returns:
        DataFrame indexed by `timestamps` with len(window_hours) * len(stats) columns
        named `bg_{w}h_{stat}`. NaN where min_periods is not met.

    Causality: uses `closed='left'`, so the row at time t reflects bg values
    in [t - w, t) -- it never includes t itself. This guarantees that when we look
    up stats at the last-input timestamp T, the rolling window only sees data
    strictly before T (which is what we want -- the last input value itself is
    already in the seq input, and any future point is unobserved).
    """
    if len(timestamps) == 0:
        return pd.DataFrame(columns=stat_columns(window_hours, stats))

    s = pd.Series(bg, index=pd.DatetimeIndex(timestamps))
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()

    out_cols: dict[str, pd.Series] = {}
    for w in window_hours:
        points_per_window = (w * 60) // sample_interval_min
        min_periods = max(1, points_per_window // 2)
        roll = s.rolling(f"{w}h", closed="left", min_periods=min_periods)
        for stat in stats:
            col = f"bg_{w}h_{stat}"
            if stat == "mean":
                out_cols[col] = roll.mean()
            elif stat == "std":
                out_cols[col] = roll.std()
            elif stat == "tir_70_180":
                out_cols[col] = roll.apply(_tir_aggregator(tir_low, tir_high), raw=True)
            elif stat == "p5":
                out_cols[col] = roll.quantile(0.05)
            elif stat == "p95":
                out_cols[col] = roll.quantile(0.95)
            else:
                raise ValueError(f"unknown stat: {stat}")

    df = pd.DataFrame(out_cols)
    df = df.reindex(columns=stat_columns(window_hours, stats))
    return df


def align_stats_to_windows(
    stats_df: pd.DataFrame, last_input_ts: np.ndarray
) -> np.ndarray:
    """Look up rolling-stats values at each window's last_input_ts.

    Returns (N, K) float32 matrix where K = len(stats_df.columns). NaN rows mean
    min_periods was not met for that anchor and the sample should be filtered out.
    """
    if stats_df.empty or last_input_ts.size == 0:
        return np.empty((last_input_ts.size, stats_df.shape[1]), dtype=np.float32)
    ix = pd.DatetimeIndex(last_input_ts)
    aligned = stats_df.reindex(ix)
    return aligned.to_numpy(dtype=np.float32)
