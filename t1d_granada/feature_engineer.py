"""序列衍生特征 (R3): bg_diff / time_of_day_sin/cos / day_of_week_sin/cos.

输入: 已经 z-scored 的 seq (N, T) + 时间戳 (N, T)。
输出: (N, T, D_seq), 其中 D_seq 由 flags 决定。
"""
from __future__ import annotations

import numpy as np


def compute_d_seq(use_bg_diff: bool, use_time_of_day: bool, use_day_of_week: bool) -> int:
    d = 1  # bg_z always present
    if use_bg_diff:
        d += 1
    if use_time_of_day:
        d += 2
    if use_day_of_week:
        d += 2
    return d


def seq_feature_names(use_bg_diff: bool, use_time_of_day: bool, use_day_of_week: bool) -> list[str]:
    names = ["bg_z"]
    if use_bg_diff:
        names.append("bg_diff")
    if use_time_of_day:
        names.extend(["tod_sin", "tod_cos"])
    if use_day_of_week:
        names.extend(["dow_sin", "dow_cos"])
    return names


def add_derived(
    seq_z: np.ndarray,
    ts_seq: np.ndarray,
    *,
    use_bg_diff: bool,
    use_time_of_day: bool,
    use_day_of_week: bool,
) -> np.ndarray:
    """Extend (N, T) bg_z to (N, T, D_seq) with optional derived channels.

    bg_diff: first-order diff along T, prepended with 0 so length stays T.
    time_of_day: minute fraction of day in [0, 1) → sin/cos.
    day_of_week: weekday in {0..6} (Mon=0) → sin/cos.

    All channels are independent per timestep (no shared scaler).
    """
    if seq_z.ndim != 2 or ts_seq.shape != seq_z.shape:
        raise ValueError(
            f"seq_z and ts_seq must be 2-D with matching shape; "
            f"got seq_z={seq_z.shape}, ts_seq={ts_seq.shape}"
        )

    parts: list[np.ndarray] = [seq_z[:, :, None].astype(np.float32)]

    if use_bg_diff:
        first = seq_z[:, :1]
        diff = np.diff(seq_z, axis=1, prepend=first)  # first step = 0
        parts.append(diff[:, :, None].astype(np.float32))

    if use_time_of_day or use_day_of_week:
        ts_int = ts_seq.astype("datetime64[s]").astype(np.int64)
        if use_time_of_day:
            sec_in_day = ts_int % 86400
            frac = sec_in_day.astype(np.float64) / 86400.0
            parts.append(np.sin(2 * np.pi * frac).astype(np.float32)[:, :, None])
            parts.append(np.cos(2 * np.pi * frac).astype(np.float32)[:, :, None])
        if use_day_of_week:
            days_since_epoch = ts_int // 86400
            # 1970-01-01 = Thursday (Mon=0 → Thu=3)
            weekday = (days_since_epoch + 3) % 7
            frac = weekday.astype(np.float64) / 7.0
            parts.append(np.sin(2 * np.pi * frac).astype(np.float32)[:, :, None])
            parts.append(np.cos(2 * np.pi * frac).astype(np.float32)[:, :, None])

    return np.concatenate(parts, axis=2)
