"""窗口构造: 严格 15-min 对齐, stride=1, 跨 NaN/大缺口的窗口直接丢弃。

输入: 每个患者按时间排序的 (timestamp, bg) 序列。
输出: 合法窗口的 seq_raw, target, last_input_ts, ts_seq。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def build_windows_for_patient(
    ts: np.ndarray,
    bg: np.ndarray,
    *,
    window_size: int,
    forecast_steps: int,
    sample_interval_min: int,
    tolerance_min: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """For one patient with sorted (ts, bg) arrays, return all valid windows.

    Each window covers `window_size + forecast_steps` consecutive points; the first
    `window_size` are inputs, the last index (forecast_steps - 1 after the last input)
    is the prediction target. **Intermediate forecast steps are skipped** -- e.g. for
    forecast_steps=2 the +15min step is not used as either input or target.

    All consecutive intervals within a window must lie in
    [sample_interval_min - tolerance_min, sample_interval_min + tolerance_min] minutes,
    and all bg values must be finite.

    Returns:
        seq_raw: (N, window_size) float32 bg values for the input slots.
        target:  (N,) float32 bg value at the forecast horizon.
        last_input_ts: (N,) datetime64[ns] timestamp of the last input (window_size-th point).
        ts_seq:  (N, window_size) datetime64[ns] timestamps for each input slot.
    """
    if window_size < 1 or forecast_steps < 1:
        raise ValueError("window_size and forecast_steps must be >= 1")

    needed = window_size + forecast_steps
    n = len(bg)
    if n < needed:
        empty = np.empty((0, window_size), dtype=np.float32)
        empty_targets = np.empty((0,), dtype=np.float32)
        empty_ts1 = np.empty((0,), dtype="datetime64[ns]")
        empty_tsW = np.empty((0, window_size), dtype="datetime64[ns]")
        return empty, empty_targets, empty_ts1, empty_tsW

    bg = np.asarray(bg, dtype=np.float64)
    ts = np.asarray(ts, dtype="datetime64[ns]")

    # Per-step intervals in minutes; intervals[i] = ts[i+1] - ts[i]
    deltas = np.diff(ts).astype("timedelta64[s]").astype(np.int64) // 60  # minutes
    valid_step = (deltas >= sample_interval_min - tolerance_min) & (
        deltas <= sample_interval_min + tolerance_min
    )

    # sliding windows
    bg_win = sliding_window_view(bg, needed)               # (n - needed + 1, needed)
    step_win = sliding_window_view(valid_step, needed - 1) # (n - needed + 1, needed - 1)
    finite_ok = np.isfinite(bg_win).all(axis=1)
    step_ok = step_win.all(axis=1)
    valid_starts = np.where(finite_ok & step_ok)[0]

    if valid_starts.size == 0:
        empty = np.empty((0, window_size), dtype=np.float32)
        empty_targets = np.empty((0,), dtype=np.float32)
        empty_ts1 = np.empty((0,), dtype="datetime64[ns]")
        empty_tsW = np.empty((0, window_size), dtype="datetime64[ns]")
        return empty, empty_targets, empty_ts1, empty_tsW

    # input columns 0..window_size-1, target column needed-1
    seq_raw = bg_win[valid_starts, :window_size].astype(np.float32)
    target = bg_win[valid_starts, needed - 1].astype(np.float32)

    ts_win = sliding_window_view(ts, needed)  # (n - needed + 1, needed)
    ts_seq = ts_win[valid_starts, :window_size]
    last_input_ts = ts_seq[:, -1]

    return seq_raw, target, last_input_ts, ts_seq


def build_windows(
    glucose_df: pd.DataFrame,
    *,
    window_size: int,
    forecast_steps: int,
    sample_interval_min: int,
    tolerance_min: int,
    progress: bool = False,
) -> dict:
    """Build windows across all patients in `glucose_df`.

    Args:
        glucose_df: must have columns ['Patient_ID', 'timestamp', 'bg'] sorted by
            (Patient_ID, timestamp).
    Returns dict with concatenated arrays and a parallel patient_id array.
    """
    required = {"Patient_ID", "timestamp", "bg"}
    missing = required - set(glucose_df.columns)
    if missing:
        raise ValueError(f"glucose_df missing columns: {missing}")

    patient_ids: list[np.ndarray] = []
    seqs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    last_ts_list: list[np.ndarray] = []
    ts_seqs: list[np.ndarray] = []

    iterator = glucose_df.groupby("Patient_ID", sort=False)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="build_windows", total=glucose_df["Patient_ID"].nunique())
        except ImportError:
            pass

    for pid, sub in iterator:
        if not sub["timestamp"].is_monotonic_increasing:
            sub = sub.sort_values("timestamp")
        seq_raw, tgt, last_ts, ts_seq = build_windows_for_patient(
            sub["timestamp"].to_numpy(),
            sub["bg"].to_numpy(),
            window_size=window_size,
            forecast_steps=forecast_steps,
            sample_interval_min=sample_interval_min,
            tolerance_min=tolerance_min,
        )
        if seq_raw.shape[0] == 0:
            continue
        patient_ids.append(np.full(seq_raw.shape[0], pid, dtype=object))
        seqs.append(seq_raw)
        targets.append(tgt)
        last_ts_list.append(last_ts)
        ts_seqs.append(ts_seq)

    if not seqs:
        return {
            "patient_id": np.empty((0,), dtype=object),
            "seq_raw": np.empty((0, window_size), dtype=np.float32),
            "target": np.empty((0,), dtype=np.float32),
            "last_input_ts": np.empty((0,), dtype="datetime64[ns]"),
            "ts_seq": np.empty((0, window_size), dtype="datetime64[ns]"),
        }

    return {
        "patient_id": np.concatenate(patient_ids),
        "seq_raw": np.concatenate(seqs, axis=0),
        "target": np.concatenate(targets, axis=0),
        "last_input_ts": np.concatenate(last_ts_list, axis=0),
        "ts_seq": np.concatenate(ts_seqs, axis=0),
    }
