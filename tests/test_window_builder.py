"""U2 单测 - window_builder."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from t1d_granada.window_builder import build_windows_for_patient, build_windows


def _ts_series(start: str, n: int, step_min: int = 15) -> np.ndarray:
    return pd.date_range(start, periods=n, freq=f"{step_min}min").to_numpy()


def test_happy_path_basic():
    n = 10
    ts = _ts_series("2026-05-09 00:00:00", n)
    bg = np.arange(100, 100 + n, dtype=np.float32)

    seq, tgt, last_ts, ts_seq = build_windows_for_patient(
        ts, bg,
        window_size=8, forecast_steps=2, sample_interval_min=15, tolerance_min=2,
    )
    # need = 4 + 2 = 6, so n - 6 + 1 = 5 candidate starts
    assert seq.shape == (5, 4)
    assert tgt.shape == (5,)
    assert last_ts.shape == (5,)
    assert ts_seq.shape == (5, 4)
    # window 0 inputs = bg[0:4] = [100,101,102,103]; target = bg[5] = 105
    np.testing.assert_array_equal(seq[0], [100, 101, 102, 103])
    assert tgt[0] == 105
    assert last_ts[0] == ts[3]


def test_drop_window_with_gap():
    n = 12
    ts_list = pd.date_range("2026-05-09 00:00:00", periods=n, freq="15min").to_list()
    # Insert a 35-min gap between index 4 and index 5
    ts_list[5:] = [t + pd.Timedelta(minutes=35) for t in ts_list[5:]]
    ts = np.array(ts_list, dtype="datetime64[ns]")
    bg = np.arange(100, 100 + n, dtype=np.float32)

    seq, tgt, last_ts, _ = build_windows_for_patient(
        ts, bg,
        window_size=8, forecast_steps=2, sample_interval_min=15, tolerance_min=2,
    )
    # Any window crossing the gap should be dropped.
    # Windows are start indices 0..6. Gap is at interval index 4 (between rows 4 and 5).
    # Each window covers needed-1=5 intervals starting at index `start`.
    # Start s is invalid if s <= 4 < s+5, i.e. s in {0,1,2,3,4}.
    # So only start s in {5,6} are valid → 2 windows.
    assert seq.shape[0] == 2
    np.testing.assert_array_equal(seq[0], bg[5:9])


def test_drop_window_with_nan():
    n = 10
    ts = _ts_series("2026-05-09 00:00:00", n)
    bg = np.arange(100, 100 + n, dtype=np.float32)
    bg[5] = np.nan

    seq, tgt, _, _ = build_windows_for_patient(
        ts, bg,
        window_size=8, forecast_steps=2, sample_interval_min=15, tolerance_min=2,
    )
    # Any window touching index 5 (within its 6-point span) is dropped.
    # Valid starts: where bg[start:start+6] all finite → only s in {0} would include 5?
    # start s covers indices s..s+5. We need 5 not in [s..s+5] → s+5 < 5 → s < 0 (none)
    # OR s > 5. So s in {6,7,8,9}? But n - needed + 1 = 5, so starts 0..4.
    # All 5 starts include row 5 → all dropped.
    assert seq.shape[0] == 0


def test_tolerance_boundary():
    # interval 13 min and 17 min should be accepted; 12 / 18 rejected
    ts = pd.to_datetime([
        "2026-05-09 00:00:00",
        "2026-05-09 00:13:00",  # +13
        "2026-05-09 00:30:00",  # +17
        "2026-05-09 00:45:00",  # +15
        "2026-05-09 01:00:00",  # +15
        "2026-05-09 01:15:00",  # +15
    ]).to_numpy()
    bg = np.arange(100, 100 + 6, dtype=np.float32)
    seq, _, _, _ = build_windows_for_patient(
        ts, bg,
        window_size=8, forecast_steps=2, sample_interval_min=15, tolerance_min=2,
    )
    assert seq.shape[0] == 1

    # Now tighten with +12 first interval → fails
    ts2 = pd.to_datetime([
        "2026-05-09 00:00:00",
        "2026-05-09 00:12:00",  # +12 → out
        "2026-05-09 00:27:00",  # +15
        "2026-05-09 00:42:00",  # +15
        "2026-05-09 00:57:00",  # +15
        "2026-05-09 01:12:00",  # +15
    ]).to_numpy()
    seq2, _, _, _ = build_windows_for_patient(
        ts2, bg,
        window_size=8, forecast_steps=2, sample_interval_min=15, tolerance_min=2,
    )
    assert seq2.shape[0] == 0


def test_too_few_rows_returns_empty():
    ts = _ts_series("2026-05-09 00:00:00", 3)
    bg = np.arange(3, dtype=np.float32)
    seq, tgt, _, _ = build_windows_for_patient(
        ts, bg, window_size=8, forecast_steps=2, sample_interval_min=15, tolerance_min=2,
    )
    assert seq.shape == (0, 4)
    assert tgt.shape == (0,)


def test_build_windows_dataframe():
    df = pd.DataFrame({
        "Patient_ID": ["P1"] * 10 + ["P2"] * 10,
        "timestamp": pd.concat([
            pd.Series(pd.date_range("2026-05-09", periods=10, freq="15min")),
            pd.Series(pd.date_range("2026-05-10", periods=10, freq="15min")),
        ]).reset_index(drop=True),
        "bg": list(range(100, 110)) + list(range(200, 210)),
    })
    out = build_windows(
        df, window_size=8, forecast_steps=2, sample_interval_min=15, tolerance_min=2,
    )
    # each patient produces 5 windows
    assert out["seq_raw"].shape == (10, 4)
    assert out["target"].shape == (10,)
    assert (out["patient_id"] == "P1").sum() == 5
    assert (out["patient_id"] == "P2").sum() == 5
