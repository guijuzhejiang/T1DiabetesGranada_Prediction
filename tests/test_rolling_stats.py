"""U2 单测 - rolling_stats. 重点: 因果性 + min_periods + TIR 边界."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from t1d_granada.rolling_stats import (
    compute_static_stats_for_patient,
    align_stats_to_windows,
    stat_columns,
)


def _series(start: str, n: int, vals=None) -> tuple[np.ndarray, np.ndarray]:
    ts = pd.date_range(start, periods=n, freq="15min").to_numpy()
    bg = np.asarray(vals if vals is not None else np.arange(100, 100 + n), dtype=np.float64)
    return ts, bg


def test_happy_path_handcheck_2h_mean():
    # 1 day of 15-min data → 96 points
    ts, bg = _series("2026-05-09 00:00:00", 96)
    df = compute_static_stats_for_patient(
        ts, bg,
        window_hours=[2, 6],
        stats=["mean", "std", "tir_70_180", "p5", "p95"],
        sample_interval_min=15,
        tir_low=70, tir_high=180,
    )

    # at the very last timestamp, 2h window covers prior 8 points (indices 87..94, exclusive of 95)
    # Wait: closed='left' → at row index 95 (time = 23:45), window = [21:45, 23:45)
    # That spans timestamps [21:45, 23:45) which include indices 87..94 (8 points) since
    # the row at index 95 is at 23:45 (excluded). So values = bg[87..95) = bg[87..94] inclusive = 8 vals.
    last_ts = ts[-1]
    expected_vals = bg[87:95]  # indices 87 through 94 (last input not included)
    assert df.loc[last_ts, "bg_2h_mean"] == pytest.approx(expected_vals.mean(), rel=1e-6)
    assert df.loc[last_ts, "bg_2h_std"] == pytest.approx(expected_vals.std(ddof=1), rel=1e-6)


def test_causality_left_excludes_anchor():
    # Inject a huge value at the anchor; closed='left' should ignore it.
    n = 32  # 8h of data
    ts, bg = _series("2026-05-09 00:00:00", n)
    bg[-1] = 9999.0  # last row

    df = compute_static_stats_for_patient(
        ts, bg,
        window_hours=[2],
        stats=["mean", "p95"],
        sample_interval_min=15,
        tir_low=70, tir_high=180,
    )
    # at the last index, the window must NOT include 9999
    val = df.loc[ts[-1], "bg_2h_mean"]
    assert val < 200, f"closed='left' violated: mean got {val}, would be huge if 9999 leaked in"


def test_causality_future_injection_invisible():
    # Computing stats at time T, then later mutating values at T+ should not change
    # the value already returned at T (sanity check on closed semantics).
    ts, bg = _series("2026-05-09 00:00:00", 24)  # 6h
    bg_clean = bg.copy()
    df_clean = compute_static_stats_for_patient(
        ts, bg_clean, window_hours=[2], stats=["mean"],
        sample_interval_min=15, tir_low=70, tir_high=180,
    )

    # Inject anomaly AFTER row index 10 (i.e., at indices 11..)
    bg_dirty = bg.copy()
    bg_dirty[11:] = 9999.0
    df_dirty = compute_static_stats_for_patient(
        ts, bg_dirty, window_hours=[2], stats=["mean"],
        sample_interval_min=15, tir_low=70, tir_high=180,
    )

    # At index 10 (anchor t=10), window = [t-2h, t) = indices 2..9. Both versions identical.
    v_clean = df_clean.loc[ts[10], "bg_2h_mean"]
    v_dirty = df_dirty.loc[ts[10], "bg_2h_mean"]
    assert v_clean == v_dirty, "closed='left' should isolate row 10 from any future mutation"


def test_min_periods_filters_early_samples():
    # 2h needs >= 4 points; 6h needs >= 12 points
    ts, bg = _series("2026-05-09 00:00:00", 48)  # 12h
    df = compute_static_stats_for_patient(
        ts, bg, window_hours=[2, 6], stats=["mean"],
        sample_interval_min=15, tir_low=70, tir_high=180,
    )
    # row 0 has 0 prior points → both NaN
    assert pd.isna(df.iloc[0]["bg_2h_mean"])
    assert pd.isna(df.iloc[0]["bg_6h_mean"])
    # row 4 has 4 prior points (indices 0..3) → 2h OK, 6h still NaN
    assert not pd.isna(df.iloc[4]["bg_2h_mean"])
    assert pd.isna(df.iloc[4]["bg_6h_mean"])
    # row 12 has 12 prior points → both OK
    assert not pd.isna(df.iloc[12]["bg_2h_mean"])
    assert not pd.isna(df.iloc[12]["bg_6h_mean"])


def test_tir_boundaries():
    # TIR_70_180: 70 and 180 inclusive; 69 and 181 outside.
    n = 16  # 4h, plenty for 2h window
    ts = pd.date_range("2026-05-09 00:00:00", periods=n, freq="15min").to_numpy()
    bg = np.full(n, 100.0)
    bg[0:8] = [69, 70, 100, 180, 181, 100, 70, 180]  # 6/8 in range
    df = compute_static_stats_for_patient(
        ts, bg, window_hours=[2], stats=["tir_70_180"],
        sample_interval_min=15, tir_low=70, tir_high=180,
    )
    # at row 8 (anchor), 2h window covers rows 0..7 → 8 values, 6 in range → tir = 6/8 = 0.75
    val = df.iloc[8]["bg_2h_tir_70_180"]
    assert val == pytest.approx(0.75)


def test_stat_columns_layout():
    cols = stat_columns([2, 6], ["mean", "std", "tir_70_180", "p5", "p95"])
    assert cols == [
        "bg_2h_mean", "bg_2h_std", "bg_2h_tir_70_180", "bg_2h_p5", "bg_2h_p95",
        "bg_6h_mean", "bg_6h_std", "bg_6h_tir_70_180", "bg_6h_p5", "bg_6h_p95",
    ]


def test_align_to_windows():
    ts, bg = _series("2026-05-09 00:00:00", 24)
    df = compute_static_stats_for_patient(
        ts, bg, window_hours=[2], stats=["mean", "std"],
        sample_interval_min=15, tir_low=70, tir_high=180,
    )
    # ask for stats at rows 5, 10, 15
    queries = ts[[5, 10, 15]]
    arr = align_stats_to_windows(df, queries)
    assert arr.shape == (3, 2)
    # row 5: 2h window has 5 points (indices 0..4) → above min_periods=4 → not NaN
    assert not np.isnan(arr).any()
