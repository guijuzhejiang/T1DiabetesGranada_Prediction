"""U2 单测 - feature_engineer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from t1d_granada.feature_engineer import (
    add_derived,
    compute_d_seq,
    seq_feature_names,
)


def test_d_seq_combinations():
    assert compute_d_seq(False, False, False) == 1
    assert compute_d_seq(True, True, True) == 6
    assert compute_d_seq(True, False, False) == 2
    assert compute_d_seq(False, True, False) == 3
    assert compute_d_seq(False, False, True) == 3


def test_feature_names_order():
    names = seq_feature_names(True, True, True)
    assert names == ["bg_z", "bg_diff", "tod_sin", "tod_cos", "dow_sin", "dow_cos"]


def test_flag_off_only_bg():
    seq_z = np.zeros((3, 4), dtype=np.float32)
    ts = np.broadcast_to(
        pd.date_range("2026-05-09", periods=4, freq="15min").to_numpy(),
        (3, 4),
    )
    out = add_derived(seq_z, ts, use_bg_diff=False, use_time_of_day=False, use_day_of_week=False)
    assert out.shape == (3, 4, 1)


def test_flag_all_on():
    seq_z = np.random.RandomState(0).randn(5, 4).astype(np.float32)
    ts = np.broadcast_to(
        pd.date_range("2026-05-09 00:00:00", periods=4, freq="15min").to_numpy(),
        (5, 4),
    )
    out = add_derived(seq_z, ts, use_bg_diff=True, use_time_of_day=True, use_day_of_week=True)
    assert out.shape == (5, 4, 6)
    # bg_z passthrough
    np.testing.assert_array_equal(out[:, :, 0], seq_z)


def test_bg_diff_first_step_zero():
    seq_z = np.array([[1.0, 2.0, 4.0, 7.0]], dtype=np.float32)
    ts = pd.date_range("2026-05-09", periods=4, freq="15min").to_numpy()[None, :]
    out = add_derived(seq_z, ts, use_bg_diff=True, use_time_of_day=False, use_day_of_week=False)
    # channel 1 = bg_diff
    np.testing.assert_allclose(out[0, :, 1], [0, 1, 2, 3])


def test_time_of_day_periodicity_at_midnight():
    """sin(2π · 0/86400) = 0; cos = 1."""
    seq_z = np.zeros((1, 1), dtype=np.float32)
    ts_midnight = np.array([["2026-05-09 00:00:00"]], dtype="datetime64[s]")
    out = add_derived(seq_z, ts_midnight, use_bg_diff=False, use_time_of_day=True, use_day_of_week=False)
    assert out[0, 0, 1] == pytest.approx(0.0, abs=1e-6)
    assert out[0, 0, 2] == pytest.approx(1.0, abs=1e-6)
