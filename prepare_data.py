"""prepare_data.py: 端到端数据准备入口。

读取 Glucose_measurements.csv + Patient_info.csv ⇒
  - 每患者构造合法 (4 input + +30min target) 窗口 (R1, R2)
  - 计算 2h / 6h 多窗口血糖统计 (R4, R4b)
  - 患者级 chronological 80/10/10 切分 (R2)
  - 训练集 fit scaler, 三段统一 transform
  - 序列衍生特征 (R3)
  - 物化为 npy + scaler.pkl + meta.json

Run from project root::

    cd T1DiabetesGranada_Prediction
    python prepare_data.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from t1d_granada import params as P
from t1d_granada.feature_engineer import (
    add_derived,
    compute_d_seq,
    seq_feature_names,
)
from t1d_granada.rolling_stats import (
    align_stats_to_windows,
    compute_static_stats_for_patient,
    stat_columns,
)
from t1d_granada.scaler import Scaler
from t1d_granada.utils import load_settings, make_dir, set_seed, timer
from t1d_granada.window_builder import build_windows_for_patient


def _load_glucose(path: Path) -> pd.DataFrame:
    """Read raw glucose csv, build a single timestamp column."""
    df = pd.read_csv(
        path,
        usecols=["Patient_ID", "Measurement_date", "Measurement_time", "Measurement"],
        dtype={"Patient_ID": str, "Measurement_date": str, "Measurement_time": str},
    )
    df["timestamp"] = pd.to_datetime(
        df["Measurement_date"] + " " + df["Measurement_time"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )
    df = df.dropna(subset=["timestamp"])
    df = df.rename(columns={"Measurement": "bg"})
    df["bg"] = pd.to_numeric(df["bg"], errors="coerce")
    df = df[["Patient_ID", "timestamp", "bg"]]
    df = df.sort_values(["Patient_ID", "timestamp"], kind="mergesort")
    # Some patients have duplicate (Patient_ID, timestamp) rows -- keep first to make
    # rolling-stats reindex unambiguous.
    df = df.drop_duplicates(subset=["Patient_ID", "timestamp"], keep="first")
    return df.reset_index(drop=True)


def _load_patient_info(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["Patient_ID", "Sex", "Birth_year"], dtype={"Patient_ID": str})
    df["Birth_year"] = pd.to_numeric(df["Birth_year"], errors="coerce")
    df = df.dropna(subset=["Sex", "Birth_year"])
    df = df[(df["Birth_year"] >= P.BIRTH_YEAR_MIN) & (df["Birth_year"] <= P.BIRTH_YEAR_MAX)]
    df = df[df["Sex"].isin(["M", "F"])]
    df = df.drop_duplicates(subset=["Patient_ID"], keep="first")
    return df.reset_index(drop=True)


def _per_patient_chrono_split(
    last_input_ts: np.ndarray, train_frac: float, val_frac: float
) -> np.ndarray:
    """Within a single patient, mark each sample's split label by time order.

    Returns int8 array: 0=train, 1=val, 2=test.
    """
    n = last_input_ts.size
    if n == 0:
        return np.empty((0,), dtype=np.int8)
    order = np.argsort(last_input_ts, kind="mergesort")
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    n_train = max(0, min(n_train, n))
    n_val = max(0, min(n_val, n - n_train))
    labels = np.empty(n, dtype=np.int8)
    labels[order[:n_train]] = 0
    labels[order[n_train : n_train + n_val]] = 1
    labels[order[n_train + n_val :]] = 2
    return labels


def build_dataset(
    glucose_df: pd.DataFrame,
    patient_info: pd.DataFrame,
    *,
    progress: bool = True,
) -> dict:
    """Build per-sample arrays across all patients (no scaling, no derived features yet)."""
    static_cols = stat_columns(P.ROLLING_WINDOWS, P.ROLLING_STATS)
    n_static_stats = len(static_cols)

    pinfo = patient_info.set_index("Patient_ID")[["Sex", "Birth_year"]].to_dict(orient="index")

    all_seq: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    all_last_ts: list[np.ndarray] = []
    all_ts_seq: list[np.ndarray] = []
    all_static: list[np.ndarray] = []
    all_pid: list[np.ndarray] = []

    n_total_windows = 0
    n_dropped_no_pinfo = 0
    n_dropped_stats = 0

    iterator = glucose_df.groupby("Patient_ID", sort=False)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, total=glucose_df["Patient_ID"].nunique(), desc="patients")
        except ImportError:
            pass

    for pid, sub in iterator:
        if pid not in pinfo:
            n_dropped_no_pinfo += 1
            continue

        ts_arr = sub["timestamp"].to_numpy()
        bg_arr = sub["bg"].to_numpy()

        seq_raw, target, last_ts, ts_seq = build_windows_for_patient(
            ts_arr,
            bg_arr,
            window_size=P.WINDOW_SIZE,
            forecast_steps=P.FORECAST_STEPS,
            sample_interval_min=P.SAMPLE_INTERVAL_MIN,
            tolerance_min=P.TIME_TOLERANCE_MIN,
        )
        if seq_raw.shape[0] == 0:
            continue
        n_total_windows += seq_raw.shape[0]

        stats_df = compute_static_stats_for_patient(
            ts_arr,
            bg_arr,
            window_hours=P.ROLLING_WINDOWS,
            stats=P.ROLLING_STATS,
            sample_interval_min=P.SAMPLE_INTERVAL_MIN,
            tir_low=P.TIR_LOW,
            tir_high=P.TIR_HIGH,
        )
        static_stats = align_stats_to_windows(stats_df, last_ts)
        keep = ~np.isnan(static_stats).any(axis=1)
        n_dropped_stats += int((~keep).sum())
        if not keep.any():
            continue

        seq_raw = seq_raw[keep]
        target = target[keep]
        last_ts = last_ts[keep]
        ts_seq = ts_seq[keep]
        static_stats = static_stats[keep]

        sex = pinfo[pid]["Sex"]
        birth_year = int(pinfo[pid]["Birth_year"])
        sex_m = np.full(seq_raw.shape[0], 1.0 if sex == "M" else 0.0, dtype=np.float32)
        years = pd.DatetimeIndex(last_ts).year.to_numpy()
        age = (years - birth_year).astype(np.float32)

        # static layout: [Sex_M, Age, *rolling_stats]
        static = np.concatenate(
            [sex_m[:, None], age[:, None], static_stats.astype(np.float32)], axis=1
        )

        all_seq.append(seq_raw)
        all_target.append(target)
        all_last_ts.append(last_ts)
        all_ts_seq.append(ts_seq)
        all_static.append(static)
        all_pid.append(np.full(seq_raw.shape[0], pid, dtype=object))

    if not all_seq:
        raise RuntimeError("no valid windows produced; check input data")

    return {
        "seq_raw": np.concatenate(all_seq, axis=0),
        "target": np.concatenate(all_target, axis=0),
        "last_input_ts": np.concatenate(all_last_ts, axis=0),
        "ts_seq": np.concatenate(all_ts_seq, axis=0),
        "static_raw": np.concatenate(all_static, axis=0),
        "patient_id": np.concatenate(all_pid, axis=0),
        "static_cols": ["Sex_M", "Age", *static_cols],
        "n_total_windows": n_total_windows,
        "n_dropped_no_pinfo": n_dropped_no_pinfo,
        "n_dropped_stats": n_dropped_stats,
    }


def split_dataset(dataset: dict) -> np.ndarray:
    """Per-patient chronological split into train/val/test."""
    pid = dataset["patient_id"]
    last_ts = dataset["last_input_ts"]
    labels = np.empty(pid.shape[0], dtype=np.int8)
    df_ix = pd.DataFrame({"pid": pid, "ts": last_ts, "ix": np.arange(pid.shape[0])})
    for _, sub in df_ix.groupby("pid", sort=False):
        ts = sub["ts"].to_numpy()
        ixs = sub["ix"].to_numpy()
        sub_labels = _per_patient_chrono_split(ts, P.SPLIT_TRAIN, P.SPLIT_VAL)
        labels[ixs] = sub_labels
    return labels


def fit_and_apply_scaler(
    dataset: dict, split_labels: np.ndarray
) -> tuple[Scaler, dict, dict, dict]:
    """Fit scaler on train rows, then transform all three splits.

    Returns (scaler, seqs_z_dict, statics_z_dict, targets_z_dict). Sex_M passes through.
    """
    train_mask = split_labels == 0
    sc = Scaler()
    # bg
    sc.fit("bg", dataset["seq_raw"][train_mask].ravel())
    # age
    sc.fit("Age", dataset["static_raw"][train_mask, 1])
    # rolling stats: cols 2..end of static_raw, names follow static_cols[2:]
    static_cols = dataset["static_cols"]
    for j in range(2, len(static_cols)):
        sc.fit(static_cols[j], dataset["static_raw"][train_mask, j])

    seq_z_all = sc.transform("bg", dataset["seq_raw"])

    static_z_all = dataset["static_raw"].copy().astype(np.float32)
    static_z_all[:, 1] = sc.transform("Age", static_z_all[:, 1])
    for j in range(2, len(static_cols)):
        static_z_all[:, j] = sc.transform(static_cols[j], static_z_all[:, j])
    # col 0 (Sex_M) passthrough.

    target_z_all = sc.transform("bg", dataset["target"])

    seqs = {}
    statics = {}
    targets = {}
    for split_id, split_name in [(0, "train"), (1, "val"), (2, "test")]:
        m = split_labels == split_id
        seqs[split_name] = seq_z_all[m]
        statics[split_name] = static_z_all[m]
        targets[split_name] = target_z_all[m]
    return sc, seqs, statics, targets


def materialize(out_dir: Path, dataset: dict, scaler: Scaler, seqs, statics, targets, split_labels):
    """Add derived seq features and dump everything."""
    make_dir(out_dir)

    feat_names = seq_feature_names(P.USE_BG_DIFF, P.USE_TIME_OF_DAY, P.USE_DAY_OF_WEEK)
    d_seq = compute_d_seq(P.USE_BG_DIFF, P.USE_TIME_OF_DAY, P.USE_DAY_OF_WEEK)
    print(f"D_seq = {d_seq}, features = {feat_names}")

    counts = {}
    for split_name in ["train", "val", "test"]:
        m = split_labels == {"train": 0, "val": 1, "test": 2}[split_name]
        ts_seq = dataset["ts_seq"][m]
        seq_z = seqs[split_name]
        seq_full = add_derived(
            seq_z,
            ts_seq,
            use_bg_diff=P.USE_BG_DIFF,
            use_time_of_day=P.USE_TIME_OF_DAY,
            use_day_of_week=P.USE_DAY_OF_WEEK,
        )
        np.save(out_dir / f"{split_name}_seq.npy", seq_full.astype(np.float32))
        np.save(out_dir / f"{split_name}_static.npy", statics[split_name].astype(np.float32))
        np.save(out_dir / f"{split_name}_target.npy", targets[split_name].astype(np.float32))
        counts[split_name] = int(seq_full.shape[0])
        print(f"  {split_name}: seq={seq_full.shape}, static={statics[split_name].shape}, target={targets[split_name].shape}")

    scaler.save(out_dir / "scaler.pkl")

    meta = {
        "window_size": P.WINDOW_SIZE,
        "forecast_steps": P.FORECAST_STEPS,
        "sample_interval_min": P.SAMPLE_INTERVAL_MIN,
        "tolerance_min": P.TIME_TOLERANCE_MIN,
        "rolling_windows_h": P.ROLLING_WINDOWS,
        "rolling_stats": P.ROLLING_STATS,
        "tir_low": P.TIR_LOW,
        "tir_high": P.TIR_HIGH,
        "split_train": P.SPLIT_TRAIN,
        "split_val": P.SPLIT_VAL,
        "use_bg_diff": P.USE_BG_DIFF,
        "use_time_of_day": P.USE_TIME_OF_DAY,
        "use_day_of_week": P.USE_DAY_OF_WEEK,
        "d_seq": d_seq,
        "seq_feature_names": feat_names,
        "static_cols": dataset["static_cols"],
        "d_static": len(dataset["static_cols"]),
        "counts": counts,
        "n_total_windows_before_stats_filter": dataset["n_total_windows"],
        "n_dropped_no_pinfo": dataset["n_dropped_no_pinfo"],
        "n_dropped_stats_filter": dataset["n_dropped_stats"],
        "scaler_keys": list(scaler.params.keys()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit-rows", type=int, default=None,
                        help="optional: read only N rows (smoke testing)")
    args = parser.parse_args(argv)

    set_seed(P.SEED)
    cfg = load_settings()
    glucose_path = Path(cfg["GLUCOSE_FILE"])
    patient_info_path = Path(cfg["PATIENT_INFO_FILE"])
    from t1d_granada.utils import processed_data_dir
    out_dir = processed_data_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output dir (WINDOW_SIZE={P.WINDOW_SIZE}): {out_dir}")

    with timer("read csvs"):
        glucose_df = _load_glucose(glucose_path)
        if args.limit_rows is not None:
            glucose_df = glucose_df.head(args.limit_rows)
        pinfo = _load_patient_info(patient_info_path)
        print(f"  glucose rows: {len(glucose_df):,}, patients: {glucose_df['Patient_ID'].nunique()}")
        print(f"  patient_info rows after filter: {len(pinfo):,}")

    with timer("build windows + rolling stats"):
        dataset = build_dataset(glucose_df, pinfo, progress=True)
        print(
            f"  total windows: {dataset['n_total_windows']:,}, "
            f"dropped (no pinfo): {dataset['n_dropped_no_pinfo']:,}, "
            f"dropped (stats min_periods): {dataset['n_dropped_stats']:,}"
        )
        print(f"  kept after stats filter: {dataset['seq_raw'].shape[0]:,}")

    with timer("split + scale"):
        split_labels = split_dataset(dataset)
        scaler, seqs, statics, targets = fit_and_apply_scaler(dataset, split_labels)
        for name in ("train", "val", "test"):
            print(f"  {name}: {(split_labels == {'train':0,'val':1,'test':2}[name]).sum():,}")

    with timer("derive seq features + dump npy"):
        meta = materialize(out_dir, dataset, scaler, seqs, statics, targets, split_labels)

    print("\n=== meta ===")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
