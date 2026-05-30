"""predict.py: 加载训练好的 xLSTMRegressor + scaler, 对一段历史 CGM 数据做 30-min 预测。

API:
- 单条:   --history h.csv --patient_id <ID> --last_ts "YYYY-MM-DD HH:MM:SS" --sex M --birth_year 1985
- 批量:   --history h.csv --input batch.csv [--output out.csv]
            batch.csv 列: Patient_ID,last_timestamp,Sex,Birth_year
            output csv 列: Patient_ID,last_timestamp,predicted_bg_30min

history.csv 必须含 (Patient_ID, timestamp, bg) 列, 时间窗口至少包含
[last_ts - 6h, last_ts] 内的 CGM 测量, 否则 6h rolling stats min_periods 不足该样本被跳过.

注意:
- 输入特征工程 / scaler 复用与训练完全相同的代码路径
- closed='left' 一致, 任何 last_ts 之后的值都不会泄漏进 rolling stats
- 默认 device=cpu, 单条目标 < 80 ms
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from t1d_granada import params as P
from t1d_granada.feature_engineer import add_derived
from t1d_granada.model import xLSTMRegressor
from t1d_granada.rolling_stats import (
    align_stats_to_windows,
    compute_static_stats_for_patient,
    stat_columns,
)
from t1d_granada.scaler import Scaler
from t1d_granada.utils import load_settings


def _load_history(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Patient_ID": str})
    if "timestamp" not in df.columns:
        # accept date+time
        df["timestamp"] = pd.to_datetime(
            df["Measurement_date"].astype(str) + " " + df["Measurement_time"].astype(str),
            format="%Y-%m-%d %H:%M:%S", errors="coerce",
        )
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "bg" not in df.columns:
        df["bg"] = pd.to_numeric(df.get("Measurement"), errors="coerce")
    df = df.dropna(subset=["Patient_ID", "timestamp", "bg"])
    df = df.sort_values(["Patient_ID", "timestamp"], kind="mergesort")
    df = df.drop_duplicates(subset=["Patient_ID", "timestamp"], keep="first")
    return df.reset_index(drop=True)


def _build_inference_input(
    history: pd.DataFrame,
    last_ts: pd.Timestamp,
    sex: str,
    birth_year: int,
    *,
    scaler: Scaler,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build (seq, static) for one (patient, last_ts) pair.

    Returns None if requirements (4 aligned input points + sufficient rolling history) are not met.
    """
    sub = history[history["timestamp"] <= last_ts]
    if len(sub) < P.WINDOW_SIZE:
        return None
    last_4 = sub.iloc[-P.WINDOW_SIZE:].copy()
    # validate intervals 13..17 between consecutive points
    deltas = (
        last_4["timestamp"].diff().dt.total_seconds().dropna() / 60.0
    ).to_numpy()
    if (deltas.size < P.WINDOW_SIZE - 1
            or not ((deltas >= P.SAMPLE_INTERVAL_MIN - P.TIME_TOLERANCE_MIN).all()
                    and (deltas <= P.SAMPLE_INTERVAL_MIN + P.TIME_TOLERANCE_MIN).all())):
        return None
    if not last_4["bg"].notna().all():
        return None
    if last_4["timestamp"].iloc[-1] != last_ts:
        # last point must coincide with last_ts; otherwise we don't have the input slot at last_ts
        return None

    # Compute rolling stats anchored at last_ts (closed='left' → strictly < last_ts)
    stats_df = compute_static_stats_for_patient(
        history["timestamp"].to_numpy(),
        history["bg"].to_numpy(),
        window_hours=P.ROLLING_WINDOWS,
        stats=P.ROLLING_STATS,
        sample_interval_min=P.SAMPLE_INTERVAL_MIN,
        tir_low=P.TIR_LOW,
        tir_high=P.TIR_HIGH,
    )
    stats_arr = align_stats_to_windows(stats_df, np.array([last_ts.to_datetime64()]))
    if np.isnan(stats_arr).any():
        return None  # min_periods not met

    # Build static row in training layout: [Sex_M, Age, *rolling_stats]
    sex_m = 1.0 if str(sex).upper().startswith("M") else 0.0
    age = float(last_ts.year - int(birth_year))
    static_raw = np.concatenate([[sex_m, age], stats_arr[0]]).astype(np.float32)

    # Scale
    bg_z = scaler.transform("bg", last_4["bg"].to_numpy(dtype=np.float32))
    static_z = static_raw.copy()
    cols = ["Sex_M", "Age", *stat_columns(P.ROLLING_WINDOWS, P.ROLLING_STATS)]
    # col 0 (Sex_M) passthrough; col 1 (Age) and 2.. (rolling stats) z-scored
    static_z[1] = scaler.transform("Age", static_raw[1:2])[0]
    for j in range(2, len(cols)):
        static_z[j] = scaler.transform(cols[j], static_raw[j:j+1])[0]

    # Add derived seq features
    ts_seq = last_4["timestamp"].to_numpy().reshape(1, -1)  # (1, T)
    seq_2d = bg_z.reshape(1, -1)
    seq_full = add_derived(
        seq_2d, ts_seq,
        use_bg_diff=P.USE_BG_DIFF, use_time_of_day=P.USE_TIME_OF_DAY,
        use_day_of_week=P.USE_DAY_OF_WEEK,
    )  # (1, T, D_seq)

    return seq_full.astype(np.float32)[0], static_z.astype(np.float32)


def _load_model(model_path: Path, device: torch.device) -> tuple[xLSTMRegressor, Scaler, dict]:
    bundle = torch.load(model_path, map_location=device, weights_only=False)
    hp = bundle["hp"]
    meta = bundle["meta"]
    model = xLSTMRegressor(
        d_seq=meta["d_seq"], d_static=meta["d_static"],
        embedding_dim=hp["embedding_dim"], num_blocks=hp["num_blocks"],
        mlstm_ratio=hp["mlstm_ratio"], mlp_hidden=hp["mlp_hidden"],
        dropout=0.0,  # inference: drop dropout
        conv_kernel_size=hp["conv_kernel_size"],
        # 老 checkpoint 没有这个 key,fallback 到 0 = 关闭 encoder, 与原架构一致
        static_embedding_dim=hp.get("static_embedding_dim", 0),
        context_length=meta["window_size"], num_heads=4, slstm_backend="vanilla",
    )
    model.load_state_dict(bundle["state_dict"])
    model.to(device)
    model.eval()
    return model, hp, meta


def predict_one(
    model: xLSTMRegressor, scaler: Scaler, history: pd.DataFrame,
    patient_id: str, last_ts: pd.Timestamp, sex: str, birth_year: int,
    device: torch.device,
) -> float | None:
    pat_history = history[history["Patient_ID"].astype(str) == str(patient_id)]
    if pat_history.empty:
        return None
    inp = _build_inference_input(pat_history, last_ts, sex, birth_year, scaler=scaler)
    if inp is None:
        return None
    seq, static = inp
    seq_t = torch.from_numpy(seq).unsqueeze(0).to(device)
    static_t = torch.from_numpy(static).unsqueeze(0).to(device)
    with torch.no_grad():
        z = model(seq_t, static_t).item()
    return float(scaler.inverse("bg", np.array([z], dtype=np.float32))[0])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True, help="csv with Patient_ID,timestamp,bg history")
    parser.add_argument("--model", default=None, help="path to xlstm_best.pt (defaults to settings.json model dir)")
    parser.add_argument("--scaler", default=None, help="path to scaler.pkl (defaults to processed dir)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--input", help="batch csv: Patient_ID,last_timestamp,Sex,Birth_year")
    grp.add_argument("--patient_id", help="single inference: patient ID")
    parser.add_argument("--last_ts", help="single inference: last timestamp 'YYYY-MM-DD HH:MM:SS'")
    parser.add_argument("--sex", choices=["M", "F"], help="single inference: M or F")
    parser.add_argument("--birth_year", type=int, help="single inference: birth year")
    parser.add_argument("--output", default=None, help="batch mode: output csv path")
    args = parser.parse_args(argv)

    cfg = load_settings()
    from t1d_granada.utils import processed_data_dir
    model_path = Path(args.model) if args.model else Path(cfg["MODEL_DIR"]) / "xlstm_best.pt"
    scaler_path = Path(args.scaler) if args.scaler else processed_data_dir(cfg) / "scaler.pkl"

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("warning: --device cuda requested but CUDA not available; falling back to cpu")
        device_str = "cpu"
    device = torch.device(device_str)

    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"scaler not found: {scaler_path}")

    history = _load_history(Path(args.history))
    scaler = Scaler.load(scaler_path)
    model, _, _ = _load_model(model_path, device)

    if args.patient_id is not None:
        if not (args.last_ts and args.sex and args.birth_year):
            parser.error("--patient_id requires --last_ts, --sex, --birth_year")
        last_ts = pd.to_datetime(args.last_ts)
        t0 = time.perf_counter()
        pred = predict_one(model, scaler, history, args.patient_id, last_ts, args.sex,
                           args.birth_year, device)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if pred is None:
            print(f"prediction skipped: insufficient history or alignment failure for "
                  f"{args.patient_id} @ {last_ts}")
            return 1
        print(f"predicted_bg_30min = {pred:.2f} mg/dL  (elapsed {elapsed_ms:.1f} ms)")
        return 0

    # batch mode
    batch = pd.read_csv(args.input, dtype={"Patient_ID": str})
    batch["last_timestamp"] = pd.to_datetime(batch["last_timestamp"])
    rows = []
    for _, row in batch.iterrows():
        pred = predict_one(
            model, scaler, history,
            str(row["Patient_ID"]), row["last_timestamp"],
            str(row["Sex"]), int(row["Birth_year"]), device,
        )
        rows.append({
            "Patient_ID": row["Patient_ID"],
            "last_timestamp": row["last_timestamp"],
            "predicted_bg_30min": pred if pred is not None else float("nan"),
        })
    out_df = pd.DataFrame(rows)
    out_path = Path(args.output) if args.output else Path("predictions.csv")
    out_df.to_csv(out_path, index=False)
    n_ok = out_df["predicted_bg_30min"].notna().sum()
    print(f"wrote {len(out_df)} rows ({n_ok} successful) to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
