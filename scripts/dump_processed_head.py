"""把 data/processed/ 下的 9 个 npy 各取前 20 行，落到 data/processed/head20/ 的 CSV。

列名带物理含义:
- seq: 8 个时间点 × 6 个特征,共 48 列。命名格式 `t-45_bg_z` 表示距 last_input_ts -45 分钟的 bg z-score
- static: 12 列,直接用 meta.json 里的 static_cols
- target: 同时给 z-score 和 inverse 回 mg/dL 两列
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from t1d_granada.scaler import Scaler
from t1d_granada.utils import processed_data_dir

N_HEAD = 20
PROC = processed_data_dir()      # data/processed/WINDOW_SIZE_<P.WINDOW_SIZE>/
OUT = PROC / "head20"
OUT.mkdir(parents=True, exist_ok=True)

meta = json.loads((PROC / "meta.json").read_text())
seq_names = meta["seq_feature_names"]              # 6 features
static_cols = meta["static_cols"]                  # 12 features
window_size = meta["window_size"]                  # 4
forecast_steps = meta["forecast_steps"]            # 2
interval = meta["sample_interval_min"]             # 15

# t=0 是最早,t=window_size-1 是 last_input_ts。相对 last_input_ts 的偏移分钟数:
# t=0 → -(window_size-1)*interval, t=window_size-1 → 0
offsets = [-(window_size - 1 - t) * interval for t in range(window_size)]  # [-45, -30, -15, 0]
seq_col_names = [f"t{off:+d}_{name}" for off in offsets for name in seq_names]

scaler = Scaler.load(PROC / "scaler.pkl")

for split in ["train", "val", "test"]:
    seq = np.load(PROC / f"{split}_seq.npy", mmap_mode="r")[:N_HEAD]   # (n, T, D_seq)
    static = np.load(PROC / f"{split}_static.npy", mmap_mode="r")[:N_HEAD]
    target = np.load(PROC / f"{split}_target.npy", mmap_mode="r")[:N_HEAD]

    # seq → flatten 成 (N, T*D_seq),按 [t0_f0, t0_f1, ..., t3_f5] 顺序
    seq_flat = np.array(seq).reshape(seq.shape[0], -1)
    df_seq = pd.DataFrame(seq_flat, columns=seq_col_names)
    df_seq.insert(0, "row_idx", np.arange(seq.shape[0]))
    df_seq.to_csv(OUT / f"{split}_seq_head{N_HEAD}.csv", index=False, float_format="%.6f")

    df_static = pd.DataFrame(np.array(static), columns=static_cols)
    df_static.insert(0, "row_idx", np.arange(static.shape[0]))
    df_static.to_csv(OUT / f"{split}_static_head{N_HEAD}.csv", index=False, float_format="%.6f")

    target_arr = np.array(target)
    target_mg = scaler.inverse("bg", target_arr)
    df_target = pd.DataFrame({
        "row_idx": np.arange(target_arr.shape[0]),
        "target_z": target_arr,
        f"target_mg_dL_at_+{forecast_steps * interval}min": target_mg,
    })
    df_target.to_csv(OUT / f"{split}_target_head{N_HEAD}.csv", index=False, float_format="%.4f")

    print(f"{split}: seq {seq.shape} static {static.shape} target {target.shape}")

print(f"\n✓ wrote 9 CSVs to {OUT}/")
for p in sorted(OUT.glob("*.csv")):
    print(f"  {p.relative_to(PROC)}  ({p.stat().st_size} B)")
