"""shap_analyze.py: 加载 model + processed 数据 → 计算 SHAP → 4 类图 + summary csv.

输出全部落到 reports/shap/. 若有 active MLflow run 也 log_artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from t1d_granada import params as P
from t1d_granada.feature_engineer import seq_feature_names
from t1d_granada.model import xLSTMRegressor
from t1d_granada.rolling_stats import stat_columns
from t1d_granada.scaler import Scaler
from t1d_granada.shap_analysis import (
    compute_shap, plot_feature_importance, plot_force_samples,
    plot_time_feature_heatmap, plot_timestep_importance, write_summary_csv,
)
from t1d_granada.utils import load_settings, make_dir, set_seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None)
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--output", default=None, help="report output dir")
    parser.add_argument("--n-bg", type=int, default=P.N_BG)
    parser.add_argument("--n-fg", type=int, default=P.N_FG)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=P.SEED)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    cfg = load_settings()
    model_path = Path(args.model) if args.model else Path(cfg["MODEL_DIR"]) / "xlstm_best.pt"
    from t1d_granada.utils import processed_data_dir
    proc_dir = Path(args.processed_dir) if args.processed_dir else processed_data_dir(cfg)
    out_dir = Path(args.output) if args.output else Path(cfg["SHAP_REPORTS_DIR"])
    make_dir(out_dir)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    bundle = torch.load(model_path, map_location=device, weights_only=False)
    hp, meta = bundle["hp"], bundle["meta"]
    model = xLSTMRegressor(
        d_seq=meta["d_seq"], d_static=meta["d_static"],
        embedding_dim=hp["embedding_dim"], num_blocks=hp["num_blocks"],
        mlstm_ratio=hp["mlstm_ratio"], mlp_hidden=hp["mlp_hidden"],
        dropout=0.0, conv_kernel_size=hp["conv_kernel_size"],
        # 老 checkpoint 没有这个 key,fallback 到 0 = 关闭 encoder, 与原架构一致
        static_embedding_dim=hp.get("static_embedding_dim", 0),
        context_length=meta.get("window_size", P.WINDOW_SIZE),
        num_heads=4, slstm_backend="vanilla",
    )
    model.load_state_dict(bundle["state_dict"])
    model.to(device).eval()

    train_seq = np.load(proc_dir / "train_seq.npy", mmap_mode="r")
    train_static = np.load(proc_dir / "train_static.npy", mmap_mode="r")
    test_seq = np.load(proc_dir / "test_seq.npy", mmap_mode="r")
    test_static = np.load(proc_dir / "test_static.npy", mmap_mode="r")

    rng = np.random.RandomState(args.seed)
    bg_idx = rng.choice(len(train_seq), size=min(args.n_bg, len(train_seq)), replace=False)
    fg_idx = rng.choice(len(test_seq), size=min(args.n_fg, len(test_seq)), replace=False)

    bg_seq_arr = np.array(train_seq[bg_idx], dtype=np.float32)
    bg_static_arr = np.array(train_static[bg_idx], dtype=np.float32)
    fg_seq_arr = np.array(test_seq[fg_idx], dtype=np.float32)
    fg_static_arr = np.array(test_static[fg_idx], dtype=np.float32)

    print(f"computing SHAP: |bg|={len(bg_idx)}, |fg|={len(fg_idx)}, device={device}")
    shap_seq, shap_static = compute_shap(
        model, bg_seq_arr, bg_static_arr, fg_seq_arr, fg_static_arr,
        device=device, batch_size=64,
    )
    print(f"shap_seq: {shap_seq.shape}, shap_static: {shap_static.shape}")

    # Predictions on fg for force plots; denormalize so axes match mg/dL
    scaler = Scaler.load(proc_dir / "scaler.pkl")
    with torch.no_grad():
        seq_t = torch.from_numpy(fg_seq_arr).to(device)
        static_t = torch.from_numpy(fg_static_arr).to(device)
        preds_z = model(seq_t, static_t).cpu().numpy()
    preds_mg = scaler.inverse("bg", preds_z)

    seq_names = seq_feature_names(P.USE_BG_DIFF, P.USE_TIME_OF_DAY, P.USE_DAY_OF_WEEK)
    static_names = ["Sex_M", "Age", *stat_columns(P.ROLLING_WINDOWS, P.ROLLING_STATS)]

    write_summary_csv(shap_seq, shap_static, seq_names, static_names, out_dir / "shap_summary.csv")
    plot_feature_importance(shap_seq, shap_static, seq_names, static_names,
                            out_dir / "feature_importance.png")
    plot_timestep_importance(shap_seq, out_dir / "timestep_importance.png")
    plot_time_feature_heatmap(shap_seq, seq_names, out_dir / "time_feature_heatmap.png")
    plot_force_samples(shap_seq, shap_static, fg_seq_arr, fg_static_arr, preds_mg,
                       seq_names, static_names, out_dir / "force_plots",
                       n_samples=P.SHAP_FORCE_PLOT_SAMPLES)

    print(f"\nReports written to {out_dir}/")
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out_dir)}  ({p.stat().st_size} B)")

    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_artifacts(str(out_dir), artifact_path="shap")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
