"""train.py: 用 train_optuna.py 搜到的最优超参做最终 fit。

流程:
1. 读 hyperparameter_tuning/best_params.json + best_trial_meta.json (含 last_epoch)
   若 best_params.json 不存在 → 回退到 argparse 传入的超参(默认值来自 P.DEFAULT_HP)
2. 用 train 训练、val 监控,触发 early stop (patience 由 P.PATIENCE 控制)
3. max_epochs:
   - P.FINAL_FIT_EPOCHS_OVERRIDE 不为 None → 直接用它
   - 否则 max_epochs = (last_epoch + 1) * FINAL_FIT_EPOCH_MULT  # 上限,实际由早停决定
4. 取 best_state_dict (best-on-val) → 在 test 集评估 → 保存 model/xlstm_best.pt
5. parent MLflow run "{FINAL_FIT_RUN_NAME}"

注: 不再合并 train+val。合并后没有 held-out 监控集,要么关掉早停,
要么用 test 监控会泄漏。保留独立 val 做早停最稳。

绝大多数参数集中在 t1d_granada/params.py:
- PATIENCE, WEIGHT_DECAY, WARMUP_RATIO, GRAD_CLIP, NUM_WORKERS
- FINAL_FIT_EPOCH_MULT, FINAL_FIT_EPOCHS_OVERRIDE, FINAL_FIT_RUN_NAME
- DEFAULT_HP (best_params.json 不存在时的 fallback)

CLI 仅在没有 best_params.json 时生效, 例如:
    python train.py --embedding-dim 96 --num-blocks 3 --lr 3e-4

GPU: CUDA_VISIBLE_DEVICES=1 python train.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch

from t1d_granada import params as P
from t1d_granada import utils as utils_mod
from t1d_granada.dataset import make_loaders
from t1d_granada.model import xLSTMRegressor
from t1d_granada.scaler import Scaler
from t1d_granada.trainer import _validate, train_one_config
from t1d_granada.utils import make_dir, set_seed, timer


def _read_meta(processed_dir: Path) -> dict:
    return json.loads((processed_dir / "meta.json").read_text())


def _resolve_max_epochs(last_epoch: int) -> int:
    """从 P.FINAL_FIT_EPOCHS_OVERRIDE 或 last_epoch * MULT 推算 max_epochs。"""
    if P.FINAL_FIT_EPOCHS_OVERRIDE is not None:
        return int(P.FINAL_FIT_EPOCHS_OVERRIDE)
    return max(5, int(round((last_epoch + 1) * P.FINAL_FIT_EPOCH_MULT)))


def _build_arg_parser() -> argparse.ArgumentParser:
    """CLI 仅在没有 best_params.json 时生效,用作 fallback 超参。默认值来自 P.DEFAULT_HP。"""
    p = argparse.ArgumentParser(
        description="Fallback hyperparameters used only when best_params.json is missing.",
    )
    d = P.DEFAULT_HP
    p.add_argument("--embedding-dim",        type=int,   default=d["embedding_dim"])
    p.add_argument("--num-blocks",           type=int,   default=d["num_blocks"])
    p.add_argument("--mlstm-ratio",          type=float, default=d["mlstm_ratio"])
    p.add_argument("--mlp-hidden",           type=int,   default=d["mlp_hidden"])
    p.add_argument("--dropout",              type=float, default=d["dropout"])
    p.add_argument("--conv-kernel-size",     type=int,   default=d["conv_kernel_size"])
    p.add_argument("--lr",                   type=float, default=d["lr"])
    p.add_argument("--static-embedding-dim", type=int,   default=d["static_embedding_dim"],
                   help="0 = 关闭 static encoder, 走 raw concat; >0 = encoder 输出维度。")
    return p


def _hp_from_args(args: argparse.Namespace) -> dict:
    return {
        "embedding_dim":        args.embedding_dim,
        "num_blocks":           args.num_blocks,
        "mlstm_ratio":          args.mlstm_ratio,
        "mlp_hidden":           args.mlp_hidden,
        "dropout":              args.dropout,
        "conv_kernel_size":     args.conv_kernel_size,
        "lr":                   args.lr,
        "static_embedding_dim": args.static_embedding_dim,
    }


def final_fit(
    processed_dir: Path, scaler: Scaler, meta: dict,
    hp: dict, last_epoch: int, model_dir: Path,
    *, patience: int,
) -> dict:
    """用 best_params 在 train 上拟合 + val 早停,test 上评估,保存 best 模型。"""
    max_epochs = _resolve_max_epochs(last_epoch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = make_loaders(
        processed_dir, batch_size=P.BATCH_SIZE, num_workers=P.NUM_WORKERS, pin_memory=True
    )

    model = xLSTMRegressor(
        d_seq=meta["d_seq"], d_static=meta["d_static"],
        embedding_dim=hp["embedding_dim"], num_blocks=hp["num_blocks"],
        mlstm_ratio=hp["mlstm_ratio"], mlp_hidden=hp["mlp_hidden"],
        dropout=hp["dropout"], conv_kernel_size=hp["conv_kernel_size"],
        static_embedding_dim=hp.get("static_embedding_dim", P.DEFAULT_HP["static_embedding_dim"]),
        context_length=P.WINDOW_SIZE, num_heads=4, slstm_backend="vanilla",
    ).to(device)

    print(f"Final fit: max_epochs={max_epochs}, patience={patience}, batch={P.BATCH_SIZE}")
    result = train_one_config(
        model, train_loader, val_loader, scaler,
        max_epochs=max_epochs, patience=patience,
        lr=hp["lr"], weight_decay=P.WEIGHT_DECAY,
        warmup_ratio=P.WARMUP_RATIO, grad_clip=P.GRAD_CLIP,
        device=device, optuna_trial=None,
    )
    best_state = result["best_state_dict"]
    model.load_state_dict(best_state)
    model.eval()
    test_metrics = _validate(model, test_loader, scaler, device)
    test_metrics = {f"test_{k.replace('val_', '')}": v for k, v in test_metrics.items()}

    make_dir(model_dir)
    torch.save({
        "state_dict": best_state,
        "hp": hp,
        "meta": meta,
        "max_epochs": max_epochs,
        "stopped_at_epoch": result["last_epoch"],
        "best_val_rmse": result["best_val_rmse"],
    }, model_dir / "xlstm_best.pt")

    try:
        mlflow.pytorch.log_model(model, artifact_path="model")
    except Exception as exc:
        print(f"warning: mlflow.pytorch.log_model failed: {exc!r}")
    mlflow.log_metrics(test_metrics)
    mlflow.log_metrics({
        "stopped_at_epoch": float(result["last_epoch"]),
        "best_val_rmse_final_fit": float(result["best_val_rmse"]),
    })
    mlflow.log_params({"max_epochs": max_epochs, "patience": patience, **hp})
    return test_metrics


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    set_seed(P.SEED)
    cfg = utils_mod.load_settings()
    processed_dir = utils_mod.processed_data_dir(cfg)
    model_dir = Path(cfg["MODEL_DIR"])
    tuning_dir = Path(cfg["HYPERPARAMETER_TUNING_DIR"])
    mlruns_dir = Path(cfg["MLFLOW_TRACKING_URI"])

    if not (processed_dir / "meta.json").exists():
        print(f"error: {processed_dir/'meta.json'} not found -- run prepare_data.py first "
              f"(P.WINDOW_SIZE={P.WINDOW_SIZE})")
        return 1

    params_path = tuning_dir / "best_params.json"
    trial_meta_path = tuning_dir / "best_trial_meta.json"
    if params_path.exists():
        hp = json.loads(params_path.read_text())
        hp_source = f"best_params.json ({params_path})"
    else:
        hp = _hp_from_args(args)
        hp_source = "argparse fallback (P.DEFAULT_HP overlaid by CLI)"
        print(f"info: {params_path} not found -- using {hp_source}")

    if trial_meta_path.exists():
        last_epoch = json.loads(trial_meta_path.read_text())["last_epoch"]
    else:
        print(f"warning: {trial_meta_path} not found -- defaulting last_epoch to MAX_EPOCHS - 1")
        last_epoch = P.MAX_EPOCHS - 1

    make_dir(model_dir)
    make_dir(mlruns_dir)

    meta = _read_meta(processed_dir)
    scaler = Scaler.load(processed_dir / "scaler.pkl")
    run_name = P.FINAL_FIT_RUN_NAME
    print(f"D_seq={meta['d_seq']}, D_static={meta['d_static']}, counts={meta['counts']}")
    print(f"GPU: {torch.cuda.is_available()}, "
          f"device={'cuda:'+str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'}")
    print(f"hp source: {hp_source}")
    print(f"hp: {hp}")
    print(f"last_epoch (from search): {last_epoch}")
    print(f"patience: {P.PATIENCE}")
    print(f"epochs override: {P.FINAL_FIT_EPOCHS_OVERRIDE}")

    mlflow.set_tracking_uri(f"file:{mlruns_dir}")
    mlflow.set_experiment(run_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "seed": P.SEED, "d_seq": meta["d_seq"], "d_static": meta["d_static"],
            "n_train": meta["counts"]["train"], "n_val": meta["counts"]["val"],
            "n_test": meta["counts"]["test"], "last_epoch_from_search": last_epoch,
        })
        with timer("final fit"):
            test_metrics = final_fit(
                processed_dir, scaler, meta, hp, last_epoch, model_dir,
                patience=P.PATIENCE,
            )

        print("\n=== test metrics ===")
        for k, v in test_metrics.items():
            print(f"  {k} = {v:.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
