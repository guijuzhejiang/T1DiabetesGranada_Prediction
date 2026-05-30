"""train_optuna.py: 用 Optuna 搜索最优超参,持久化到 SQLite + best_params.json.

流程:
1. 读 processed npy + scaler.pkl
2. parent MLflow run "{STUDY_NAME}_parent"
3. study = optuna.create_study(TPESampler + MedianPruner), storage='sqlite:///<db>',
   load_if_exists=True (中断后 `python train_optuna.py` 自动续跑)
4. objective(trial): 采样超参 → 构造模型 + loaders → train_one_config → 返回 best_val_rmse
5. 保存 study.pkl + best_params.json + best_trial_meta.json(供 train.py 用)

每个 epoch 控制台都会打印 train_loss / val_rmse / val_mae / val_r2 / lr / 耗时。

所有可调参数集中在 t1d_granada/params.py:
- N_TRIALS, MAX_EPOCHS, PATIENCE, WEIGHT_DECAY, WARMUP_RATIO, GRAD_CLIP
- STUDY_NAME, OPTUNA_STORAGE, OPTUNA_N_STARTUP_TRIALS, OPTUNA_N_WARMUP_STEPS
- OPTUNA_SEARCH_SPACE

实验跟踪:
- SQLite db: hyperparameter_tuning/{STUDY_NAME}.db
  → optuna-dashboard sqlite:///hyperparameter_tuning/xlstm_search.db
- MLflow runs: mlruns/  (mlflow ui --backend-store-uri ./mlruns)

完成后跑 `python train.py` 用搜到的最优超参做最终 fit。

GPU: 用环境变量 CUDA_VISIBLE_DEVICES=1 指定具体卡, 例如:
    CUDA_VISIBLE_DEVICES=1 python train_optuna.py
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import mlflow
import optuna
import torch
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from t1d_granada import params as P
from t1d_granada.dataset import make_loaders
from t1d_granada.model import xLSTMRegressor
from t1d_granada.scaler import Scaler
from t1d_granada.trainer import train_one_config
from t1d_granada import utils as utils_mod
from t1d_granada.utils import make_dir, set_seed, timer


# ---------------------------------------------------------------------------
# Optuna helpers
# ---------------------------------------------------------------------------

def _suggest(trial: optuna.Trial, name: str, spec: dict):
    typ = spec["type"]
    if typ == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    if typ == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    if typ == "float":
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False),
                                    step=spec.get("step"))
    raise ValueError(f"unknown spec type: {typ}")


def sample_hyperparams(trial: optuna.Trial) -> dict:
    return {name: _suggest(trial, name, spec) for name, spec in P.OPTUNA_SEARCH_SPACE.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _read_meta(processed_dir: Path) -> dict:
    return json.loads((processed_dir / "meta.json").read_text())


def _build_loaders(processed_dir: Path, batch_size: int, num_workers: int):
    return make_loaders(processed_dir, batch_size=batch_size, num_workers=num_workers, pin_memory=True)


def _resolve_storage(tuning_dir: Path, study_name: str) -> tuple[str | None, Path | None]:
    """根据 P.OPTUNA_STORAGE 解析 storage URL 与 db 文件路径。

    - None  → 默认 sqlite:///<tuning_dir>/<study_name>.db
    - "memory" → 纯内存,返回 (None, None)
    - 其他字符串 → 直接当 RDB URL 用,db_path=None (不知道具体文件位置)
    """
    cfg = P.OPTUNA_STORAGE
    if cfg is None:
        db_path = (tuning_dir / f"{study_name}.db").resolve()
        return f"sqlite:///{db_path}", db_path
    if isinstance(cfg, str) and cfg.lower() == "memory":
        return None, None
    return cfg, None


def run_study(
    processed_dir: Path, scaler: Scaler, meta: dict, n_trials: int,
    *, study_name: str, storage: str | None, patience: int,
) -> optuna.Study:
    """Run TPE+Median search; each trial logged as a nested MLflow run.

    `storage` 是 RDB URL(典型 `sqlite:///path/to/xxx.db`)。传 None 走纯内存,
    适合单测;生产/调试都建议用 SQLite 以便 optuna-dashboard 复盘 + 中断续跑。
    """
    d_seq = meta["d_seq"]
    d_static = meta["d_static"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial: optuna.Trial) -> float:
        hp = sample_hyperparams(trial)
        prefix = f"[trial {trial.number + 1:>3}/{n_trials}] "
        print(f"\n{prefix}params: {hp}", flush=True)
        train_loader, val_loader, _ = _build_loaders(
            processed_dir, batch_size=P.BATCH_SIZE, num_workers=P.NUM_WORKERS
        )
        model = xLSTMRegressor(
            d_seq=d_seq, d_static=d_static,
            embedding_dim=hp["embedding_dim"], num_blocks=hp["num_blocks"],
            mlstm_ratio=hp["mlstm_ratio"], mlp_hidden=hp["mlp_hidden"],
            dropout=hp["dropout"], conv_kernel_size=hp["conv_kernel_size"],
            static_embedding_dim=hp["static_embedding_dim"],
            context_length=P.WINDOW_SIZE, num_heads=4, slstm_backend="vanilla",
        )
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            mlflow.log_params(hp)
            try:
                result = train_one_config(
                    model, train_loader, val_loader, scaler,
                    max_epochs=P.MAX_EPOCHS, patience=patience,
                    lr=hp["lr"], weight_decay=P.WEIGHT_DECAY,
                    warmup_ratio=P.WARMUP_RATIO, grad_clip=P.GRAD_CLIP,
                    device=device, optuna_trial=trial, log_prefix=prefix,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{prefix}CUDA OOM → pruned", flush=True)
                raise optuna.TrialPruned("CUDA OOM")
            mlflow.log_metric("best_val_rmse", result["best_val_rmse"])
            mlflow.log_metric("best_val_mae", result["best_val_mae"])
            mlflow.log_metric("best_val_r2", result["best_val_r2"])
            trial.set_user_attr("last_epoch", result["last_epoch"])
            trial.set_user_attr("best_val_mae", result["best_val_mae"])
            trial.set_user_attr("best_val_r2", result["best_val_r2"])
        print(
            f"{prefix}done: best_val_rmse={result['best_val_rmse']:.3f} mg/dL "
            f"(mae={result['best_val_mae']:.3f}, r2={result['best_val_r2']:+.4f}, "
            f"last_epoch={result['last_epoch']})",
            flush=True,
        )
        return result["best_val_rmse"]

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=TPESampler(seed=P.SEED),
        pruner=MedianPruner(
            n_startup_trials=P.OPTUNA_N_STARTUP_TRIALS,
            n_warmup_steps=P.OPTUNA_N_WARMUP_STEPS,
        ),
    )
    n_done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n_done > 0:
        print(f"resuming study '{study_name}' with {n_done} completed trials", flush=True)
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    return study


def main() -> int:
    # Optuna 默认 WARN,这里抬到 INFO 让 trial 开始/结束、最优更新都打到 stderr
    optuna.logging.set_verbosity(optuna.logging.INFO)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

    set_seed(P.SEED)
    cfg = utils_mod.load_settings()
    processed_dir = utils_mod.processed_data_dir(cfg)
    tuning_dir = Path(cfg["HYPERPARAMETER_TUNING_DIR"])
    mlruns_dir = Path(cfg["MLFLOW_TRACKING_URI"])

    if not (processed_dir / "meta.json").exists():
        print(f"error: {processed_dir/'meta.json'} not found -- run prepare_data.py first "
              f"(P.WINDOW_SIZE={P.WINDOW_SIZE})")
        return 1

    make_dir(tuning_dir)
    make_dir(mlruns_dir)

    study_name = P.STUDY_NAME
    n_trials = P.N_TRIALS
    patience = P.PATIENCE
    storage, db_path = _resolve_storage(tuning_dir, study_name)

    meta = _read_meta(processed_dir)
    scaler = Scaler.load(processed_dir / "scaler.pkl")
    print(f"D_seq={meta['d_seq']}, D_static={meta['d_static']}, counts={meta['counts']}")
    print(f"GPU: {torch.cuda.is_available()}, "
          f"device={'cuda:'+str(torch.cuda.current_device()) if torch.cuda.is_available() else 'cpu'}")
    print(f"study_name: {study_name}")
    print(f"storage:    {storage if storage else '(in-memory, NOT persisted)'}")
    print(f"n_trials:   {n_trials}")
    print(f"patience:   {patience} (early stop after N epochs without val_rmse improvement)\n")

    mlflow.set_tracking_uri(f"file:{mlruns_dir}")
    mlflow.set_experiment(study_name)

    with mlflow.start_run(run_name=f"{study_name}_parent"):
        mlflow.log_params({
            "n_trials": n_trials, "max_epochs": P.MAX_EPOCHS, "patience": patience,
            "seed": P.SEED, "d_seq": meta["d_seq"], "d_static": meta["d_static"],
            "n_train": meta["counts"]["train"], "n_val": meta["counts"]["val"],
            "n_test": meta["counts"]["test"],
            "optuna_storage": storage or "memory",
            "optuna_study_name": study_name,
        })

        with timer("optuna study"):
            study = run_study(
                processed_dir, scaler, meta, n_trials,
                study_name=study_name, storage=storage, patience=patience,
            )

        with open(tuning_dir / "study.pkl", "wb") as f:
            pickle.dump(study, f)
        (tuning_dir / "best_params.json").write_text(json.dumps(study.best_params, indent=2))
        # Persist last_epoch separately so train.py doesn't need to unpickle the study.
        last_epoch = study.best_trial.user_attrs.get("last_epoch", P.MAX_EPOCHS - 1)
        (tuning_dir / "best_trial_meta.json").write_text(json.dumps({
            "trial_number": study.best_trial.number,
            "best_val_rmse": study.best_value,
            "last_epoch": last_epoch,
            "study_name": study_name,
            "storage": storage or "memory",
        }, indent=2))

        print(f"\n{'='*70}")
        print(f"search complete: {len(study.trials)} trials total")
        print(f"  completed: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
        print(f"  pruned:    {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        print(f"  failed:    {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
        print(f"\nbest trial #{study.best_trial.number}: val_rmse={study.best_value:.4f} mg/dL")
        print(f"  params:     {study.best_params}")
        print(f"  last_epoch: {last_epoch}")
        if db_path is not None:
            print(f"\nartifacts:")
            print(f"  {tuning_dir / 'best_params.json'}")
            print(f"  {tuning_dir / 'best_trial_meta.json'}")
            print(f"  {tuning_dir / 'study.pkl'}")
            print(f"  {db_path}")
            print(f"\ninspect via:  optuna-dashboard sqlite:///{db_path}")
        print(f"\nNext: run `python train.py` to fit the best params on train+val and evaluate on test.")
        mlflow.log_metric("best_val_rmse_overall", study.best_value)

    return 0


if __name__ == "__main__":
    sys.exit(main())
