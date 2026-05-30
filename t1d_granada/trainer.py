"""训练循环 + MLflow 记录 + Early Stopping。

`train_one_config(...)` 是 trial 内核与最终 fit 共用的执行单元:
- MSE on z-score, AdamW + CosineAnnealingLR + 可选 warmup
- val: 反归一化后报 RMSE / MAE / R²
- 早停: 连续 patience 次 val_rmse 未下降即停, 返回 best state_dict
- 可选 optuna_trial: 每 epoch 调 trial.report(val_rmse, step=epoch); should_prune → 抛 TrialPruned

返回 dict {best_val_rmse, best_state_dict, last_epoch, history}.
"""
from __future__ import annotations

import math
import time
from copy import deepcopy
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from t1d_granada.dataset import denormalize_bg
from t1d_granada.scaler import Scaler


def _validate(
    model: nn.Module, loader: DataLoader, scaler: Scaler, device: torch.device,
    *, desc: str = "val", show_progress: bool = True,
) -> dict[str, float]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    iterator = loader
    if show_progress:
        iterator = tqdm(loader, desc=desc, total=len(loader), leave=False, dynamic_ncols=True)
    with torch.no_grad():
        for seq, static, tgt in iterator:
            seq = seq.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)
            out = model(seq, static)
            preds.append(out.detach().cpu().numpy())
            targets.append(tgt.numpy())
    pred_z = np.concatenate(preds)
    tgt_z = np.concatenate(targets)
    # denormalize for human-readable mg/dL metrics
    pred_mg = denormalize_bg(pred_z, scaler)
    tgt_mg = denormalize_bg(tgt_z, scaler)

    err = pred_mg - tgt_mg
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((tgt_mg - tgt_mg.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"val_rmse": rmse, "val_mae": mae, "val_r2": r2}


def train_one_config(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    scaler: Scaler,
    *,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    warmup_ratio: float = 0.0,
    grad_clip: float | None = None,
    device: torch.device | None = None,
    optuna_trial: Any | None = None,
    log_to_mlflow: bool = True,
    verbose: bool = True,
    log_prefix: str = "",
) -> dict:
    """Train `model` for up to `max_epochs`; early-stop on val_rmse plateau.

    Returns dict with `best_val_rmse`, `best_val_mae`, `best_val_r2`,
    `best_state_dict`, `last_epoch`, `history`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if len(train_loader) == 0:
        raise ValueError("train_loader is empty")

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Cosine schedule with optional linear warmup
    warmup_epochs = max(1, int(round(max_epochs * warmup_ratio))) if warmup_ratio > 0 else 0

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        # cosine over the remaining epochs
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn = nn.MSELoss()

    best_rmse = float("inf")
    best_metrics: dict[str, float] = {}
    best_state: dict[str, torch.Tensor] | None = None
    no_improve = 0
    history: list[dict[str, float]] = []
    last_epoch = -1

    n_batches = len(train_loader)
    for epoch in range(max_epochs):
        last_epoch = epoch
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_seen = 0
        epoch_desc = f"{log_prefix}epoch {epoch + 1:>3}/{max_epochs} train"
        batch_iter = train_loader
        progress = None
        if verbose:
            progress = tqdm(
                train_loader, desc=epoch_desc, total=n_batches,
                leave=False, dynamic_ncols=True, mininterval=0.3,
            )
            batch_iter = progress
        for seq, static, tgt in batch_iter:
            seq = seq.to(device, non_blocking=True)
            static = static.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(seq, static)
            loss = loss_fn(pred, tgt)
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            batch_loss = float(loss.item())
            running_loss += batch_loss * seq.size(0)
            n_seen += seq.size(0)
            if progress is not None:
                progress.set_postfix(
                    loss=f"{batch_loss:.4f}",
                    avg=f"{running_loss / max(1, n_seen):.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                    refresh=False,
                )
        if progress is not None:
            progress.close()

        scheduler.step()
        train_loss = running_loss / max(1, n_seen)
        val_metrics = _validate(
            model, val_loader, scaler, device,
            desc=f"{log_prefix}epoch {epoch + 1:>3}/{max_epochs} val",
            show_progress=verbose,
        )
        epoch_time = time.time() - t0

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time,
        }
        history.append(record)

        if log_to_mlflow and mlflow.active_run() is not None:
            mlflow.log_metrics(
                {k: float(v) for k, v in record.items() if k != "epoch"}, step=epoch
            )

        improved = val_metrics["val_rmse"] < best_rmse
        if improved:
            best_rmse = val_metrics["val_rmse"]
            best_metrics = dict(val_metrics)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose:
            mark = " ✓" if improved else "  "
            print(
                f"{log_prefix}epoch {epoch + 1:>3}/{max_epochs} | "
                f"train_loss={train_loss:.4f} (z) | "
                f"val_rmse={val_metrics['val_rmse']:7.3f} mg/dL | "
                f"val_mae={val_metrics['val_mae']:7.3f} | "
                f"val_r2={val_metrics['val_r2']:+.4f} | "
                f"lr={optimizer.param_groups[0]['lr']:.2e} | "
                f"t={epoch_time:5.1f}s{mark}",
                flush=True,
            )

        if optuna_trial is not None:
            optuna_trial.report(val_metrics["val_rmse"], step=epoch)
            if optuna_trial.should_prune():
                import optuna

                if verbose:
                    print(f"{log_prefix}pruned at epoch {epoch + 1} (val_rmse={val_metrics['val_rmse']:.3f})",
                          flush=True)
                raise optuna.TrialPruned()

        if no_improve >= patience:
            if verbose:
                print(f"{log_prefix}early stop at epoch {epoch + 1} "
                      f"(no improvement for {patience} epochs, best_val_rmse={best_rmse:.3f})",
                      flush=True)
            break

    # If even epoch 0 was best, best_state is set; if every epoch worsened, also set.
    if best_state is None:
        # No epoch ran (max_epochs=0 misuse): snapshot current weights.
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_metrics = {"val_rmse": float("inf"), "val_mae": float("inf"), "val_r2": float("nan")}

    return {
        "best_val_rmse": best_metrics.get("val_rmse", float("inf")),
        "best_val_mae": best_metrics.get("val_mae", float("inf")),
        "best_val_r2": best_metrics.get("val_r2", float("nan")),
        "best_state_dict": best_state,
        "last_epoch": last_epoch,
        "history": history,
    }
