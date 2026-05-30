"""路径解析、计时器、目录创建、随机种子工具。"""
import json
import os
import random
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np


def project_root() -> Path:
    """T1DiabetesGranada_Prediction/ 目录绝对路径。"""
    # this file: t1d_granada/utils.py → parents[1] = T1DiabetesGranada_Prediction/
    return Path(__file__).resolve().parents[1]


def load_settings() -> dict:
    """读取 settings.json 并把相对路径解析为绝对路径。"""
    root = project_root()
    with open(root / "settings.json") as f:
        cfg = json.load(f)
    resolved = {}
    for key, val in cfg.items():
        if isinstance(val, str) and val.startswith("./"):
            resolved[key] = str((root / val[2:]).resolve())
        else:
            resolved[key] = val
    return resolved


def make_dir(path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def processed_data_dir(cfg: dict | None = None) -> Path:
    """返回当前 WINDOW_SIZE 对应的 processed 子目录。

    约定: cfg["PROCESSED_DATA_DIR"] / f"WINDOW_SIZE_{P.WINDOW_SIZE}".
    这样改 P.WINDOW_SIZE 不会覆盖之前 window 的物化数据。

    使用 lazy import 避免与 params 循环依赖。
    """
    from t1d_granada import params as P
    if cfg is None:
        cfg = load_settings()
    return Path(cfg["PROCESSED_DATA_DIR"]) / f"WINDOW_SIZE_{P.WINDOW_SIZE}"


def seconds_to_hh_mm_ss(duration: float) -> str:
    h = int(duration // 3600)
    m = int((duration % 3600) // 60)
    s = duration % 60
    return f"{h}h {m}m {s:.2f}s"


@contextmanager
def timer(name: str):
    print(f"{datetime.now()} - [{name}] ...")
    t0 = time.time()
    yield
    print(f"{datetime.now()} - [{name}] done in {seconds_to_hh_mm_ss(time.time() - t0)}\n")


def set_seed(seed: int) -> None:
    """同步 python / numpy / torch 随机种子, 含 cuda."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
