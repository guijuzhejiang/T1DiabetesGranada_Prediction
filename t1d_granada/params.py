"""集中所有可调参数 / flag / 阈值。

- 数据 / 窗口构造: WINDOW_SIZE, FORECAST_STEPS, STRIDE, SAMPLE_INTERVAL_MIN, TIME_TOLERANCE_MIN
- 衍生序列特征 flag: USE_BG_DIFF, USE_TIME_OF_DAY, USE_DAY_OF_WEEK
- 静态滚动统计: ROLLING_WINDOWS (单位 hour), ROLLING_STATS, TIR_LOW, TIR_HIGH
- 切分: SPLIT_TRAIN, SPLIT_VAL (按时间, per-patient chronological)
- Patient 过滤: BIRTH_YEAR_MIN/MAX
- 训练: BATCH_SIZE, NUM_WORKERS, MAX_EPOCHS, PATIENCE, LR, WEIGHT_DECAY
- Optuna 搜索: N_TRIALS, OPTUNA_SEARCH_SPACE
- SHAP: N_BG, N_FG
- 复现: SEED
"""

# ----- 数据 / 窗口 -----
WINDOW_SIZE = 8              # 输入序列步数 (T)
FORECAST_STEPS = 2           # 距 last input 多少个 15-min 步是预测目标 (2 → +30min)
STRIDE = 1                   # 滑窗步长 (个 15-min 槽)
SAMPLE_INTERVAL_MIN = 15
TIME_TOLERANCE_MIN = 2       # 单步间隔在 [SAMPLE_INTERVAL_MIN ± TIME_TOLERANCE_MIN] 内才合法

# ----- 序列衍生特征 (R3) -----
USE_BG_DIFF = True
USE_TIME_OF_DAY = True       # sin/cos
USE_DAY_OF_WEEK = True       # sin/cos
# D_seq = 1 (bg_z) + 1 if USE_BG_DIFF + 2 if USE_TIME_OF_DAY + 2 if USE_DAY_OF_WEEK

# ----- 静态特征: 多窗口血糖统计 (R4 / R4b) -----
# 选 2h + 6h: 兼顾"信息增量 vs 部署可用性"
ROLLING_WINDOWS = [2, 6]                                   # hours
ROLLING_STATS = ["mean", "std", "tir_70_180", "p5", "p95"] # 5 个 / 窗口
TIR_LOW = 70                                                # mg/dL
TIR_HIGH = 180                                              # mg/dL
# D_static = 2 (Sex_M, Age_z) + len(ROLLING_WINDOWS) * len(ROLLING_STATS) = 2 + 10 = 12

# ----- 切分 -----
SPLIT_TRAIN = 0.8
SPLIT_VAL = 0.1
# test = 1 - SPLIT_TRAIN - SPLIT_VAL

# ----- Patient sanity 过滤 -----
BIRTH_YEAR_MIN = 1900
BIRTH_YEAR_MAX = 2020

# ----- 训练 -----
BATCH_SIZE = 1024
NUM_WORKERS = 8
MAX_EPOCHS = 20
PATIENCE = 2
LR = 1e-3
WEIGHT_DECAY = 1e-5
WARMUP_RATIO = 0.05
GRAD_CLIP = 1.0

# ----- Optuna -----
N_TRIALS = 25
OPTUNA_N_STARTUP_TRIALS = 5
OPTUNA_N_WARMUP_STEPS = 5
FINAL_FIT_EPOCH_MULT = 1.2  # best trial epoch * 1.2 → 最终 fit 的 epoch 数

# Optuna study 跟踪
STUDY_NAME = "xlstm_search"     # MLflow experiment + Optuna study 名
OPTUNA_STORAGE = None           # None = sqlite:///<HYPERPARAMETER_TUNING_DIR>/<STUDY_NAME>.db
                                # "memory" = 纯内存(不持久化,不可 dashboard)
                                # 也可填自定义 RDB URL(e.g. "postgresql://...")
# train.py (最终 fit) 配置
FINAL_FIT_RUN_NAME = "xlstm_final_fit"  # MLflow experiment + run 名
FINAL_FIT_EPOCHS_OVERRIDE = None        # None = 用 best_trial_meta.json 的 last_epoch * MULT
                                        # int = 强制 max_epochs (上限,仍受早停控制)

# train.py 的 fallback 超参 (best_params.json 不存在时用,argparse 默认值)
# 注: batch_size 不参与搜索/CLI,固定使用 P.BATCH_SIZE。
DEFAULT_HP = {
    "embedding_dim":        128,
    "num_blocks":           4,
    "mlstm_ratio":          0.5,
    "mlp_hidden":           128,
    "dropout":              0.1,
    "conv_kernel_size":     3,
    "lr":                   1e-4,
    "static_embedding_dim": 32,   # 0 = 关闭(走 raw concat),其他 = static encoder 输出维度
}

OPTUNA_SEARCH_SPACE = {
    "embedding_dim":        {"type": "categorical", "choices": [128, 192, 256, 512]},
    "num_blocks":           {"type": "int", "low": 2, "high": 6},
    "mlstm_ratio":          {"type": "categorical", "choices": [0.0, 0.5, 1.0]},
    "mlp_hidden":           {"type": "categorical", "choices": [256, 512, 1024]},
    "dropout":              {"type": "float", "low": 0.0, "high": 0.4, "step": 0.05},
    "conv_kernel_size":     {"type": "categorical", "choices": [2, 3, 4]},
    "lr":                   {"type": "float", "low": 5e-5, "high": 3e-3, "log": True},
    "static_embedding_dim": {"type": "categorical", "choices": [0, 32, 64]},
}

# ----- SHAP -----
N_BG = 256                     # background 集大小
N_FG = 1024                    # foreground 集大小
SHAP_FORCE_PLOT_SAMPLES = 5    # 选几个样本做 force plot

# ----- 复现 -----
SEED = 42
