# T1DiabetesGranada · xLSTM 30-min 血糖预测

> English version: [README.md](README.md)

基于 [T1DiabetesGranada 数据集](https://www.nature.com/articles/s41597-023-02737-4)的 1 型糖尿病患者 CGM 数据，用 **xLSTM 2.0.5** 训练一个回归模型，输入过去 2 小时（8 个 15-min 采样点）的多模态特征，预测 **30 分钟后**的血糖值（mg/dL）。

> 本目录是仓库根 BrisT1D LightGBM 方案的独立子项目，**代码、数据、栈完全不共享**。

---

## 技术栈

| 组件 | 版本 / 选型 | 用途 |
|---|---|---|
| Python | 3.12 | conda env `py312_cu121` |
| PyTorch | ≥ 2.4（CUDA 12.x） | 训练框架 |
| xlstm | **2.0.5** | sLSTM/mLSTM 块（`slstm_backend='vanilla'`，避开 Triton） |
| Optuna | ≥ 3.5 | TPE + MedianPruner 超参搜索（SQLite 持久化） |
| MLflow | ≥ 2.10 | 实验跟踪（params/metrics/artifacts） |
| SHAP | ≥ 0.45 | 特征重要性归因（`GradientExplainer`） |
| numpy | **< 2.3** | 兼容 numba/shap，**不可升级** |

完整列表见 [requirements.txt](requirements.txt)。

---

## 任务定义

| 项 | 值 |
|---|---|
| 输入序列长度 | 8 步 × 15 min = 2 小时 |
| 预测目标 | `last_input_ts + 30 min` 时刻的 `bg`（mg/dL） |
| 序列特征 (D_seq=6) | `bg_z`, `bg_diff`, `tod_sin/cos`, `dow_sin/cos` |
| 静态特征 (D_static=12) | `Sex_M`, `Age_z` + 2h/6h 滚动统计 × 5（mean/std/tir_70_180/p5/p95） |
| 训练损失 | z-score 域 MSE |
| 验证 / 报告指标 | mg/dL 域 RMSE / MAE / R²（反归一化后） |
| 切分 | per-patient chronological **80 / 10 / 10** |

> 滚动统计严格因果：`closed='left'` ⇒ 窗口 `[t-w, t)` **不含** anchor 行，杜绝未来泄漏。

---

## 目录结构

```
T1DiabetesGranada_Prediction/
├── prepare_data.py        # 阶段 1: 原始 CSV → npy + scaler + meta
├── train_optuna.py        # 阶段 2: Optuna 超参搜索
├── train.py               # 阶段 3: 用 best_params 做最终 fit
├── predict.py             # 阶段 4: 单条 / 批量推理
├── shap_analyze.py        # 阶段 5（可选）: SHAP 归因 + 4 类图
├── settings.json          # 路径配置（数据 / 模型 / mlruns / reports）
├── requirements.txt
├── t1d_granada/           # 核心包
│   ├── params.py          # ★ 所有可调参数集中于此 ★
│   ├── window_builder.py  # 矢量化窗口构造
│   ├── rolling_stats.py   # 严格因果 2h/6h 滚动统计
│   ├── feature_engineer.py# 衍生特征 (bg_diff, time-of-day...)
│   ├── scaler.py          # 12 字段 z-score scaler（仅 train fit）
│   ├── dataset.py         # PyTorch Dataset / DataLoader
│   ├── model.py           # xLSTMRegressor
│   ├── trainer.py         # 训练循环 + early stop + tqdm
│   ├── shap_analysis.py   # SHAP wrapper + 绘图
│   └── utils.py           # settings / seed / timer / make_dir
├── tests/                 # pytest 单测 + smoke
├── docs/
│   ├── plans/             # 设计文档
│   └── solutions/         # 已解决问题 / 决策（带 frontmatter）
└── scripts/
    └── dump_processed_head.py  # 把 npy 头 N 行落 csv 方便查看
```

---

## 快速开始

### 1. 环境

```bash
conda create -n py312_cu121 python=3.12 -y
conda activate py312_cu121
pip install -r requirements.txt
```

> ⚠️ `numpy<2.3` 是硬约束，否则 `import shap` 会因 numba 冲突崩。

### 2. 准备数据

把 T1DiabetesGranada 的两个原始 CSV 放到 `settings.json` 里 `RAW_DATA_DIR` 指向的目录：

```
data/
├── Glucose_measurements.csv
└── Patient_info.csv
```

```bash
cd T1DiabetesGranada_Prediction
python prepare_data.py
```

物化产物（`data/processed/`）：

| 文件 | 形状 / 内容 |
|---|---|
| `{train,val,test}_seq.npy` | `(N, 8, 6)` float32 |
| `{train,val,test}_static.npy` | `(N, 12)` float32 |
| `{train,val,test}_target.npy` | `(N,)` float32（z-score） |
| `scaler.pkl` | `Scaler` 对象（仅 train fit） |
| `meta.json` | 形状、特征名、计数、配置 — **单一可信来源** |

典型规模：train ≈ 16.27M / val ≈ 2.03M / test ≈ 2.03M。

想直观看每行的物理含义：

```bash
PYTHONPATH=. python scripts/dump_processed_head.py
# → data/processed/head20/*.csv，列名带物理偏移如 t-45_bg_z, target_mg_dL_at_+30min
```

### 3. 超参搜索

```bash
CUDA_VISIBLE_DEVICES=1 python train_optuna.py
```

跑完产出：

```
hyperparameter_tuning/
├── best_params.json        # 最佳超参（供 train.py 用）
├── best_trial_meta.json    # trial 编号 + last_epoch + best_val_rmse
├── study.pkl               # Optuna study 对象
└── xlstm_search.db         # SQLite 持久化（dashboard 复盘用）
```

中断后再跑同一条命令会**自动续跑**（`load_if_exists=True`）。

### 4. 最终拟合 + test 评估

```bash
CUDA_VISIBLE_DEVICES=1 python train.py
```

产出：

- `model/xlstm_best.pt` — best-on-val checkpoint（含 `state_dict / hp / meta / max_epochs / stopped_at_epoch / best_val_rmse`）
- 控制台报告 test 集 RMSE / MAE / R²（mg/dL 域）

> 训练用 `train` 集，监控用 `val` 集触发早停（`P.PATIENCE`），test 集仅在最后评估一次 — 全程零泄漏。

### 5. 推理

```bash
# 单条
python predict.py --history hist.csv --patient_id P001 \
                  --last_ts "2025-03-15 08:30:00" --sex M --birth_year 1985

# 批量
python predict.py --history hist.csv --input batch.csv --output out.csv
```

`hist.csv` 至少要覆盖 `[last_ts - 6h, last_ts]` 范围内的 CGM 数据，否则 6h rolling 统计的 `min_periods` 不足，该样本会被跳过。

### 6. SHAP 解释（可选）

```bash
python shap_analyze.py
# → reports/shap/{summary.csv, feature_importance.png, timestep_importance.png, ...}
```

---

## 配置（`t1d_granada/params.py`）

**所有可调参数集中在 [t1d_granada/params.py](t1d_granada/params.py)，无 CLI flag**。改这里就行：

```python
# ----- 数据 / 窗口 -----
WINDOW_SIZE        = 8         # 输入序列步数
FORECAST_STEPS     = 2         # +30 min 目标的偏移（2 个 15-min 步）
SAMPLE_INTERVAL_MIN = 15
ROLLING_WINDOWS    = [2, 6]    # 小时
ROLLING_STATS      = ["mean", "std", "tir_70_180", "p5", "p95"]

# ----- 切分 -----
SPLIT_TRAIN, SPLIT_VAL = 0.8, 0.1   # test = 1 - train - val

# ----- 训练 -----
BATCH_SIZE     = 512
NUM_WORKERS    = 8
MAX_EPOCHS     = 20            # 上限,实际由早停决定
PATIENCE       = 3             # 连续 N 个 epoch val_rmse 未优化 → 早停
WEIGHT_DECAY   = 1e-5
WARMUP_RATIO   = 0.05          # 余弦调度 + 线性 warmup
GRAD_CLIP      = 1.0

# ----- Optuna -----
N_TRIALS                = 25
OPTUNA_N_STARTUP_TRIALS = 5
OPTUNA_N_WARMUP_STEPS   = 5
STUDY_NAME              = "xlstm_search"
OPTUNA_STORAGE          = None  # None=默认 sqlite, "memory"=不持久化, str=自定义 RDB URL

# ----- train.py -----
FINAL_FIT_EPOCH_MULT       = 1.2     # max_epochs = (last_epoch+1) * MULT
FINAL_FIT_EPOCHS_OVERRIDE  = None    # int=强制 max_epochs,跳过 MULT 推算
FINAL_FIT_RUN_NAME         = "xlstm_final_fit"

# ----- 复现 -----
SEED = 42

# ----- 搜索空间 -----
OPTUNA_SEARCH_SPACE = {
    "embedding_dim":    {"type": "categorical", "choices": [64, 96, 128, 192, 256]},
    "num_blocks":       {"type": "int", "low": 2, "high": 6},
    "mlstm_ratio":      {"type": "categorical", "choices": [0.0, 0.5, 1.0]},
    "mlp_hidden":       {"type": "categorical", "choices": [64, 128, 256]},
    "dropout":          {"type": "float", "low": 0.0, "high": 0.4, "step": 0.05},
    "conv_kernel_size": {"type": "categorical", "choices": [2, 3, 4]},
    "lr":               {"type": "float", "low": 1e-5, "high": 2e-4, "log": True},
    "batch_size":       {"type": "categorical", "choices": [256, 512, 1024]},
}
```

路径配置在 [settings.json](settings.json)：`PROCESSED_DATA_DIR / MODEL_DIR / HYPERPARAMETER_TUNING_DIR / MLFLOW_TRACKING_URI / REPORTS_DIR / SHAP_REPORTS_DIR`。

---

## 实验跟踪

### MLflow

```bash
mlflow ui --backend-store-uri ./mlruns
# → http://localhost:5000
```

实验结构（嵌套 run）：

```
xlstm_search                    ← experiment
├── xlstm_search_parent         ← parent run（study 级别 params + best_val_rmse_overall）
│   ├── trial_0                 ← nested run（每个 trial 的超参 + 每 epoch 曲线）
│   ├── trial_1
│   └── ...
└── ...

xlstm_final_fit                 ← experiment（train.py）
└── xlstm_final_fit             ← run（test 指标 + 训练曲线 + pytorch model artifact）
```

> Per-epoch 曲线（`train_loss / val_rmse / val_mae / val_r2 / lr / epoch_time`）在 **trial run** 上，不是 parent。在 UI 主页展开 parent 才能看到子 trial。

### Optuna Dashboard

```bash
optuna-dashboard sqlite:///hyperparameter_tuning/xlstm_search.db
# → http://localhost:8080
```

或在 Python 里：

```python
import optuna
study = optuna.load_study(
    study_name="xlstm_search",
    storage="sqlite:///hyperparameter_tuning/xlstm_search.db",
)
print(study.trials_dataframe())
```

---

## 数据契约速查

### 序列特征顺序（按 `meta.json` 的 `seq_feature_names`）

| index | 名 | 含义 |
|---|---|---|
| 0 | `bg_z` | z-score 归一化血糖（按 train 统计） |
| 1 | `bg_diff` | 一阶差分（与上一时间点 bg_z 之差，t=0 处补 0） |
| 2-3 | `tod_sin / cos` | 一天内时刻三角编码 |
| 4-5 | `dow_sin / cos` | 星期几三角编码（`(day+3) % 7` 让周一对齐 0） |

时间偏移：`t=0..7` 对应 `[-105min, -90min, -75min, -60min, -45min, -30min, -15min, 0]`（相对 `last_input_ts`），target 在 `last_input_ts + 30min`。

### 静态特征顺序（按 `meta.json` 的 `static_cols`）

```
[0]  Sex_M                     # 独热（M=1, F=0）
[1]  Age_z                     # z-score 归一化
[2-6]  bg_2h_{mean,std,tir_70_180,p5,p95}    # 2 小时滚动统计
[7-11] bg_6h_{mean,std,tir_70_180,p5,p95}    # 6 小时滚动统计
```

每个滚动统计字段的 scaler 是按它**自己的分布** fit 的，**不是 `bg` 那个**。

### z-score ↔ mg/dL 转换

```python
pred_z = model(seq, static)              # (B,)
pred_mg = scaler.inverse("bg", pred_z)   # mg/dL,可读尺度
```

---

## 关键决策（务必遵守）

- **`slstm_backend='vanilla'`**：训练 / 推理 / SHAP 三条路径共用同一后端，避开 Triton 依赖和 SHAP autograd 冲突。详见 [docs/solutions/design-patterns/xlstm-shap-regressor-pattern-2026-05-09.md](docs/solutions/design-patterns/xlstm-shap-regressor-pattern-2026-05-09.md)。
- **`closed='left'` 严格因果**：所有 rolling 统计都不能包含 anchor 行；改这块代码必须配套跑因果注入测试（在 `last_input_ts` 之后注入 9999，断言预测不变）。
- **`numpy<2.3`** 必须保留在 [requirements.txt](requirements.txt)，否则 `import shap` 会被 numba 报错挡住。
- **`from t1d_granada import utils as utils_mod`** 这种 import 形式：让 `pytest.monkeypatch` 能替换 `load_settings`。直接 `from t1d_granada.utils import load_settings` 会失去可测性。
- **`train.py` 不再合并 train+val**：合并后没有 held-out 监控集，要么关早停要么用 test 监控会泄漏。保留独立 val 做早停最稳。

---

## 测试

```bash
pytest tests/ -q
```

包含：
- 单元测试（`test_dataset.py / test_feature_engineer.py / test_model.py / test_rolling_stats.py / test_scaler.py / test_shap_analysis.py / test_trainer.py / test_window_builder.py`）
- 端到端 smoke（`test_train_smoke.py / test_predict.py`）

---

## 进一步阅读

- [CLAUDE.md](CLAUDE.md) — 给 Claude Code 的项目指引（数据 schema、模块表、关键决策）
- [docs/plans/](docs/plans/) — 设计与计划文档
- [docs/solutions/](docs/solutions/) — 按类目组织的已解决问题 / 决策（含 xLSTM/SHAP 集成、`slstm_backend='vanilla'` 选择、因果滚动统计契约、`numpy<2.3` 环境约束等）

## 引用

- T1DiabetesGranada 数据集：Rodríguez-Rodríguez, I. et al. *Sci Data* 10, 916 (2023). <https://www.nature.com/articles/s41597-023-02737-4>
- xLSTM：Beck, M. et al. *NeurIPS* (2024). <https://arxiv.org/abs/2405.04517>
