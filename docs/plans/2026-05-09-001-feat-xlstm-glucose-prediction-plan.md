---
title: T1DiabetesGranada · 30-min 血糖预测 (xLSTM) 实施计划
type: feat
status: active
date: 2026-05-09
origin: T1DiabetesGranada_Prediction/REQUIREMENTS.md
---

# T1DiabetesGranada · 30-min 血糖预测 (xLSTM) 实施计划

## Summary

在 `T1DiabetesGranada_Prediction/` 目录下落地一个 PyTorch + xLSTM 的 30 min 血糖预测 pipeline，复用 BrisT1D 1st-place 项目的"窗口扩张 + 特征工程 + Optuna 调参"骨架，但替换为单变量短序列 (T=4)、xLSTMBlockStack 模型、MLflow 实验追踪、SHAP 可解释性输出。结构上拆为 8 个有序 implementation unit，从环境与脚手架到 SHAP 报告每步可独立验证。

---

## Problem Frame

需求由 `T1DiabetesGranada_Prediction/REQUIREMENTS.md` 完整定义（与 BrisT1D 项目的差异见该文档第 10 节）。计划层面要解决的核心问题是：把单变量、严格 15-min 间隔、22.67M 行 / 736 患者的 CGM 数据，转成 xLSTM 可吃的张量；用 Optuna 搜参 + MLflow 跟踪；末了用 SHAP 给出特征重要性、时间步重要性两类可视化。环境侧已发现 `shap 0.51 → numba 0.61.2` 与 `numpy 2.3` 冲突，需在脚手架阶段先修。

---

## Requirements

每条对应 `T1DiabetesGranada_Prediction/REQUIREMENTS.md` 中的章节。

- **R1.** 窗口构造严格 15-min 对齐、stride=1、跨 NaN/大缺口的窗口直接丢弃（origin §2.4）
- **R2.** 每患者按时间切 80/10/10，scaler 仅在训练集上拟合（origin §2.3 / §2.5）
- **R3.** 时序输入：`bg_z`、`bg_diff`、`time_of_day_sin/cos`、`day_of_week_sin/cos` 由 `params.py` flag 控制，默认 ON → D_seq=6（origin §3.1）
- **R4.** 静态输入：`Sex_M` (M=1, F=0) + `Age_z` + 多窗口血糖统计（`bg_{2h|6h}_{mean|std|tir_70_180|p5|p95}` 共 10 列），D_static=**12**；不使用 patient embedding 与 Diagnostics（origin §3.2）。窗口选择 2h/6h 是为兼顾"信息增量 vs 部署可用性"——2h 几乎总能拿到、6h 在 Libre 类设备的常规场景下也可用。
- **R4b.** 多窗口统计严格因果（`closed='left'`，不含窗口右端点本身）；窗口内可用点数 < `min_periods = window_points // 2`（2h: <4 点、6h: <12 点）的样本整体丢弃（origin §3.2）
- **R5.** 模型：xLSTMBlockStack → 取 `y[:, -1, :]` → concat(static) → 2 层 MLP → 标量；mLSTM/sLSTM 比例可配（origin §4.1）
- **R6.** 训练：MSE on z-score；AdamW + cosine LR + early stopping by val RMSE（反归一化）（origin §4.2）
- **R7.** Optuna：8 个超参、25–30 trials、TPESampler + MedianPruner，目标 = 反归一化 val RMSE（origin §4.3）
- **R8.** MLflow：每 trial + 最终 fit 各一个 run，记录超参、loss/RMSE/MAE/R² 曲线、模型 artifact（origin §4.4）
- **R9.** 评估指标：RMSE / MAE / R²，反归一化后 mg/dL（origin §5）
- **R10.** SHAP：4 类图（特征重要性条形、时间步重要性条形、time×feature 热力图、5 个单样本 force plot），背景集 256、目标集 1024（origin §6）
- **R11.** 交付：`prepare_data.py` / `train.py` / `predict.py` / `shap_analyze.py` + `t1d_granada/` 包 + `settings.json` + `requirements.txt`（origin §7）
- **R12.** 成功阈值：test RMSE < 18 mg/dL、MAE < 13、R² > 0.85；单 seed 三次重跑 RMSE 标准差 < 0.5（origin §1）

---

## Scope Boundaries

继承自 origin §8。

- 不做多步预测、不做模型对比、不做部署 / API 服务、不做 Clarke Error Grid / MARD（首版）
- 不做嵌套 CV、不做 patient-level held-out
- 不构建 `end_to_end.py`（保持 BrisT1D 风格，3 + 1 个独立脚本）
- 不使用 `Diagnostics.csv`、`Biochemical_parameters.csv`、`Preprocessed_data*.npy`、`Resampled_data.npy`
- 不引入 patient embedding 或 ICD-9 编码

### Deferred to Follow-Up Work

- 临床指标（MARD、Clarke Error Grid）：可在 `predict.py` 输出后单独脚本计算
- per-patient 归一化的消融实验：作为后续工作
- 多步预测扩展（+15/+45/+60 min）：模型结构已支持，仅需改输出层

---

## Context & Research

### Relevant Code and Patterns

借鉴自 BrisT1D 1st-place（同仓库根目录，作为结构与思路模板，非直接 import）：

- `prepare_data.py` 的 expand→transform 双阶段拆分 → 本计划简化为单阶段窗口构造（无需 data_expander，stride=1 已自然产生大量窗口）
- `brist1d/utils.py` 的 `timer` / `make_dir` 工具 + `settings.json` 路径解析模式
- `brist1d/params.py` 的"集中超参 + flag 注释"风格（[brist1d/params.py:1](../../../brist1d/params.py#L1)）
- 文件名 suffix 模式：`{model}_gap_{...}_..._{suffix}.pkl` 区分实验配置 → 本项目用更简单的 `xlstm_{suffix}.pt`

xLSTM 2.0.5 实测约束（已验证）：

- `xLSTMBlockStack(B, T, D=embedding_dim) → (B, T, D)`，需在前面加 `Linear(D_seq → embedding_dim)`
- `mLSTMLayerConfig.conv1d_kernel_size` 默认 4；T=4 可用但保险起见在配置里允许 Optuna 搜 {2,3,4}
- `slstm_at: list[int]` 指定哪些 block index 用 sLSTM；空列表 = 纯 mLSTM
- `mLSTMLayerConfig.embedding_dim` / `context_length` 须与 `xLSTMBlockStackConfig` 同步设置（库内部依赖）

### Institutional Learnings

仓库尚无 `docs/solutions/`，无可复用的过往学习记录。本计划完成后建议把"xlstm 2.0.5 + numpy 2.3 冲突"经验写成一条 solution 文档供后续项目复用。

### External References

- xlstm 2.0.5 README + 源码（已通过本地 inspect 验证 API）
- SHAP `GradientExplainer` 文档：原生支持 PyTorch 多输入模型，传 `[seq_input, static_input]` 列表
- MLflow `mlflow.pytorch.log_model` 自动序列化 `state_dict` + 框架元信息

---

## Key Technical Decisions

- **窗口构造一次性物化为 `.npy`**：22M 点 × 6 (4 input + 2 forecast)，物化后单文件约 1 GB float32 / 500 MB float16；优于每个 epoch 重做窗口对齐。落到 `data/processed/{train,val,test}_{seq,static,target}.npy`。
- **使用 numpy memmap 加载到 PyTorch Dataset**：避免一次性把 1 GB 拉进内存，配合 `num_workers > 0` 的 DataLoader。
- **xLSTM 输入投影 = `Linear(D_seq, embedding_dim) + LayerNorm`**：D_seq 仅 1~6，无 LayerNorm 时数值范围不一致（bg_z 与 sin/cos 量纲差距），加 LN 帮助优化。
- **静态特征注入 = LSTM 输出端 concat**：确认采纳 origin §4.1 推荐方案；不广播到序列维度。
- **MedianPruner 启用条件**：`n_warmup_steps = 5 epochs`、`n_startup_trials = 5`，避免早期 trial 被误裁。
- **MLflow tracking_uri**：本地 file store `./mlruns/`；不部署 server，保持仓库可移植。
- **SHAP explainer 选 `GradientExplainer`** 而非 `DeepExplainer`：xlstm 内有 conv1d 与自定义 cuda kernel，`DeepExplainer` 的 hook 注册路径在 sLSTM 上可能失败；`GradientExplainer` 走 autograd，更鲁棒。
- **环境冲突修复**：在 `requirements.txt` 中显式 pin `numpy<2.3` 并升级 numba 到 `>=0.61.2`（已具备），由 U1 验收时跑一次 `import shap` smoke test。
- **设备调度**：训练固定 GPU（`torch.cuda.is_available()` 必须 True）；`predict.py` 默认 CPU 单条推理（满足 origin §1 < 50 ms 目标）。
- **patient_id 仅作分组键**：构造窗口 + 切分时使用，不进入模型；显式从模型输入路径剔除以避免泄漏。

---

## Open Questions

### Resolved During Planning

- **xLSTM API 形态**：经本地 inspect 确认 `(B,T,D) → (B,T,D)`，需手动取最后步 + Linear 投影。
- **shap × numba × numpy 冲突**：通过 `numpy<2.3` pin 解决；备选用 captum 的 IntegratedGradients。
- **窗口物化 vs 在线构造**：物化（22M 点构造耗时 1 次，复用每个 epoch）。
- **batch 内是否允许混合患者**：允许（不破坏 patient-time 切分边界即可）。

### Deferred to Implementation

- **xLSTM 在 T=4 下 mLSTM vs sLSTM 比例的实测最优值**：交给 Optuna 搜索决定。
- **early stopping 的 patience 具体取值**：先用 10 epochs，由首版训练曲线决定是否调。
- **22M 点过滤后实际剩多少有效窗口**：U2 跑完后才知道；若过低需调宽 `TIME_TOLERANCE_MIN`。
- **GPU 显存容量是否够 batch_size=1024 + embedding_dim=256**：U5 调试时实测，必要时 Optuna 减小 batch_size 上限。
- **MLflow 多 trial 并发写 file store 是否冲突**：Optuna 默认串行；若改并行需切 SQLite backend，本版不并行。

---

## Output Structure

```
T1DiabetesGranada_Prediction/
├── data/                                # 原始 csv（已有 symlink）
│   ├── Glucose_measurements.csv
│   ├── Patient_info.csv
│   └── processed/                       # U2 产出
│       ├── train_seq.npy   train_static.npy   train_target.npy
│       ├── val_seq.npy     val_static.npy     val_target.npy
│       ├── test_seq.npy    test_static.npy    test_target.npy
│       ├── scaler.pkl                   # bg/age 的 (μ, σ)
│       └── meta.json                    # 样本数、特征列名、配置 hash
├── t1d_granada/                         # python 包（mirror brist1d/）
│   ├── __init__.py
│   ├── params.py                        # 集中所有可调参数 + flag
│   ├── utils.py                         # 路径、timer、make_dir、seed
│   ├── window_builder.py                # 窗口构造 + 时间对齐过滤
│   ├── feature_engineer.py              # bg_diff / time_sin_cos / day_sin_cos
│   ├── dataset.py                       # PyTorch Dataset (memmap)
│   ├── model.py                         # xLSTMRegressor
│   ├── trainer.py                       # 训练循环 + MLflow + early stopping
│   └── shap_analysis.py                 # SHAP 计算 + 4 类图
├── prepare_data.py                      # U2 入口
├── train.py                             # U6 入口（Optuna + 最终 fit）
├── predict.py                           # U7 入口
├── shap_analyze.py                      # U8 入口
├── settings.json                        # 路径配置
├── requirements.txt                     # 含 numpy<2.3 pin
├── REQUIREMENTS.md                      # （已存在）
├── model/                               # U6 输出
│   └── xlstm_best.pt
├── mlruns/                              # MLflow file store
├── hyperparameter_tuning/               # U6 输出（Optuna study pickle）
│   └── study.pkl
└── reports/shap/                        # U8 输出
    ├── feature_importance.png
    ├── timestep_importance.png
    ├── time_feature_heatmap.png
    ├── force_plots/sample_*.png
    └── shap_summary.csv
```

---

## High-Level Technical Design

> *直观展示数据 → 模型 → 输出的形状变化，方便审稿人验证设计方向；不是实现规范。*

```
prepare_data.py
─────────────────────────────────────────────────────────────
Glucose_measurements.csv       # 22.67M 行
   ├ groupby Patient_ID 排序
   ├ window_builder: 滑动窗口 stride=1
   │     · 严格 15-min 对齐 (±2 min 容差)
   │     · 6 个连续点 (T=4 输入 + 2 步到 +30min)
   │     · 任意 NaN → 丢弃
   ├ join Patient_info → Sex, Age (Age = year(date) - Birth_year)
   ├ 切分: 每患者前 80/10/10 (chronological)
   ├ scaler.fit(train.bg)  →  (μ, σ)
   └ feature_engineer:  apply scaler + flag-controlled 派生特征
        seq:    (N, 4, D_seq=6)  float32
        static: (N, 2)            float32
        target: (N,)              float32
─────────────────────────────────────────────────────────────
                    ↓ npy memmap


train.py / model forward
─────────────────────────────────────────────────────────────
seq_input  (B, 4, 6)
   ↓ Linear(6→E) + LayerNorm
   ↓ xLSTMBlockStack(num_blocks, mlstm/sLSTM mix)
   ↓ output  (B, 4, E)
   ↓ take last step → (B, E)
   ↓
   ├── concat ←─── static_input (B, 12)  # Sex + Age + 10 个 2h/6h 滚动统计
   ↓
   MLP(E+12 → H → 1)
   ↓
   ŷ_z  (B,)  ←─ MSE vs target_z (训练)
   inverse_scale → mg/dL  ←─ RMSE/MAE/R² 报告
─────────────────────────────────────────────────────────────


train.py / Optuna 调参
─────────────────────────────────────────────────────────────
study = TPESampler + MedianPruner
for trial in range(N_TRIALS):
    sample 8 hyperparams
    with mlflow.start_run(nested=trial):
        train_loop(trainer.py)
        mlflow.log_metrics(epoch loss/val RMSE …)
        report intermediate val_rmse → Optuna pruner
    return best_val_rmse_denorm
final_run: 用 best_params 在 train+val 上训练 → save model
─────────────────────────────────────────────────────────────


shap_analyze.py
─────────────────────────────────────────────────────────────
load model + scaler
bg_set ← random 256 from train
fg_set ← random 1024 from test
explainer = GradientExplainer(model, [bg_seq, bg_static])
shap_vals = explainer.shap_values([fg_seq, fg_static])
   ↓
4 plots + shap_summary.csv → reports/shap/
─────────────────────────────────────────────────────────────
```

---

## Implementation Units

### U1. 项目脚手架与依赖修复

**Goal:** 建立目录结构、settings.json、requirements.txt、`t1d_granada/` 包骨架；先解决 numpy/numba/shap 兼容问题，确保后续单元有可用的开发环境。

**Requirements:** R11

**Dependencies:** None

**Files:**
- Create: `T1DiabetesGranada_Prediction/settings.json`
- Create: `T1DiabetesGranada_Prediction/requirements.txt`
- Create: `T1DiabetesGranada_Prediction/t1d_granada/__init__.py`
- Create: `T1DiabetesGranada_Prediction/t1d_granada/params.py`
- Create: `T1DiabetesGranada_Prediction/t1d_granada/utils.py`
- Create: `T1DiabetesGranada_Prediction/.gitignore`（忽略 `data/processed/`、`mlruns/`、`model/`、`reports/`）
- Test: `T1DiabetesGranada_Prediction/tests/test_smoke.py`

**Approach:**
- `params.py` 集中所有 flag / 阈值 / 搜索范围（mirror `brist1d/params.py` 风格）。包含：`WINDOW_SIZE=8`、`FORECAST_STEPS=2`、`STRIDE=1`、`TIME_TOLERANCE_MIN=2`、特征 flag、训练超参、Optuna 搜索空间字典、SHAP 采样数。
- `utils.py` 复用 BrisT1D 的 `timer` / `make_dir`，新增 `set_seed(seed)`、`load_settings()` 工具。
- `requirements.txt` 显式：`numpy<2.3`、`numba>=0.61.2`、`shap>=0.51`、`mlflow>=3.10`、`optuna>=4.6`、`torch>=2.10`、`xlstm==2.0.5`、`pandas>=2.2`、`scikit-learn>=1.5`。
- 在 `tests/test_smoke.py` 写一个 import-only 测试，验证所有关键库能干净导入。

**Patterns to follow:**
- [brist1d/utils.py](../../../brist1d/utils.py)（timer、make_dir、settings 加载）
- [brist1d/params.py](../../../brist1d/params.py)（参数风格与注释）

**Test scenarios:**
- Happy path: 在 conda env `py312_cu121` 内 `python -c "import shap; import xlstm; import mlflow; import optuna; import numba; import numpy; print('OK')"` 无错。
- Happy path: `python -c "from t1d_granada.params import *; print(WINDOW_SIZE)"` 输出 8。
- Edge case: `from t1d_granada.utils import set_seed; set_seed(42)` 后 `torch.rand(1)` 与 numpy 同 seed 重复执行结果一致。

**Verification:**
- `pytest T1DiabetesGranada_Prediction/tests/test_smoke.py` 通过。
- `pip install -r requirements.txt` 在 py312_cu121 环境内执行成功（或仅打印 "already satisfied"）。

---

### U2. 数据准备脚本（窗口构造 + 切分 + scaler 拟合）

**Goal:** 从 `Glucose_measurements.csv` + `Patient_info.csv` 构造 train/val/test 三元组并物化为 npy，附带 scaler.pkl + meta.json。是后续所有单元的数据依赖。本单元同时计算 2h / 6h 多窗口血糖统计作为静态特征。

**Requirements:** R1, R2, R3, R4, R4b

**Dependencies:** U1

**Files:**
- Create: `T1DiabetesGranada_Prediction/t1d_granada/window_builder.py`
- Create: `T1DiabetesGranada_Prediction/t1d_granada/feature_engineer.py`
- Create: `T1DiabetesGranada_Prediction/t1d_granada/rolling_stats.py`  *(新增：多窗口因果统计)*
- Create: `T1DiabetesGranada_Prediction/prepare_data.py`
- Test: `T1DiabetesGranada_Prediction/tests/test_window_builder.py`
- Test: `T1DiabetesGranada_Prediction/tests/test_feature_engineer.py`
- Test: `T1DiabetesGranada_Prediction/tests/test_rolling_stats.py`

**Approach:**
- `window_builder.build_windows(df, patient_info)`：按 `Patient_ID` 排序 → 计算每行与上一行的时间差 → 用 `numpy.lib.stride_tricks.sliding_window_view` 取连续 6 点窗口 → 检查全部 5 个间隔在 [13, 17] min（容差 ±2）→ 合法窗口前 4 步为输入序列、第 6 步为标签（跳过第 5 步=+15min）。
- 切分：每个 patient 的合法窗口按时间排序后前 80%/中 10%/后 10%。
- `feature_engineer.add_derived(seq_raw, timestamps)` 按 flag 加 `bg_diff`（首步填 0）、`time_of_day_sin/cos`（每步独立）、`day_of_week_sin/cos`（每步独立）。flag 全 OFF 时 D_seq=1，全 ON 时 D_seq=6。
- **`rolling_stats.compute_static_stats(df_per_patient, window_end_timestamps)`**：对每位患者按时间设 `DatetimeIndex`，用 `df.bg.rolling('2h', closed='left').agg(...)` / `'6h'` 一次性算出 mean、std、TIR_70_180、p5、p95（**`closed='left'` 是因果性的硬性保证；不能省**）。TIR 用 `(70 ≤ x ≤ 180).mean()` 的自定义 aggregator。结果对齐到每个窗口的 `window_end_timestamp`，得到 (N, 10) 的静态扩展矩阵。
- 缺失边界：每个窗口需 `min_periods = window_size_in_15min_points // 2`（2h: 4、6h: 12）；任一尺度不达 → 该样本丢弃。2h 阈值非常宽松（基本只过滤"刚开机第一小时"），6h 偶有不达。
- 静态特征：`Sex_M = 1 if M else 0`；`Age = (window_end_date.year - Birth_year)`；与上面的 10 列 rolling stats 拼接得 D_static=12。
- scaler 仅在训练集上拟合：`bg_mean`、`bg_std`、`age_mean`、`age_std`，**以及 10 列 rolling stats 各自的 (μ, σ)**（TIR 是比例不需缩放，但仍按 z-score 处理保持统一接口）。
- `prepare_data.py` 串起：读 csv → `build_windows` → `compute_static_stats` → 切分 → `fit_scaler(train)` → `apply_scaler` 到三段 → `add_derived` → 写 npy + scaler.pkl + meta.json。
- 用 `tqdm` 显示进度（22M 点处理约 8–12 min 一次性；rolling 步骤的 2h+6h 两窗口比之前的三窗口轻很多）。

**Patterns to follow:**
- [brist1d/tabular_transformers.py](../../../brist1d/tabular_transformers.py)（特征工程函数风格）
- pandas `df.rolling('6h', closed='left').agg({'bg': [mean, std, ...]})` 标准用法

**Test scenarios:**
- Happy path: 用合成的小 DataFrame（10 个连续 15-min 点）→ `build_windows` 返回 (5 个窗口, 4 步输入, target=第 6 个点)。
- Edge case: 中间插入一个 35-min 间隔 → 跨该间隔的所有窗口被丢弃，前后段独立产生窗口。
- Edge case: 序列内含 NaN → 该窗口被丢弃。
- Edge case: 容差边界 13 min 与 17 min 都通过；12 min / 18 min 被拒绝。
- Happy path (feature_engineer): flag 全 OFF → (N, 4, 1)；flag 全 ON → (N, 4, 6)。
- Edge case (feature_engineer): `bg_diff` 首步 = 0；`time_of_day_sin/cos` 在 00:00 与 24:00 取值一致。
- **Happy path (rolling_stats):** 构造 1 位患者 1 天连续数据，对最后一个 timestamp 计算 2h / 6h stats → 与手算 mean/std/TIR/p5/p95 数值匹配（容差 1e-6）。
- **Edge case (rolling_stats — 因果性):** 在 window_end 之后的位置注入异常值（如 9999），rolling stats 不应包含此值（验证 `closed='left'` 严格生效）。
- **Edge case (rolling_stats):** 患者前 2h 窗口可用点 < 4 → 该样本丢弃；前 6h 不足 12 点 → 同样丢弃。
- **Edge case (rolling_stats — TIR 边界):** bg=70 和 bg=180 都计入 TIR；bg=69 / bg=181 不计入。
- Integration: 用 1000 行真实 csv 子集跑 `prepare_data.py`，验证 static.shape == (N, 12)、scaler.pkl 含 12 + 1 = 13 个 (μ, σ) 条目（12 列 static + 1 列 bg）。

**Verification:**
- `python prepare_data.py` 完整跑通，控制台打印过滤前后样本数（含 rolling-stats 阶段过滤多少样本）、各段大小、scaler 参数。
- `data/processed/{train,val,test}_{seq,static,target}.npy` 全部存在；形状满足 `seq.shape == (N, 4, D_seq)`、`static.shape == (N, 12)`、`target.shape == (N,)`。
- meta.json 中样本量 ≥ 1M（2h+6h 过滤温和，预估接近原始时间对齐过滤后水平）。

---

### U3. PyTorch Dataset / DataLoader 工厂

**Goal:** 用 memmap 高效加载 U2 物化的 npy；提供 `make_loaders()` 一次性返回三段 DataLoader；同时支持反归一化辅助函数。

**Requirements:** R1, R2

**Dependencies:** U1, U2

**Files:**
- Create: `T1DiabetesGranada_Prediction/t1d_granada/dataset.py`
- Test: `T1DiabetesGranada_Prediction/tests/test_dataset.py`

**Approach:**
- `class GlucoseWindowDataset(Dataset)`：构造函数接收 npy 路径，用 `np.load(..., mmap_mode='r')` 三连。`__getitem__` 返回 `(seq_t, static_t, target_t)` 三元组（torch.float32 tensor）；不在 worker 内做 augmentation。
- `make_loaders(processed_dir, batch_size, num_workers)` 返回 `train_loader, val_loader, test_loader`；train shuffle=True；val/test shuffle=False。`pin_memory=True`，`persistent_workers=True`。
- 提供模块级辅助函数 `denormalize_bg(z, scaler)` / `normalize_bg(x, scaler)`，全局 import 复用。

**Patterns to follow:**
- 标准 PyTorch Dataset 模板；不引入第三方 dataloader 框架。

**Test scenarios:**
- Happy path: 用 U2 单测中创建的 mini npy 构造 Dataset，`len()` 与 npy 第 0 维一致。
- Happy path: `__getitem__(0)` 返回三个 torch.float32 tensor，形状分别为 `(4, D_seq)`、`(17,)`、`()`。
- Happy path: `make_loaders` 产出三 loader，迭代一个 batch 形状为 `(B, 4, D_seq)`、`(B, 12)`、`(B,)`。
- Edge case: `denormalize_bg(normalize_bg(x, s), s) ≈ x`（数值容差 < 1e-5）。

**Verification:**
- 单测通过；手工跑 1 个 epoch 完整迭代 train_loader 不报错且 GPU pin 正常。

---

### U4. xLSTM 回归模型

**Goal:** 实现 `xLSTMRegressor`：输入投影 → xLSTMBlockStack → 取最后步 → concat(static) → MLP → 标量。所有可调超参通过构造函数注入，便于 Optuna。

**Requirements:** R5

**Dependencies:** U1

**Files:**
- Create: `T1DiabetesGranada_Prediction/t1d_granada/model.py`
- Test: `T1DiabetesGranada_Prediction/tests/test_model.py`

**Approach:**
- 构造函数签名：`(d_seq, d_static, embedding_dim, num_blocks, mlstm_ratio, mlp_hidden, dropout, conv_kernel_size, context_length=4)`；
- `mlstm_ratio` ∈ {0, 0.5, 1.0} → 计算 `slstm_at` 列表（例如 `num_blocks=4, ratio=0.5` → `slstm_at=[1, 3]`，交错布置）。
- 输入投影 = `Linear(d_seq, embedding_dim) → LayerNorm`。
- 主干 = `xLSTMBlockStack(xLSTMBlockStackConfig(...))`，传入对应 `mLSTMBlockConfig` / `sLSTMBlockConfig`。
- 取 `out[:, -1, :]` 后与 static 拼接 (`embedding_dim + d_static`)，过 `MLP = Linear → GELU → Dropout → Linear → 标量`。
- forward 签名：`def forward(self, seq, static): -> (B,)`，注意 squeeze 最后维。

**Patterns to follow:**
- xlstm 2.0.5 实测 API（见 Context & Research 节）。

**Test scenarios:**
- Happy path: 构造默认参数模型，输入 `(2, 4, 6)` + `(2, 12)` → 输出 shape `(2,)`、dtype float32、不含 NaN。
- Happy path: backward 跑通；`loss = output.sum().backward()` 后所有 `nn.Parameter` 的 grad 非 None 且非全 0。
- Edge case: `mlstm_ratio=0`（纯 sLSTM）、`=1`（纯 mLSTM）、`=0.5`（混合）三种均可构造 + forward。
- Edge case: `num_blocks=2` 与 `num_blocks=6` 均可构造（堆栈深度边界）。
- Edge case: `conv_kernel_size=2/3/4` 三种值在 T=4 下均可 forward（4 时刚好等于 context_length，可能需要 padding 设置）。

**Verification:**
- 单测全部通过；手工在 GPU 上 forward 一个 batch 用时 < 50 ms (B=512)。

---

### U5. 训练循环 + MLflow 记录 + Early Stopping

**Goal:** 单一 `train_one_config(model, loaders, params, mlflow_run)` 函数：跑 max_epochs，每 epoch 记录 train_loss / val_RMSE / val_MAE / val_R²（反归一化），patience 触发后早停；返回 best val_RMSE 与对应 state_dict。这是 trial 内核与最终 fit 共用的执行单元。

**Requirements:** R6, R8, R9

**Dependencies:** U3, U4

**Files:**
- Create: `T1DiabetesGranada_Prediction/t1d_granada/trainer.py`
- Test: `T1DiabetesGranada_Prediction/tests/test_trainer.py`

**Approach:**
- `train_one_config` 接收：模型、train/val loader、scaler、超参 dict、`max_epochs`、`patience`、`mlflow_run` 句柄、`optuna_trial` 可选（若给则每 epoch 调 `trial.report(val_rmse, step=epoch)` 并检查 `should_prune`）。
- 损失：`nn.MSELoss()` on z-score 标签。
- 优化器：`AdamW(lr, weight_decay)`；scheduler：`CosineAnnealingLR(T_max=max_epochs)` + warmup（前 5% epochs LR 线性升温，可选）。
- 每 epoch 末做 val 评估：把所有 batch 预测拼接 → `denormalize` → 计算 RMSE / MAE / R²；用 `mlflow.log_metric` 写入。
- Early stopping：维护 `best_val_rmse, best_state_dict, no_improve_count`；连续 patience 次未提升即停。
- 训练 loss 也每 epoch 写一次（取 epoch 平均）。
- 退出时返回 `(best_val_rmse, best_state_dict, last_epoch)`。

**Patterns to follow:**
- 不引入 PyTorch Lightning；保持纯 PyTorch 训练循环风格，与 BrisT1D 项目"无框架"取向一致。

**Test scenarios:**
- Happy path: 用 U4 测试中的 mini 模型 + U3 mini loader 跑 3 epoch，返回 `best_val_rmse` 为正数；MLflow run 内含 6 个指标键（train_loss、val_rmse、val_mae、val_r2、lr、epoch_time）。
- Happy path: patience=2 下，故意构造一个让 val_rmse 始终上升的场景，应在 epoch 3 就停止。
- Error path: 传入空 train_loader → 抛 ValueError。
- Integration: 与 Optuna trial 联用，`trial.should_prune` 返回 True 时函数应抛 `optuna.TrialPruned`。
- Edge case: 训练完所有 epoch 都没改善 → 仍返回首个 epoch 的 state_dict 而非空。

**Verification:**
- 单测通过；手工跑 1 个 fold 5 epochs，确认 MLflow UI 中曲线连续无断点、指标范围合理（RMSE 在 mg/dL 量级 10–50）。

---

### U6. Optuna 调参 + 最终 fit (`train.py`)

**Goal:** 用 Optuna TPESampler + MedianPruner 跑 N_TRIALS 次搜参，每 trial 一个 MLflow nested run；找到 best_params 后在 train+val 合并集上做最终 fit，模型 + best_params 持久化到 `model/`。

**Requirements:** R7, R8, R12

**Dependencies:** U5

**Files:**
- Create: `T1DiabetesGranada_Prediction/train.py`
- Test: `T1DiabetesGranada_Prediction/tests/test_train_smoke.py`

**Approach:**
- `train.py` 主入口：`mlflow.set_tracking_uri('./mlruns/')`，开 parent run "xlstm_search"。
- `objective(trial)` 函数：从 `params.OPTUNA_SEARCH_SPACE` 字典读出搜索空间 → `trial.suggest_*` 采样 → 构造模型 → `make_loaders(... batch_size=trial)` → `with mlflow.start_run(nested=True)` → `train_one_config(...)` → 返回 best val RMSE。
- `study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=SEED), pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5))`。注：`n_warmup_steps` 计的是 `trial.report(value, step=epoch)` 的 step 索引，与本计划的 epoch 数一致；前 5 个 trial 因 `n_startup_trials` 不会被裁。
- `study.optimize(objective, n_trials=N_TRIALS)`。
- pickle `study` → `hyperparameter_tuning/study.pkl`。
- 最终 fit：`best_params = study.best_params`；用 train+val 合并 loader（保持时间序内部排序）训练相同 epoch 数 × 1.2（参考 BrisT1D 的 1.5x 经验，但因 NN 不同改为 1.2）；`mlflow.pytorch.log_model(model, "model")`；torch.save 到 `model/xlstm_best.pt`。
- test 集评估：加载 best 模型 → 跑 test_loader → 报 RMSE/MAE/R²；用 `mlflow.log_metrics(prefix='test_')`。

**Patterns to follow:**
- [train.py](../../../train.py)（Optuna 流程、`run_study` + `fit_model` + `refit_model` 三函数拆分；本计划合并为单文件三函数）

**Test scenarios:**
- Happy path: `python train.py` 在 mini 数据集（U2 测试构造的 1000 行子集）上跑 3 trials × 3 epochs 完整通过。
- Happy path: MLflow `./mlruns/` 内出现 1 parent + 3 nested + 1 final = 5 个 run。
- Error path: `data/processed/` 不存在 → 抛清晰错误信息提示先跑 `prepare_data.py`。
- Integration: best_params.pkl 中的 keys 与 OPTUNA_SEARCH_SPACE 完全一致。
- Verification: test RMSE 写入最终 run 的 metrics（key 名以 `test_` 前缀）。

**Verification:**
- 单测通过；手工跑全量 25 trials × 实际 epoch 数 ≥ 1 次完整训练，确认 `model/xlstm_best.pt` 可被 U7 加载。

---

### U7. 推理脚本 (`predict.py`)

**Goal:** 加载 `model/xlstm_best.pt` + `scaler.pkl`；接受 csv 或单条参数化输入，输出 30 min 后的血糖（mg/dL）；CPU 单条推理 < 50 ms。

**Requirements:** R3, R4, R12  *(R3/R4 因复用 U2 `feature_engineer.add_derived` 与 U3 归一化逻辑，特征构造一致性必须在推理路径中得到验证)*

**Dependencies:** U6

**Files:**
- Create: `T1DiabetesGranada_Prediction/predict.py`
- Test: `T1DiabetesGranada_Prediction/tests/test_predict.py`

**Approach:**
- **API 形态调整**：因为 D_static 包含 2h/6h 滚动统计，`predict.py` 不再支持仅传 4 个当前 bg 的"轻量调用"；必须随附该患者过去 ≥6 小时的 CGM 历史。命令行参数：
  - `--history csv_path`（**必填**）：csv 列 `Patient_ID, timestamp, bg`，至少包含从 `last_ts - 6h` 到 `last_ts` 的所有 CGM 测量；多患者则每患者独立处理。
  - `--input csv_path`（批量推理）：csv 列 `Patient_ID, last_timestamp, Sex, Birth_year`；每行的 4 步 bg 从 `--history` 中按 `last_timestamp` 取最近 4 个对齐点；输出 `Patient_ID, last_timestamp, predicted_bg_30min`。
  - `--patient_id LIB123 --last_ts "2026-05-09 12:00:00" --sex M --birth_year 1985`（单条）：从 `--history` 取该患者数据并切出 4 步窗口与历史；stdout 输出预测值。
  - `--model path` / `--scaler path`：覆写默认路径。
  - `--device cpu|cuda`：默认 cpu。
- 内部流程：从 history csv 取每个 (patient, last_ts) 对应的 4 步 → `feature_engineer.add_derived` → 用 history 计算 2h/6h 静态统计（**严格 `closed='left'` 一致于训练**）→ `apply_scaler` → 模型 forward → 反归一化 → 输出。
- 复用 U2 的 `rolling_stats.compute_static_stats`、`feature_engineer.add_derived` 与 U3 的 `normalize_bg`，**关键**：训练 / 推理用同一函数路径保证特征一致性。
- 加载模型：`torch.load(... map_location=device)` + `model.eval()`。
- 单条耗时打印（`time.perf_counter`）；目标 < 50 ms（不含 history I/O 与 rolling stats 计算）；含 stats 的端到端单条耗时新目标 < **80 ms**（CPU 上 6h 滚动统计 ≈ 30 ms，比 7d 轻很多）。

**Patterns to follow:**
- [predict.py](../../../predict.py)（argparse + `make_dir` + csv 输出风格）

**Test scenarios:**
- Happy path (单条): 构造一份 mock history csv（某 patient 1 天 15-min bg）+ `--patient_id ... --last_ts ... --sex M --birth_year 1985` → 返回一个数值（0 < x < 500），打印推理耗时。
- Happy path (批量): `--history h.csv --input batch.csv`，100 行 batch → 输出 csv 行数 100 + 1 表头，列名 `[Patient_ID, last_timestamp, predicted_bg_30min]`。
- Edge case: history 不足 6h（如冷启动只有 1h 数据）→ 报错信息明确指出该样本因 `min_periods` 不足被跳过；2h 窗口够但 6h 不够时同样跳过。
- Edge case: history 中存在 NaN / 短缺口 → 仍能产出预测，只要 `min_periods` 阈值满足；统计量自动忽略 NaN。
- Edge case: `--device cuda` 在无 GPU 机器上 → 自动 fallback 到 cpu 并 warn。
- Error path: 模型文件不存在 → FileNotFoundError 信息含完整路径。
- **Critical (因果性):** 在 history 中 `last_ts` 之后位置注入异常值 → 预测结果与不注入时**完全相同**（验证 `closed='left'` 在推理路径生效）。
- Performance: CPU 单条 100 次平均推理（不含 stats）< 50 ms；端到端（含 stats）< 80 ms。

**Verification:**
- 单测通过；手工跑 `--history` + 单条 + 批量两种调用方式各成功一次。
- 用训练集中的某个 (patient, last_ts) 对推理，结果与训练阶段同一样本的预测值数值一致（容差 1e-4，验证 inference path 与 train path 等价）。

---

### U8. SHAP 可解释性分析 (`shap_analyze.py`)

**Goal:** 计算 SHAP，输出 4 类图 + shap_summary.csv 到 `reports/shap/`，并把 artifact 上传至 final MLflow run。

**Requirements:** R10

**Dependencies:** U6

**Files:**
- Create: `T1DiabetesGranada_Prediction/t1d_granada/shap_analysis.py`
- Create: `T1DiabetesGranada_Prediction/shap_analyze.py`
- Test: `T1DiabetesGranada_Prediction/tests/test_shap_analysis.py`

**Approach:**
- `shap_analysis.compute_shap(model, bg_set, fg_set, device)`：使用 `shap.GradientExplainer(model, [bg_seq, bg_static])`；返回两个张量 `shap_seq: (N, T, D_seq)`、`shap_static: (N, D_static)`。
- 模型在 SHAP 调用前必须 `eval()`，且 `shap.GradientExplainer` 不接受 `nn.Module` 的多输入版本——需写一个适配 wrapper：把 seq 和 static 接收为单一 tensor 拼接（`(B, T*D_seq + D_static)`）后内部 reshape 再调原模型。或者：调用 `explainer = shap.GradientExplainer(wrapper_fn, list_of_arrays)`。**实现时需小心**：选定方案后写明在代码注释里。
- 4 类图：
  1. `feature_importance.png`：把 (T, D_seq) SHAP 在 T 维度求 mean(|·|)，与 (D_static) SHAP 拼成长 D_seq + D_static 的向量；条形图按重要性排序。
  2. `timestep_importance.png`：把 (T, D_seq) SHAP 在 D_seq 维求 mean(|·|)，得长 T=4 向量；条形图。
  3. `time_feature_heatmap.png`：(T, D_seq) SHAP 跨 N 求 mean(|·|)，imshow 热力图，xticks=feature names、yticks=t-3/t-2/t-1/t-0。
  4. `force_plots/sample_*.png`：从 fg_set 选 5 个样本（含 1 个最高 prediction、1 个最低、3 个中位）画 `shap.plots.force` 或 `shap.waterfall_plot`。
- `shap_summary.csv`：列 `feature_name, mean_abs_shap, rank`。
- `shap_analyze.py` 入口：argparse 接收 `--model`、`--processed_dir`、`--output reports/shap/`、`--n_bg 256`、`--n_fg 1024`、`--seed`；调用 compute_shap + 4 类绘图函数；调用 `mlflow.log_artifact` 把 reports/shap/ 推到 MLflow（若有 active run；否则只本地保存）。
- 用 matplotlib，不依赖 plotly；保存 `dpi=120, bbox_inches='tight'`。

**Patterns to follow:**
- 无项目内既有 SHAP 代码；遵循 SHAP 官方 PyTorch GradientExplainer 文档。

**Test scenarios:**
- Happy path: 用 U6 单测中的 mini 模型 + 32 个 bg 样本 + 64 个 fg 样本计算 SHAP，返回值形状 (64, 4, D_seq) 与 (64, 2)，无 NaN。
- Happy path: 调 `plot_feature_importance(...)` 后产出 png 文件存在且 file size > 1KB。
- Happy path: `shap_summary.csv` 行数 = D_seq + D_static = 8（默认 flag 全 ON）；列名为 `feature_name, mean_abs_shap, rank`；rank 单调。
- Edge case: 模型为纯 mLSTM（含自定义 cuda kernel）时 GradientExplainer 仍工作（回退到 `device='cpu'` 计算 SHAP，因部分 cuda 内核不可微）。
- Edge case: D_seq=1（flag 全 OFF 配置）时所有 4 类图仍能产出。
- Integration: `python shap_analyze.py --model model/xlstm_best.pt --output reports/shap/` 端到端跑通，全部 4 张 png + 1 csv 落盘。

**Verification:**
- 单测通过；手工查看 `reports/shap/feature_importance.png` 可读、特征名能对上、最重要特征是 `bg_z`（业务直觉验证）。

---

## System-Wide Impact

- **不修改既有 BrisT1D 代码**：本计划完全在 `T1DiabetesGranada_Prediction/` 子目录新增；不 import `brist1d/` 包。这是 design decision，避免两个项目互相耦合。
- **数据来源边界**：所有 csv 通过 `data/` 下的 symlink 读，不复制大文件入仓。`data/processed/` 与 `mlruns/` 加 `.gitignore`。
- **进程边界**：`prepare_data.py` 一次性物化、`train.py` 多次调试调参、`predict.py` 推理、`shap_analyze.py` 解释——四脚本互不依赖运行时，仅通过磁盘产物耦合（npy / pkl / pt）。这与 BrisT1D 项目的"3 脚本独立"原则一致。
- **GPU 资源占用**：训练默认占满 1 张卡；SHAP 与 predict 默认 CPU 以避免抢资源；如需并行 Optuna trial 必须改 SQLite backend，本版不并行。
- **不变的原项目 API**：所有 BrisT1D 项目入口（`prepare_data.py` / `train.py` / `predict.py` / `end_to_end.py`）保持完全不变；CLAUDE.md / REQUIREMENTS.md 等文档亦不修改。

---

## Risks & Dependencies

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `shap` × `numba` × `numpy 2.3` 冲突在不同机器上重现度不一 | High | Medium | U1 显式 pin `numpy<2.3`；CI/本地都跑 import smoke test；备选：用 captum IntegratedGradients 替代 |
| 严格 ±2 min 容差过滤后样本量低于 1M | Medium | Medium | U2 实测过滤率；过低则按 `params.TIME_TOLERANCE_MIN` 调宽到 ±5 min；记录在 meta.json |
| 6h rolling 的 `min_periods=12` 在患者入组首小时仍可能不达 | Low | Low | 实测后若过滤超 5%，可考虑放宽到 `min_periods=8`；记录在 meta.json |
| rolling 统计的因果性破坏（`closed='left'` 漏掉、或 inference 路径误用 `closed='right'`） | Medium | High | U2 + U7 单测各加一条注入未来异常值的因果性验证；CI 中必跑 |
| `predict.py` 接口扩展为需要 ≥6h history，旧 demo 调用失效 | Medium | Low | 已记入 origin §7.1；写一段 README 说明新接口；保留接口以 csv 为主路径。Libre 设备本机存 8h 历史，6h 在常规部署场景下不构成阻碍 |
| xLSTM 在 T=4 短序列上不收敛或不如简单 MLP | Medium | High | 先跑 `mlstm_ratio=1` + `num_blocks=2` 的最小配置 baseline，再让 Optuna 探索 |
| MLflow file store 在 25 trials 后膨胀至 GB 级 | Low | Low | `.gitignore` 排除 `mlruns/`；只保留最优 trial artifact，nested run 不存 model |
| GPU OOM at `batch_size=1024 * embedding_dim=256` | Medium | Low | Optuna 搜索空间设上限；trial 内 try/except OOM → `raise TrialPruned` |
| SHAP `GradientExplainer` 在含 cuda-only kernel 的 sLSTM 上失败 | Medium | Medium | U8 实现时 fallback 到 CPU；若仍失败用 captum 替代 |
| 22M 行 csv 一次读入内存 (~3 GB) | Low | Low | 已实测 200k 行用 64 MB；全量 ≈ 3 GB，py312_cu121 机 32 GB RAM 充足；必要时改 chunked read |
| `Birth_year` 偶有 NaN 或异常（如 1900） | Low | Low | U2 加 sanity 过滤 (Birth_year ∈ [1900, 2020])，否则丢弃整个患者；warn 并写入 meta.json |
| 不同患者血糖基线差异大、全局 z-score 损失个体信号 | Medium | Medium | 已记入 origin §9 假设；首版完成后做 per-patient 归一化消融实验 |

---

## Documentation / Operational Notes

- **本计划完成后必更新**：仓库根目录 [CLAUDE.md](../../../CLAUDE.md) 应增加一节"T1DiabetesGranada_Prediction 子项目"介绍；指向 `T1DiabetesGranada_Prediction/REQUIREMENTS.md` 与本计划。
- **学习沉淀**：U1 的环境冲突修复、U4 的 xLSTM API 实测、U8 的 SHAP wrapper 适配——三条值得在交付后写成 `docs/solutions/2026-05-XX-*-learning.md`。
- **MLflow UI 启动方式**：`mlflow ui --backend-store-uri ./T1DiabetesGranada_Prediction/mlruns/ --port 5000` 在浏览器查看曲线（在交付的 README 里写一句即可）。
- **运行入口顺序**：U2 → U6 → U7 / U8 任选；U3-U5 不直接由用户运行。

---

## Sources & References

- **Origin document:** [T1DiabetesGranada_Prediction/REQUIREMENTS.md](../../REQUIREMENTS.md)
- **结构模板（不直接 import）:** [prepare_data.py](../../../prepare_data.py), [train.py](../../../train.py), [predict.py](../../../predict.py), [brist1d/](../../brist1d/)
- **CLAUDE.md (仓库根):** [CLAUDE.md](../../../CLAUDE.md)
- **xlstm 2.0.5:** 已通过本地 inspect 验证 `xLSTMBlockStack`、`xLSTMBlockStackConfig`、`mLSTMLayerConfig`、`sLSTMLayerConfig`、`FeedForwardConfig` 字段
- **数据集论文:** Carrillo-Larco et al. 2023 _Sci Data_, [T1DiabetesGranada](https://www.nature.com/articles/s41597-023-02737-4)
