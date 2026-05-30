# T1DiabetesGranada · 30-min 血糖预测 (xLSTM) — 需求文档

**Status**: ready for planning · **Created**: 2026-05-09 · **Owner**: jeff zhou
**Path**: `T1DiabetesGranada_Prediction/`

## 1. 目标

在 [T1DiabetesGranada](https://www.nature.com/articles/s41597-023-02737-4) 数据集上，训练一个 **xLSTM** 模型，输入过去 2 小时（8 个 15-min 间隔的 CGM 血糖采样）+ 静态患者画像，输出 30 分钟后的血糖值（mg/dL，单标量）。

参考 BrisT1D 1st-place solution 的"窗口扩张 + 特征工程"思路，但因新数据更简单（单变量、固定 15-min 间隔、无胰岛素/碳水/活动多模态），方法显著简化。

### 成功标准

| 维度 | 度量 | 目标 |
|---|---|---|
| 准确度 | test set RMSE (mg/dL) | **< 18 mg/dL**（业界 30-min PH 单变量基准约 18-22）|
| 准确度 | test set MAE (mg/dL) | < 13 mg/dL |
| 解释力 | test set R² | > 0.85 |
| 可解释性 | SHAP 可视化 | 特征重要性条形图 + 时间步重要性 + 单样本 force plot |
| 可复现 | 单 seed 三次重跑 RMSE 标准差 | < 0.5 mg/dL |
| 工程 | `predict.py` 单条推理延迟 | < 50 ms (CPU) |

> 数值目标根据 30-min PH (Prediction Horizon) 文献中位水平设定，初期可放宽 ±10%；首版跑出 baseline 后再回顾。

## 2. 数据决策

### 2.1 使用的数据文件

| 文件 | 用途 |
|---|---|
| `data/Glucose_measurements.csv` | 主时序数据，22.67M 行，736 名患者，15-min CGM |
| `data/Patient_info.csv` | 仅取 `Sex` (M/F) 和由 `Birth_year` + 测量日期算出的 `Age`（年） |

### 2.2 显式不使用

- `Diagnostics.csv` —— 30% 患者无诊断条目，NaN ≠ "无疾病"，引入系统性偏差
- `Biochemical_parameters.csv` —— 与本任务时间尺度不匹配
- `Preprocessed_data*.npy` / `Resampled_data.npy` —— 预处理流程自定义，不复用第三方
- `Number_of_*` 等 `Patient_info` 中的元数据列（数据集统计量、易泄漏）

### 2.3 切分策略

**每患者按时间切**：每位患者按测量时间排序后，前 80% / 中 10% / 后 10% 切为 train / val / test。
- 736 患者 × 3 段 = 2208 块；同一患者三段连续不重叠。
- 不做 patient-level 切分（已与用户对齐：要求"对已知患者预测其未来"语义）。
- 单 fold，**不做嵌套 CV**（数据量足够）。

### 2.4 窗口构造

- 输入窗口长度 T = 8（2 小时）；预测点距窗口最右端 +30 min（=2 步之后）。
- 窗口内时间戳必须**严格 15-min 间隔，容差 ±2 min**；包括预测点本身（即 10 个连续点皆需对齐）。
- 不满足上述时间对齐 → 丢弃该窗口。
- 序列内任何 NaN → 丢弃该窗口（T=8 缺一即损 12.5%，mask 不划算）。
- **stride = 1**（每个有效点都作为窗口右端起点）。

预估有效样本数：约 5-10M（粗估 22M × 切分留存率 × 窗口对齐留存率），train/val/test 切分后训练集仍 > 4M，对 xLSTM 充足。

### 2.5 归一化

| 字段 | 方法 | 拟合范围 |
|---|---|---|
| `bg`（输入序列） | z-score | 训练集全局 μ/σ |
| `bg+30min`（标签） | 与输入共享同一 (μ, σ) | 同上 |
| `Age` | z-score | 训练集 μ/σ |
| `Sex` | M=1, F=0 | — |
| 派生 sin/cos 特征 | 已在 [-1,1]，不再缩放 | — |

> 不做 per-patient 归一化（用户已确认）：保留"高基线 vs 低基线"的临床信号。
> 训练集统计量保存为 `scaler.pkl`，predict.py 复用。

## 3. 特征工程

### 3.1 时序特征（每时间步）

| 特征 | 默认开关 | 说明 |
|---|---|---|
| `bg_z` | 必选 | z-score 后的血糖 |
| `bg_diff` | flag, 默认 ON | 与上一时刻的差分；首步填 0 |
| `time_of_day_sin/cos` | flag, 默认 ON | 当前时间步的钟表分钟数 → 周期 1440 min |
| `day_of_week_sin/cos` | flag, 默认 ON | 周一到周日 → 周期 7 |

flag 集中在 `params.py` 中：`USE_BG_DIFF=True`、`USE_TIME_OF_DAY=True`、`USE_DAY_OF_WEEK=True`。

序列特征维度 D_seq：
- 全开（默认）：D_seq = 1 + 1 + 2 + 2 = **6**
- 全关：D_seq = 1（baseline）

### 3.2 静态特征（每样本一组）

| 特征 | 来源 |
|---|---|
| `Sex_M` | Patient_info.Sex |
| `Age_z` | (Measurement_date.year − Birth_year) → z-score |

**临床多窗口血糖统计**（在窗口最右端 timestamp 处、严格因果计算）：

| 窗口 | 统计量 | 列名 |
|---|---|---|
| 2 小时 | mean, std, TIR_70_180, p5, p95 | `bg_2h_{stat}` × 5 |
| 6 小时 | 同上 | `bg_6h_{stat}` × 5 |

flag：`USE_ROLLING_STATS=True`（默认 ON）；可逐窗口关闭：`ROLLING_WINDOWS=[2, 6]`。

> **窗口选择动因**：以 FreeStyle Libre 为代表的 CGM 设备本机普遍存储 8h 历史，2h / 6h 是几乎所有部署场景都能拿到的窗口。舍弃 24h / 7d 是为了避免冷启动场景（新换传感器、新患者入组首日）下 predict.py 拒绝服务。

**因果性约束**：所有 rolling 计算必须 `closed='left'`（pandas）或显式 `t < window_end_timestamp`，**严禁包含窗口右端点本身**（否则会泄漏当前 4-步输入区间内的信息）。

**TIR 定义**：`TIR_70_180 = #{x : 70 ≤ x ≤ 180} / #{x}`，全部以原始 mg/dL 计算（在 z-score 之前）。

**缺失边界**：若窗口内可用 CGM 点数 < `min_periods = window_size_in_points // 2`（即 2h: <4 点、6h: <12 点），整条样本丢弃。这与 §2.4 的"窗口对齐过滤"叠加生效。2h 的门槛非常宽松（只需约 1h 内有过 4 个点）；6h 偶有不达。

D_static = 2 + 5 × 2 = **12**。**不**使用 patient_id embedding。

### 3.3 张量形状

```
seq_input    : (B, T=8, D_seq=6)        # 时序
static_input : (B, D_static=12)         # 静态：Sex + Age + 10 个 rolling stats (2h/6h × 5)
target       : (B,)                     # 单标量，反归一化前的 z-score
```

## 4. 模型与训练

### 4.1 模型结构

```
seq_input  ─► xLSTMBlockStack ─► 取最后 step hidden ─┐
                                                       ├─► concat ─► MLP(2 层) ─► 标量
static_input ─────────────────────────────────────────┘
```

- xLSTMBlockStack 来自 `xlstm 2.0.5`（已安装在 conda env `py312_cu121`）
- mLSTM/sLSTM 比例由 Optuna 搜索
- 输出端 MLP（hidden_dim 由 Optuna 搜索）→ 标量

### 4.2 训练超参（固定项）

- 损失：MSE on z-score 后的标签
- 优化器：AdamW（cosine LR + warmup）
- Early stopping：监测 val RMSE（反归一化），patience 由配置决定
- batch sampler：按时间顺序内部 shuffle 每个 epoch（不破坏 patient-time 关系，但允许批内混合患者）
- 设备：CUDA（py312_cu121 含 cu121）；CPU 仅用于 predict.py 单条推理

### 4.3 Optuna 搜索（可调项）

| 超参 | 范围 |
|---|---|
| `num_blocks` | [2, 6] |
| `embedding_dim` | [64, 256] |
| `mlstm_ratio` | {0, 0.5, 1.0}（纯 sLSTM / 混合 / 纯 mLSTM）|
| `mlp_hidden` | [32, 128] |
| `learning_rate` | [1e-4, 5e-3] log |
| `dropout` | [0.0, 0.3] |
| `batch_size` | {256, 512, 1024} |
| `weight_decay` | [1e-6, 1e-3] log |
| trials | 25–30 |
| sampler | TPESampler |
| pruner | MedianPruner（按 val RMSE 在每 epoch 上裁枝） |

目标：val RMSE 最小（反归一化后的 mg/dL）。

### 4.4 实验追踪 (MLflow)

每个 Optuna trial 一个 MLflow run；最终拟合再开一个 run。记录：

- 超参（所有搜索空间项 + 固定项 + 数据配置 hash）
- 训练 loss / val loss / val RMSE / val MAE 每 epoch 曲线
- 最终 test set RMSE / MAE / R²
- artifacts：模型 `.pt`、`scaler.pkl`、optuna study `.pkl`、SHAP 图

## 5. 评估指标

```
RMSE = sqrt(mean((y_true − y_pred)²))   # 主指标
MAE  = mean(|y_true − y_pred|)           # 鲁棒性辅指
R²   = 1 − SSE/SST                       # 解释方差比例
```

全部在**反归一化后的 mg/dL 空间**计算。

## 6. 可解释性 (SHAP)

独立的 `shap_analyze.py` 脚本（或 `t1d_granada/shap_analysis.py` 模块）：

### 6.1 计算

- 使用 `shap.GradientExplainer` 或 `shap.DeepExplainer`（均原生支持 PyTorch）
- 背景集：从训练集随机采 256 个样本作 baseline
- 目标集：从测试集采 1024 个样本计算 SHAP

### 6.2 可视化（保存为 png + 写入 MLflow artifact）

| 图 | 含义 |
|---|---|
| **特征重要性条形图** | 按 (D_seq + D_static) 维度聚合 SHAP 绝对值均值，跨所有时间步求和 |
| **时间步重要性条形图** | 把 SHAP 跨特征求和后，按 t = 0/1/2/3 聚合，看哪一时间步对预测贡献最大 |
| **(time × feature) 热力图** | 跨样本平均的 |SHAP|，行=时间步、列=特征，发现交互模式 |
| **单样本 waterfall / force plot** | 选 5 个测试样本（含极高血糖、极低血糖、平稳）展示具体归因 |

### 6.3 接口

```bash
python shap_analyze.py --model model/xlstm_best.pt --output reports/shap/
```

输出目录含 4 类图 + 一份 `shap_summary.csv`（特征 × 平均 |SHAP| 排名）。

## 7. 交付物

### 7.1 脚本 (mirror BrisT1D 风格)

| 脚本 | 作用 |
|---|---|
| `prepare_data.py` | 读 csv → 切分 → 构造窗口 → 拟合 scaler → 写 train/val/test 张量 + scaler.pkl |
| `train.py` | 加载张量 → Optuna 搜参（每 trial MLflow run）→ 拟合最终模型 → 写 model |
| `predict.py` | 加载 model + scaler → 输入 (**该患者过去 ≥6 小时 CGM 历史 + 8 步当前 bg + Sex + Age + last_timestamp**) → 在内部计算 2h/6h 滚动统计 → 输出 +30min bg |
| `shap_analyze.py` | 加载 model → 算 SHAP → 出图 |

### 7.2 配置

`t1d_granada/params.py`（mirror `brist1d/params.py`）集中存放：
- 数据相关：`WINDOW_SIZE=8`, `FORECAST_STEPS=2`, `STRIDE=1`, `TIME_TOLERANCE_MIN=2`, `ROLLING_WINDOWS=[2, 6]`, `ROLLING_STATS=['mean','std','tir_70_180','p5','p95']`, `TIR_LOW=70`, `TIR_HIGH=180`
- 特征 flag：`USE_BG_DIFF=True`, `USE_TIME_OF_DAY=True`, `USE_DAY_OF_WEEK=True`
- 训练相关：`SEED`, `MAX_EPOCHS`, `EARLY_STOP_PATIENCE`, `N_TRIALS=25`
- SHAP 相关：`SHAP_BG_SAMPLES=256`, `SHAP_FG_SAMPLES=1024`

### 7.3 路径配置

`settings.json` 同 BrisT1D 风格：
```json
{
  "RAW_DATA_DIR": "./data/",
  "GLUCOSE_FILE": "./data/Glucose_measurements.csv",
  "PATIENT_INFO_FILE": "./data/Patient_info.csv",
  "PROCESSED_DATA_DIR": "./data/processed/",
  "MODEL_DIR": "./model/",
  "MLFLOW_DIR": "./mlruns/",
  "SHAP_OUT_DIR": "./reports/shap/",
  "HYPERPARAMETER_TUNING_DIR": "./hyperparameter_tuning/"
}
```

## 8. 范围边界

### 8.1 在范围内

- 30-min 单点预测
- 单 seed 训练流程（可手动重跑做稳定性检查）
- xLSTM 单一架构 + 上述 8 个超参的 Optuna 搜索
- SHAP 解释（4 类图）
- MLflow 本地 file-store（不部署 server）

### 8.2 不在范围内（YAGNI）

- 多步预测（+15 / +45 / +60 min）
- 与 LSTM / GRU / Transformer / N-BEATS 等基线对比（用户已选定 xLSTM）
- 临床指标 MARD / Clarke Error Grid（首版只用 RMSE/MAE/R²；后续可加）
- 部署 / API server / docker
- patient-level held-out 评估
- 患者 embedding、ICD-9 诊断编码
- 在线学习 / 持续训练 pipeline
- end_to_end.py（与 BrisT1D 一致：3 个独立脚本足够）

## 9. 假设 & 风险

| # | 假设 | 风险 | 缓解 |
|---|---|---|---|
| A1 | 22M 行经过严格时间对齐过滤后仍剩 ≥ 4M 训练样本 | 若过滤过严样本不足 | 监测过滤率；必要时放宽容差至 ±5 min |
| A2 | xLSTM 在 T=8 短序列上的优势仍能体现 | T=8 仍偏短，Transformer/MLP 可能同样好 | 用户已明确选定 xLSTM 作为学习目标，不做对比 |
| A3 | 训练集全局 μ/σ 对所有患者归一化合理 | 个体血糖基线差异大，可能损一定精度 | 后续可加 per-patient 归一化作消融 |
| A4 | `Birth_year` 都准确 | 个别 NaN 或异常值 | 加入数据加载阶段的 sanity check |
| A5 | `xlstm 2.0.5` API 稳定 | 该库较新，可能有 breaking change | 锁定版本到 requirements.txt |

## 10. 与 BrisT1D 项目的差异速查

| 维度 | BrisT1D（原） | 本项目 |
|---|---|---|
| 数据 | 7 模态多变量（bg/insulin/carbs/hr/steps/cals/activity） | 单变量 bg |
| 采样 | 5 min（实际多为 15 min） | 严格 15 min |
| 输入窗口 | 6 h × 5 min = 72 步 | 2 h × 15 min = 8 步 |
| 预测 | +60 min | +30 min |
| 模型 | LightGBM | xLSTM |
| 患者编码 | target encoder × 4 (mean/std/skew/kurt) | Sex + Age 静态特征 |
| 数据增强 | data_expander 滑窗扩张 ~48× | stride=1 滑窗（自然产生 ~T倍样本）|
| 切分 | TabularExpandingWindowCV | 每患者 80/10/10 chronological |
| 调参 | Optuna + LGBM 早停 | Optuna + MedianPruner |
| 解释性 | 无 | SHAP（4 类图）|
| 追踪 | 无 | MLflow |

---

## 下一步

1. ✅ 需求确认（本文档）
2. ⏭ `/ce-plan` 制定实现计划（拆分文件结构、函数 API、单测计划）
3. ⏭ 实现 4 个脚本 + `t1d_granada/` 包
4. ⏭ 跑 baseline + Optuna search + SHAP 报告
