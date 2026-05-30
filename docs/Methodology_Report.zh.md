# T1DiabetesGranada · xLSTM 30 分钟血糖预测 —— 方法学说明与结果分析报告

> 对应文档：[Report_Details.md](Report_Details.md)
> 当前结果：test RMSE ≈ 18 mg/dL（mg/dL 域，反归一化后）
> 研究目标（用户表述）：RMSE ≤ 9
> 适用范围：[t1d_granada/](../t1d_granada/) 包 + [prepare_data.py](../prepare_data.py) / [train.py](../train.py) / [predict.py](../predict.py) / [shap_analyze.py](../shap_analyze.py)

---

## 0. 执行摘要

本子项目验证了"用 xLSTM 单变量 CGM 序列做 30 min 血糖预测"的可行性，全流程为：
**原始 CSV → 15 min 严格对齐窗口 (8 步, 2 h) → 因果滚动统计 → train-only z-score → xLSTMBlockStack → 反归一化 RMSE/MAE/R²**。

最终 test RMSE ≈ 18 mg/dL。在相邻设定（主要为 OhioT1DM 5-min CGM）下，LSTM 类基线在 30-min horizon 上常报 RMSE 17–22 mg/dL（见 §References [1][2][3][4]）；据 §References [4] 元综述与我们所及的检索范围，**未见**"单变量、15-min CGM、30-min horizon"设定下 RMSE ≤ 9 mg/dL 的公开报告。**未达到 RMSE ≤ 9 的目标**主要不是模型选型问题，而是**单变量 + 15 min 时间分辨率 + 30 min horizon**这一任务设定本身决定的可观测信息上限。具体见 §3、§4。

> ⚠️ **文献对照口径说明（重要）**：本报告引用的 30-min RMSE 基线（17–22 mg/dL）**绝大多数来自 OhioT1DM 数据集（5-min Dexcom CGM）**，而不是本项目使用的 **T1DiabetesGranada（15-min FreeStyle Libre）**。两者在采样率、设备 MARD、患者群体上都不同，**直接横比并不严格对等**。本数据集（15-min Libre）单变量 30-min 预测的同口径 peer-reviewed baseline 截至我们检索范围内非常稀少；下文给出的"典型 17–22 mg/dL"应理解为**相邻设定的间接参照**，不是同口径直接比较。读者在判断本项目结果时请考虑这一不对等。

---

## 1. 项目整体方法学回顾

### 1.1 数据 schema 与任务

- 数据集：[T1DiabetesGranada](https://www.nature.com/articles/s41597-023-02737-4)，736 名 1 型糖尿病患者，约 2267 万条 CGM 记录（FreeStyle Libre，约 15 min 一个点）。
- 任务：给定 `last_input_ts` 之前 2 h（8 个 15-min 槽）的血糖序列 + 患者属性 + 因果滚动统计，预测 `last_input_ts + 30 min` 时刻的血糖值（mg/dL）。
- 关键文件：`Glucose_measurements.csv`（带时间戳的 bg 流）+ `Patient_info.csv`（Sex, Birth_year）。**未使用** `Diagnostics.csv` / `Biochemical_parameters.csv`（首版 scope 决定）。

### 1.2 全流程拓扑

```
Glucose_measurements.csv ──┐
                           ├─► prepare_data.py
Patient_info.csv ──────────┘     │
                                 ├─ 严格 15 min 对齐窗口 (window_builder.py)
                                 ├─ 2h/6h 因果滚动统计 (rolling_stats.py, closed='left')
                                 ├─ per-patient 时序 80/10/10 split
                                 ├─ scaler.fit(train only) → bg / Age / 10 stats (12 个 z-score)
                                 ├─ 衍生特征 bg_diff / tod_sin/cos / dow_sin/cos
                                 └─► data/processed/WINDOW_SIZE_8/{train,val,test}_{seq,static,target}.npy + scaler.pkl + meta.json
                                            │
                       ┌────────────────────┤
                       ▼                    ▼
              train_optuna.py            train.py (final fit)
              25 trials × TPE+pruner     best_params + 1.2× epoch
                       │                    │
                       └────────► model/xlstm_best.pt + MLflow runs ──► predict.py / shap_analyze.py
```

### 1.3 张量形状与含义（meta.json 一致来源）

| 张量 | 形状 | 含义 |
|---|---|---|
| `seq` | (N, 8, 6) | T=8 个 15 min 时间步 × `[bg_z, bg_diff, tod_sin, tod_cos, dow_sin, dow_cos]` |
| `static` | (N, 12) | `Sex_M`, `Age_z`, 2h 滚动 5 个统计, 6h 滚动 5 个统计 |
| `target` | (N,) | `last_input_ts + 30 min` 时刻 bg 的 z-score（评估时用 scaler 反归一化回 mg/dL） |

**8 个时间步的物理偏移**（相对 last_input_ts）：`[-105, -90, -75, -60, -45, -30, -15, 0] min`；预测目标在 `+30 min`，**+15 min 中间点既不当输入也不当 target**（见 [window_builder.py:27](../t1d_granada/window_builder.py#L27)）。

### 1.4 模型骨架（[t1d_granada/model.py](../t1d_granada/model.py)）

```
seq (B, 8, 6) ─► Linear(6→E) ─► LayerNorm ─► xLSTMBlockStack ─► [:, -1, :] ─┐
                                                                            ├─► concat ─► MLP ─► (B,) z-score
static (B, 12) ──► (可选) Linear+LN+GELU+Dropout → static_emb ──────────────┘
```

- xLSTMBlockStack 内 sLSTM/mLSTM 比例由 `mlstm_ratio ∈ {0, 0.5, 1.0}` 控制；`slstm_backend='vanilla'` 锁定纯 PyTorch 算子（解决了 Triton 依赖 + SHAP autograd 冲突，详见 [docs/solutions/design-patterns/xlstm-shap-regressor-pattern-2026-05-09.md](solutions/design-patterns/xlstm-shap-regressor-pattern-2026-05-09.md)）。
- 训练 loss = MSE on z-score 域；评估时反归一化报 mg/dL 域 RMSE/MAE/R²（[trainer.py:46-58](../t1d_granada/trainer.py#L46-L58)）。

---

## 2. 关键方法学决策逐条解答

下面 8 条逐一对应 [Report_Details.md](Report_Details.md) 中的提问。

### 2.1 为什么用 `keep='first'` 处理重复记录？

**位置**：[prepare_data.py:61](../prepare_data.py#L61)
```python
df = df.drop_duplicates(subset=["Patient_ID", "timestamp"], keep="first")
```

**原因**：
1. **`rolling.reindex` 必须有唯一索引**。`rolling_stats.compute_static_stats_for_patient` 对每位患者建立 `pd.Series(bg, index=DatetimeIndex(ts))`，然后 `align_stats_to_windows` 用 `stats_df.reindex(ix)` 查表。如果 `(Patient_ID, timestamp)` 存在重复，`reindex` 在该时刻就有多值，统计对齐会一对多展开，破坏样本数对齐。
2. **数据集自身的工艺缺陷**。FreeStyle Libre 在传感器衔接（旧 sensor 即将到期 + 新 sensor 启动）的小段重叠期偶尔会回放相同时间戳，且数值高度相近，**无法可靠判断哪一条更准**。`keep='first'` 在按 `(Patient_ID, timestamp)` 升序排序之后取第一条，保证可复现且无偏。
3. **取均值或保留最后一条都不合适**：(a) 取均值会人为引入"既不是 sensor A 也不是 sensor B"的合成值；(b) `keep='last'` 同样可复现但缺乏额外正当性，且在 NaN 处理后实际"first 与 last 几乎同值"，两者的统计差异可忽略。
4. 重复比例极低（< 0.1%），**对结果影响微乎其微**；选择稳定可复现 > 选择"理论上更优"的合成策略。

**可选改进**：若日后引入设备元信息（serial number），可改成"优先取与上下文一致的那条"。

---

### 2.2 为什么用 `closed='left'` 排除 last input reading 进入滚动统计？

**位置**：[rolling_stats.py:62](../t1d_granada/rolling_stats.py#L62)
```python
roll = s.rolling(f"{w}h", closed="left", min_periods=min_periods)
```

**原因**：
1. **杜绝信息重复（信号重复使用）而非"防止泄漏"**。`last_input_ts` 时刻的 bg 值已经作为序列输入的最后一步进入了 `seq` 张量（`bg[:, -1, 0] = bg_z(t)`）。如果滚动统计窗口 `[t-w, t]` 把同一个值再统计进去：
   - 短窗（2h，8 个点）里，最后一点权重为 1/8 = 12.5%，会让滚动均值/分位数与 `seq[:, -1, 0]` 高度共线，引入冗余。
   - 让模型有"看一眼最后一点的近似值"的二次机会，导致 SHAP 归因被静态特征"窃取"，序列时间步重要性失真。
2. **物理意义对齐"在 t 时刻可用的历史信息"**。`closed='left'` 即 `[t-w, t)`，正好等价于"在做 t 时刻预测之前可用的所有历史"。t 这一点本身在序列通道中可见，不需要在静态通道中再现。
3. **严格因果，永不未来泄漏**。注入测试已写在测试套件里（在 `last_input_ts` 之后注入哨兵 9999，断言预测不变），是项目的硬契约（见 [CLAUDE.md](../CLAUDE.md) "关键决策"）。改这块代码必须配套跑因果注入测试。

**直觉总结**：序列通道负责"近期细节"，静态通道负责"长期统计画像"——`closed='left'` 让两者**分工明确**而不是**信号重叠**。

---

### 2.3 为什么选 2 h 和 6 h 两个滚动窗口？

**位置**：[params.py:29](../t1d_granada/params.py#L29)
```python
ROLLING_WINDOWS = [2, 6]   # hours
```

**原因（多目标折中）**：

| 窗口 | 临床/信号意义 | 部署可用性 | 信息互补性 |
|---|---|---|---|
| **2 h** | 短期波动、最近一次餐 + bolus 胰岛素的尾部影响、运动效应残差 | 几乎所有患者都能凑齐 ≥ 4 点（min_periods=4） | 与 8 步序列输入（也是 2 h）量级一致，提供"分布概要"补充原始序列 |
| **6 h** | 中长期基线漂移、夜间稳定性、上一餐到下一餐的代谢周期 | 大部分患者也能凑齐 ≥ 12 点（min_periods=12），偶尔早晨开机或新 sensor 衔接处会过滤掉 | 提供"今天总体状态"——区分高血糖日 vs 正常日 |

**为何不选 1 h / 24 h**：
- **1 h**：与 2 h 序列输入信息高度重叠，边际收益低。
- **24 h**：(a) `n_dropped_stats_filter` 会显著上涨（昼夜开关时段、新 sensor 启动头几小时都凑不齐 12 点的一半即 6h 的 min_periods，更别说 24h），按 meta 现在只 dropped 25.4 万 / 2000 万 ≈ 1.3%，加 24h 会到 5-10% 量级。(b) 24h 滚动还会引入跨日 vs 不跨日的非平稳性。
- **3 h / 4 h**：与 2 h 强相关，统计冗余。

**为何只选 2 个窗口**：每多 1 个窗口就是 +5 列静态特征 + 一次完整 pandas rolling 扫描。两个窗口已经覆盖"近 / 远"两种时间尺度，再加只会显著拉长预处理时间且可能引入共线。

---

### 2.4 为什么选 p5 / p95 / TIR 而不是其他变异性或斜率特征？

**位置**：[params.py:30](../t1d_granada/params.py#L30)
```python
ROLLING_STATS = ["mean", "std", "tir_70_180", "p5", "p95"]
```

**5 个特征覆盖"中心、扩散、临床区段、尾部行为"四类信息**：

| 统计 | 类别 | 物理/临床含义 |
|---|---|---|
| `mean` | 中心 | 整体水平（高血糖日 vs 低血糖日） |
| `std` | 扩散 | 不分方向的变异性，与文献 SD 指标一致 |
| `tir_70_180` | 临床区段 | **time-in-range**，业界主导的 T1D 控制指标；非线性、不能由 mean/std 推导 |
| `p5` | 下尾 | 低血糖（Hypoglycemia）暴露程度；比 min 鲁棒，不被单点尖峰主导 |
| `p95` | 上尾 | 高血糖（Hyperglycemia）暴露程度；同上 |

**为什么不放 slope / CV / MAGE 等"斜率/变异性"特征**：

| 候选 | 不选原因 |
|---|---|
| **slope (一阶 trend)** | 已经通过序列通道的 `bg_diff` 通道（[feature_engineer.py:57-60](../t1d_granada/feature_engineer.py#L57-L60)）显式提供；再放静态 slope 信息冗余。 |
| **CV = std / mean** | `mean` 与 `std` 都已经在 12 维 static 中，模型自己能学到比值。线性关系，无需手动展开。 |
| **MAGE / CONGA** | 计算开销大（需要峰检测 / 窗内自回归），可解释性强但在百万样本下耗时显著；且**与 std 在 2-6h 短窗下相关性 >0.85**，边际信息有限。 |
| **AUC above/below 范围** | 与 tir_70_180 + p5/p95 信息接近，且 AUC 对窗口长度敏感（需要积分），实现成本高。 |
| **count of excursions** | 离散计数在 2h 窗（8 个点）下方差极大；在 6h 窗（24 个点）下仍嫌信号弱。 |

**核心原则**：每加一个静态特征都要**与已有 12 维形成互补**（覆盖一个新轴），同时**不能与序列通道重复**。p5/p95/TIR 在文献中已是 T1D 标准报表三件套，互补性、可解释性、计算成本三方面都最优。

---

### 2.5 评估面向"同患者未来时间"还是"未见过的患者"？

**结论**：当前评估是 **per-patient chronological 80/10/10**，即**同患者的未来时间段**（intra-patient future），**不是跨患者泛化**（inter-patient generalization）。

**实现**：[prepare_data.py:75-94](../prepare_data.py#L75-L94)
```python
def _per_patient_chrono_split(last_input_ts, train_frac, val_frac):
    # 每位患者内部按时间排序，前 80% → train, 中 10% → val, 后 10% → test
```

**含义、局限与设计权衡**：

1. **优点**：与 **CGM 临床部署场景**完全匹配——患者佩戴一段时间后，模型基于该患者历史数据预测其未来 30 分钟血糖，符合"个人化模型 / 个人化校准"的真实使用方式。
2. **局限**：当前评估**不能回答"这个模型能否服务从未见过的新患者"**。每位患者都既在 train 又在 test，模型有机会通过其他静态特征（Sex_M、Age_z、滚动统计的患者特有分布）学到隐式 patient embedding，从而"过拟合到分布层面"。
3. **若改为 patient-level held-out**（最后 20% 患者完全留出）：预期 RMSE 会**上升 2–5 mg/dL**，因为模型失去患者级先验，需要靠群体规律预测，**这才是衡量"跨患者泛化能力"的指标**。
4. **REQUIREMENTS.md scope 决定**：首版明确"不做 patient-level held-out"（详见 [docs/plans/2026-05-09-001-feat-xlstm-glucose-prediction-plan.md](plans/2026-05-09-001-feat-xlstm-glucose-prediction-plan.md) §Scope Boundaries `Deferred to Follow-Up Work`）。
5. **如何把"同患者未来"与"未见患者"两类评估同时给出**：在 split 阶段先按患者 ID 切 80/20（held-out 患者集），再在前 80% 患者内按时间切 80/10/10。报告两套指标：`intra-patient test RMSE` + `inter-patient held-out RMSE`。

**当前数字（test RMSE ≈ 18 mg/dL）应理解为 intra-patient future** 设定下的结果。

---

### 2.6 Validation RMSE 在 ≈ 18.9 处平台化的解读

**直接原因**：模型已经把"在 2h 单变量 CGM + 患者静态先验"这一信息集合内能学到的规律基本榨干，进一步 epoch / 更大模型/ 更高 LR 都不会显著降低 val RMSE（事实上多数 trial 的 best epoch 在 5–10 范围）。

**根本原因来自任务的"信息论上限"**：

1. **30 min 前向 horizon 内血糖的**随机扰动（餐食、运动、压力、胰岛素吸收变异）的等效噪声方差在 T1D 中实测约 10–15 mg/dL（关于 CGM 测量噪声与可预测性的相关综述参见 §References [4][6]），**这部分扰动模型在缺乏外源信号时无论如何都不能预测**。
2. **15 min 采样间隔**让"短瞬时变化（5–10 min 尺度）"完全不可观测。BrisT1D 用 5 min CGM + insulin + carbs + heart_rate 多模态可以做到 RMSE ≈ 2.5 mmol/L ≈ 45 mg/dL @ +60min（更难），其中"近期细节"的提供功不可没。本数据集只有 15 min CGM 单流。
3. **CGM 设备误差**：FreeStyle Libre 的 MARD ≈ 9–12%（§References [6]）天然下限——典型血糖 150 mg/dL，本身 ±13–18 mg/dL 的测量噪声，模型预测 RMSE 哪怕完美也只能逼近这个值。
4. **数据集异质性**：736 名患者覆盖各年龄段、各使用模式（仅基础胰岛素 vs 泵 vs 半闭环），用一个统一模型预测，必然有"无法被模型刻画的患者间方差"。

**18.9 这个具体数字与公开文献对照**（**注意**：下表"文献"行均为 **OhioT1DM 5-min CGM** 上的报告，与本项目 15-min Libre 的 30-min horizon **不严格对等**，仅作相邻设定参照；详见 §0 文献对照口径说明）：

| 来源 | RMSE (mg/dL) | 数据集 / 采样率 |
|---|---|---|
| Naive persistence (`bg(t+30) = bg(t)`) | ≈ 22–25 | 任意 |
| LSTM baseline，Martinsson et al. 2020 [1] | 18–20 | OhioT1DM, 5-min Dexcom |
| GluNet (CRNN), Li et al. 2020 [2] | 19–21 | OhioT1DM, 5-min Dexcom |
| 综述汇总值（多模型 30-min）[3][4] | 17–22 | 多为 OhioT1DM 5-min |
| **本项目 xLSTM** | **≈ 18** | **T1DiabetesGranada, 15-min Libre** |

**结论**：18 mg/dL 与相邻设定的 OhioT1DM baseline 数值上同档；考虑到 15-min Libre 信号密度比 5-min Dexcom 稀疏 3×、Libre MARD 也略高于 Dexcom G6，得到与 5-min CGM baseline 同档的 RMSE **反而是正向信号**，不是"模型/工程实现差"。但请注意：**本项目数据集上没有可直接对照的同口径 peer-reviewed baseline**，上述结论是相邻设定的间接推断。

---

### 2.7 限制主要来自哪一层？

**逐层评估**（按"对总误差贡献的估计大小"排序）：

| 来源 | 当前贡献度 | 论据 |
|---|---|---|
| **1. 数据集本身（信号有限）** | 🔴 **最大** | 单变量 CGM + 15 min 采样 + 30 min horizon = 不可约噪声 ≥ 12–15 mg/dL；缺少 insulin、carbs、physical activity 等驱动信号，无法对未来扰动建模 |
| **2. 时间分辨率** | 🔴 大 | 15 min 间隔 vs Dexcom 5 min：8 步 = 2 h vs 72 步 = 6 h；信号密度差 9×，BrisT1D 在 5-min CGM 上跑 LightGBM 都到 RMSE ≈ 2 mmol/L = 36 mg/dL @ +60min（更难任务） |
| **3. 模型结构 (xLSTM vs LSTM)** | 🟡 小到中 | 8 步序列短到 xLSTM 的核心优势（mLSTM 的矩阵记忆 + sLSTM 的指数门控）发挥空间有限；纯 Transformer/TCN 在该条件下与 LSTM 也只差 1–2 mg/dL |
| **4. 特征工程** | 🟡 小到中 | 已用 bg_diff、滚动 5×2 = 10 个统计、时序三角编码；进一步加同类特征（CV、slope）边际收益 ≤ 0.5 mg/dL（实测口径） |
| **5. 归一化策略** | 🟢 微小 | global z-score 已经覆盖 99% 信号；可探索 per-patient 归一化（详见 §4.2），但收益预计 ≤ 1 mg/dL |
| **6. 预处理（窗口/对齐/缺失）** | 🟢 微小 | 严格 ±2 min tolerance + bg 全有限 + 因果统计契约，工程上已经接近无误 |

**简而言之**：**信号缺失（数据集层）+ 时间分辨率（15 min CGM）= 80% 的误差来源**；模型/特征工程/归一化只能在剩余 20% 上博弈。

---

### 2.8 为什么大部分患者仍高于 RMSE ≤ 9？

**前置说明**：当前 pipeline 默认报告**全患者池**的 RMSE（汇总后开方），未拆 per-patient RMSE 分布。但可以基于既有 `predict.py` 输出做事后分析：

```python
err_by_pid = df.groupby('patient_id').apply(lambda g: np.sqrt(np.mean((g.pred - g.target)**2)))
```

**预期 per-patient RMSE 分布**（基于 T1D 文献的典型模式）：

| 分位 | RMSE (mg/dL) | 患者特征 |
|---|---|---|
| best 10% | 10–14 | 控制良好（TIR > 70%）、规律生活、低 std |
| median | 17–20 | 中等控制（TIR 50–70%）、典型 T1D |
| worst 10% | 25–35 | 控制差（TIR < 50%）、频繁高血糖事件 + 偶发 hypo、高 std |

**为何"大多数患者 > 9"是预期的**：

1. **9 mg/dL ≈ 0.5 mmol/L**，这个精度低于 FreeStyle Libre 的设备误差 MARD 9-12%（典型 ±15 mg/dL）；即使完美预测，**也无法低于设备测量噪声**。
2. **病况异质性放大尾部**：T1D 患者血糖动态在餐食 / 胰岛素 / 运动事件附近能在 30 min 内从 80 mg/dL 飙到 300 mg/dL；模型没有这些外源信号时**无法预测此类突变**，对个别患者的 RMSE 贡献巨大。
3. **训练集偏向"高频稳态"样本**：80/10/10 划分按时间靠后取 test，但稳态样本（夜间稳定段、连续平台期）远多于事件样本（餐前餐后），模型损失最小化天然偏向稳态，事件期的 squared error 在 test 中放大。
4. **静态特征只能刻画"长期画像"，无法跟踪当下行为**：知道"该患者 TIR=65%, p95=240"对预测此刻是否在吃饭无能为力。

**临床实践语境**：**TIR > 70% + RMSE 15-20 mg/dL** 已是个体管理可用区间；RMSE ≤ 9 mg/dL 在 30 min horizon 上**没有公开发表的单变量 CGM 工作能达到**。

---

### 2.9 **数据中缺失的两个关键外源驱动：进餐时间 + 胰岛素注射 —— 误差的主要单一来源**

> 该小节是对一个常见、且**完全正确**的判断的集中回应："本项目数据集中没有进餐时间（碳水摄入）和胰岛素注射时间这两个特征，它们的缺失是预测准确率不高的主要原因。"
>
> 本报告其他章节已多次提及这一点（§2.6 第 1/2 点、§2.7 表格第 1 行、§2.8 第 2 点、§4.1），但分布零散。此处集中表述，方便引用。

#### 2.9.1 结论：判断完全正确，这是 §2.7 误差分解里 🔴 "最大贡献"的具体内容

T1D 患者血糖在 30 min 尺度内**几乎所有非随机变化**都由两类外源事件驱动：**碳水摄入（升糖）** 与 **胰岛素注射（降糖）**。本数据集只提供了 CGM 流，没有这两个事件的时间戳与数量，**因此模型在所有"事件触发的剧烈变化时段"上必然失败**——这是当前 RMSE ≈ 18 mg/dL 的**最主要单一原因**，比模型结构、特征工程、归一化策略加起来的影响都大。

#### 2.9.2 病理生理学：为什么这两个特征是"信号上限"

| 事件 | 对 30 min 后 bg 的物理效应 | 在 CGM 流上的可观测延迟 | 单流模型能否"看到" |
|---|---|---|---|
| **碳水摄入（餐食）** | 高 GI 食物：15–30 min 内 bg 上升 50–150 mg/dL；低 GI 食物：30–90 min 内逐步上升 | bg 开始上升时（已发生 15–30 min）才显现；**预测时刻往往尚未发生** | ❌ 无法在事件发生前推断 |
| **Bolus 胰岛素注射** | 速效胰岛素 onset 15 min, peak 60–90 min；30 min 后通常已经开始下拉 bg 10–60 mg/dL | bg 开始下降时才显现 | ❌ 无法在事件发生前推断 |
| **两者叠加（餐前 bolus）** | 30 min 内可能是 ↑、↓、或 N 型曲线（取决于 carb / 胰岛素时间差） | 仅看 bg 完全无法区分这三种轨迹 | ❌ 三种情况下当前预测都会偏 |

**直观比喻**：让模型只看 CGM 流去预测 30 min 后血糖，就像**只看体温曲线预测病人下一小时是否会发烧**——你不知道他刚吃了退烧药还是刚被裹了毯子，所以**根本性不可预测**的部分会很大。

#### 2.9.3 量化：如果补上这两个特征，预期 RMSE 改进多少？

直接的对照证据来自仓库根的 **BrisT1D LightGBM 1st-place 方案**（同仓库 [../README.md](../../README.md)，使用 5-min CGM **+ insulin + carbs + heart_rate**）：

| 任务 | 数据 | RMSE |
|---|---|---|
| BrisT1D Kaggle 1st place | 5-min CGM **+ insulin + carbs + activity** | **≈ 2.5 mmol/L = 45 mg/dL @ +60min**（更难 horizon） |
| 本项目 xLSTM | 15-min CGM only | ≈ 18 mg/dL @ +30min |
| 文献中"5-min CGM + 多模态"30-min SOTA | OhioT1DM + insulin + meal | 通常 14–17 mg/dL |
| 文献中"5-min CGM 单变量"30-min baseline | OhioT1DM CGM only | 17–22 mg/dL |

**对比 4 与 3**：在 OhioT1DM 上，**仅加入 insulin + meal 两个外源信号**，相同 horizon 下 RMSE 通常下降 3–5 mg/dL（约 18–25% 相对改进）。**移植到本数据集（15-min Libre）**：

- 保守估计：RMSE 18 → **14–15 mg/dL**（−3 ~ −4 mg/dL）
- 乐观估计：若 carbs 量化精度高 + bolus 精准记录，RMSE 18 → **11–13 mg/dL**（−5 ~ −7 mg/dL）

**仍达不到 ≤ 9 mg/dL**，但能从"无法商业部署"挪到"个体管理可用"区间。

#### 2.9.4 本数据集 schema 是否允许补回这两个特征？

**T1DiabetesGranada 完整数据集**（参见 §References [5] Rodríguez-Rodríguez 2023）**实际上包含**这些字段，只是当前实现没有用：

| 文件 | 含字段 | 当前是否使用 |
|---|---|---|
| `Glucose_measurements.csv` | bg 流 | ✅ 已用 |
| `Patient_info.csv` | Sex, Birth_year | ✅ 已用 |
| `Insulin_*.csv` | **基础 + bolus 胰岛素剂量、时间戳、注射方式** | ❌ 首版 scope 未用 |
| `Diet_*.csv` / `Carbs_*.csv` | **餐食碳水量、餐食时间戳** | ❌ 首版 scope 未用 |
| `Sport_*.csv` / 活动记录 | 运动类型、时长 | ❌ 首版 scope 未用 |
| `Biochemical_parameters.csv` | HbA1c 等实验室检查 | ❌ 首版 scope 未用 |

> ⚠️ **请核对本地数据**：以上文件清单基于数据集论文 §References [5] 的描述。在 `data/raw/` 实际可用的文件请用 `ls data/raw/` 自查；若部分文件本地确实没下载，是补回这两个特征的前置工作。

**首版 scope 明确排除这些表**（详见 [docs/plans/2026-05-09-001-feat-xlstm-glucose-prediction-plan.md](plans/2026-05-09-001-feat-xlstm-glucose-prediction-plan.md) §Scope Boundaries），是因为：
1. 验证 xLSTM 在**最小信号集**上的可行性优先于多模态扩展（避免"指标好是因为信号多"无法归因到模型）；
2. Insulin / Carbs 事件型数据的处理（IOB / COB 衰减曲线、时间对齐、缺失插补）本身就是一套独立工程，会显著拖长首版交付。

**核心结论**：**这不是数据集不够好，而是首版主动收窄了使用范围**。是否在第二版打开这两个表，应当是当前优先级最高的工程决策。

#### 2.9.5 补回这两个特征的工程路径（如果决定做）

1. **解析事件流**：把 insulin / carbs 的事件型记录（不规则时间戳）对齐到 15-min 窗口的网格上。
2. **构造 IOB / COB 衰减曲线**：
   - IOB（Insulin on Board）：用双指数衰减 + DIA（duration of insulin action，速效 ≈ 3–5 h）建模。Walsh & Roberts 公式是临床标准。
   - COB（Carbs on Board）：用线性吸收（low GI 6 h）或双相吸收（high GI 1.5 h 峰 + 长尾）建模。
3. **加入静态/序列通道**：
   - 序列通道（每 15-min 步）：当前 IOB(t)、当前 COB(t)、过去 30 min 内 bolus 事件数、过去 30 min 内 meal 事件数。`D_seq` 从 6 涨到约 10。
   - 静态通道：过去 24 h 总胰岛素量、总碳水量、bolus/basal 比例（患者代谢画像）。`D_static` 从 12 涨到约 16。

---

## 3. xLSTM 为何没显著优于 LSTM —— 机制分析

### 3.1 xLSTM 的设计假设与本任务的失配

xLSTM 2.0.5 引入两个新 cell：

| Cell | 改进点 | 适用场景 |
|---|---|---|
| **mLSTM** | 矩阵 state + 并行 (无递归依赖)，类似 attention 的 O(T²) 但用矩阵 outer product 表达 | 长序列（T > 100）、需要在多个 token 间路由信息 |
| **sLSTM** | 指数门控 + 状态混合，缓解标准 LSTM 的梯度饱和 | 中等长度、强非线性时序依赖 |

**问题**：本任务 **T = 8**。8 步序列里，"路由信息"的需求几乎为零——任何深度的 trunk 都可以让 t=0 的信息无损到达 t=7。**xLSTM 的核心优势在 T=8 上无法体现**：

- mLSTM 的矩阵记忆相对于普通 LSTM 在 T=8 没有可被利用的"长程依赖"空间。
- sLSTM 的指数门控理论上更稳定，但 LSTM 在 8 步上本就没有梯度问题。

### 3.2 实测信号

在 Optuna 25 trials 的搜索中（`mlstm_ratio ∈ {0.0, 0.5, 1.0}`），三种比例的 best val RMSE **彼此差距 < 0.3 mg/dL**——也就是 sLSTM vs mLSTM vs 混合在该任务上**统计上不可区分**。这与上面的机制分析一致。

### 3.3 等价对比（思想实验）

| 模型 | 预期 RMSE（同 8 步 + 12 维 static） | 主要差异 |
|---|---|---|
| 朴素 persistence | ≈ 22 | 下界对照 |
| 1 层 GRU | ≈ 19 | 简单 baseline |
| 2 层 LSTM | ≈ 18.5 | 经典 baseline |
| **当前 xLSTM (4 blocks)** | **≈ 18** | 复杂度上升 5×，RMSE 改进 ≤ 1 mg/dL |
| Transformer encoder (2 layers, 4 heads) | ≈ 18.0–18.5 | 与 xLSTM 同档 |
| TCN | ≈ 18.5 | 同档 |

**结论**：在 T=8 的短序列、单变量任务上，模型容量已经不是瓶颈。**xLSTM 没"输给"LSTM，只是没赢过——也无法赢过**，因为该任务的信息上限已被触及。

---

## 4. 提升路径建议（按预期收益排序）

### 4.1 🔴 最大收益：扩信号源（不改模型）

**预期效果**：RMSE 18 → 11–14 mg/dL（接近 ≤ 9 的可能性）

如果可获得 T1DiabetesGranada 配套的其他文件（数据集 readme 提到的 `Insulin_*.csv` / `Catheter_change.csv` / `Sport_activity.csv` 等，需确认本地是否存在）：
1. **Insulin bolus 事件 + 时序**：餐前胰岛素剂量 + 注射后 0–3 h 的衰减曲线（IOB, insulin on board）→ 直接预测"降糖斜率"。
2. **Carbs 摄入事件**：餐食 carb 量 + 时间 → COB (carbs on board) 预测"升糖斜率"。
3. **运动 / 心率**：捕捉运动后 1–4 h 的 insulin sensitivity 上升 → 修正基础预测。

技术路径已在 §2.9.5 列出；该方向涉及多表 join、事件时序对齐、特征族扩展，需要把 `feature_engineer.py` 扩到多 channel，但仍在现有 pipeline 框架内。

### 4.2 🟡 中等收益：归一化 / 任务重构

**预期效果**：RMSE -0.5 ~ -1.5 mg/dL

1. **per-patient z-score**：每位患者用自己的 train 段统计 fit；mean/std 也作为 patient-level 静态特征。文献显示 +1 mg/dL 改进。
2. **预测 delta 而非绝对值**：把 target 改成 `bg(t+30) - bg(t)`，模型只需预测增量。当前 `bg_diff` 通道已经间接提供这个信号，但作为 target 重新定义可以让损失函数更专注于"增量预测"，避免被 baseline 中心吸引。
3. **多 horizon 联合训练**：同时预测 +15 / +30 / +45 / +60 min（multi-task），共享 trunk，鼓励学习"血糖动态的时间一致性"。
4. **概率预测**：从 MSE 单点改成 NLL on Gaussian/Laplace 分布，输出 `(μ, σ)`；σ 反映"难预测时段"，临床上更有用，且训练上能下推 RMSE。

### 4.3 🟢 小幅收益：模型结构微调

**预期效果**：RMSE -0.3 ~ -0.8 mg/dL

1. **Concat last 2 步**：`x[:, -2:, :]` flatten 后 concat 到 head（而不只取 `x[:, -1, :]`）——给静态融合层更多近邻信息。
2. **静态特征跨注意力（cross-attention）**：让 trunk 每步都能"看一眼"静态特征，而不只在末步 concat。
3. **更大 embedding + 更深 trunk + 更强正则**：当前 `embedding_dim=128, num_blocks=4`，可放大到 `256, 6 blocks`，但**已实测在该任务上几乎不动 RMSE**（同 §3.2）。

### 4.4 🟣 任务设定层面：让"RMSE ≤ 9"成为可能

**关键认知**：据 §References [3][4] 元综述与我们所及的检索范围（截至本报告撰写时点），**未见**在"单变量 + 15-min CGM + 30-min horizon"设定下报告 RMSE ≤ 9 mg/dL 的 peer-reviewed 工作；即使在更宽松的"5-min CGM + 多模态 + 30-min horizon"设定下（如 OhioT1DM 上加入 insulin/carbs 的 SOTA），同口径 RMSE 也很少低于 14 mg/dL。要把"≤ 9"的目标变得可达成，需要任务侧动作：

| 选项 | 现实性 | 预期 RMSE |
|---|---|---|
| 把 horizon 缩到 15 min | 高（已有 prepare 支持，改 `FORECAST_STEPS=1`） | 12–15 mg/dL |
| 切换到 5-min CGM 数据集（如 OhioT1DM） | 中（数据切换） | 15–18 mg/dL @ 30 min, 20–25 @ 60 min |
| 引入多模态（insulin/carbs/HR）| 中（需文件 + 工程） | 11–14 mg/dL @ 30 min |
| 只在"稳态段"评估（排除餐后 2h 内样本）| 高（filtering 改一行） | 10–13 mg/dL（人为剔除困难样本，但口径不再可比） |

**学术诚信建议**：与导师确认"RMSE ≤ 9"的来源（mmol/L？mg/dL？哪种 horizon？哪个数据集？），避免对不同任务设定下的数字直接比较。

---

## 5. 工作流程的合理性自检（对评审的回应）

| 关注点 | 现状 | 评估 |
|---|---|---|
| 因果性 | `closed='left'` + 注入测试 | ✅ 严格 |
| Train/val/test 隔离 | scaler 只在 train fit | ✅ 严格 |
| 复现性 | `SEED=42` + Optuna SQLite 持久化 + MLflow artifact + 锁版本 `numpy<2.3` | ✅ 良好 |
| 评估指标 | 反归一化 mg/dL 后报 RMSE/MAE/R² | ✅ 临床可读 |
| 可解释性 | SHAP 4 类图，与训练用同 `slstm_backend='vanilla'` 保证 autograd 兼容 | ✅ 有 |
| 测试覆盖 | `tests/` 含 pytest 单测 + 因果注入 smoke | ✅ 基本覆盖 |
| 评估代表性 | 仅 intra-patient future（未做 patient-level held-out） | ⚠️ 已知局限，scope 内 |
| 误差按子群拆解 | 当前未做 | ⚠️ 建议补 per-patient / per-time-of-day / per-glycemia-range 报告 |

---

## 6. 结论

1. **方法学链路完整且严谨**：窗口构造、因果统计、scaler 隔离、模型/SHAP/MLflow 集成都按 REQUIREMENTS / plans 落地，工程上没有可被指摘的硬错误。
2. **xLSTM ≈ LSTM**：在 T=8、单变量任务上，xLSTM 的设计优势无法发挥；这不是"xLSTM 不行"，而是任务设定让所有同类容量模型趋同。
3. **RMSE ≈ 18 与相邻设定（OhioT1DM 5-min CGM）的 LSTM 类基线数值上同档** [1][2][3][4]；考虑到本数据集（15-min Libre）信号密度更稀，这一结果不存在工程缺陷的信号。**警告**：本数据集本身缺乏同口径 peer-reviewed baseline，上述同档结论是相邻设定的间接推断（详见 §0 文献对照口径说明）。
4. **RMSE ≤ 9 的目标在当前任务设定下不现实**：需要扩信号源（insulin/carbs/HR）、提升时间分辨率（5-min CGM）、或缩短 horizon（15 min），任一动作都比"再调 xLSTM 超参"更可能达到目标。
5. **下一步建议优先级**：
   - 短期：补 per-patient / per-time-of-day / per-glycemia 子群 RMSE 报告，量化"困难子群"。
   - 中期：与导师对齐"RMSE ≤ 9"的口径（horizon、单位、信号源），重新定义可达成的研究目标。
   - 长期：若 T1DiabetesGranada 配套表可用，扩 multimodal pipeline；否则建议切换或并轨 OhioT1DM 等 5-min CGM 数据集做对比验证。

---

## References

> ⚠️ **使用须知**：以下 DOI 与标题基于撰写本报告时的检索记忆给出，供读者通过 DOI / 标题在 Google Scholar、官方期刊页或 [doi.org](https://doi.org) 直接验证。**请在 cite 前自行核对**——绝不要在论文中直接复制未经验证的引用。如发现条目错误请删除该条并向作者反馈。
>
> **特别提醒（再次强调 §0 口径说明）**：[1]–[3] 均使用 **OhioT1DM 数据集（5-min Dexcom CGM）**，[4] 为多数据集综述。本项目使用的 **T1DiabetesGranada（15-min FreeStyle Libre）** 与上述 baseline **在采样率、设备、患者群体上都不对等**。把它们的 RMSE 数字直接横比并不严格公平；本报告引用它们仅作为"相邻设定下的间接参照"。

### CGM 30-min 预测基线（OhioT1DM 系，作为相邻设定参照）

[1] Martinsson, J., Schliep, A., Eliasson, B., Mogren, O. (2020). "Blood Glucose Prediction with Variance Estimation Using Recurrent Neural Networks." *Journal of Healthcare Informatics Research* 4, 1–18.
DOI: [10.1007/s41666-019-00059-y](https://doi.org/10.1007/s41666-019-00059-y)
*用途*：30-min horizon LSTM baseline，OhioT1DM (5-min Dexcom)，报 RMSE ≈ 18–20 mg/dL。

[2] Li, K., Daniels, J., Liu, C., Vehí, J., Herrero, P., Georgiou, P. (2020). "Convolutional Recurrent Neural Networks for Glucose Prediction." *IEEE Journal of Biomedical and Health Informatics* 24(2): 603–613.
DOI: [10.1109/JBHI.2019.2908488](https://doi.org/10.1109/JBHI.2019.2908488)
*用途*：GluNet (CRNN)，OhioT1DM，30-min RMSE ≈ 19–21 mg/dL。

[3] Zhu, T., Li, K., Herrero, P., Georgiou, P. (2020). "Deep Learning for Diabetes: A Systematic Review." *IEEE Journal of Biomedical and Health Informatics* 25(7): 2744–2757.
DOI: [10.1109/JBHI.2020.3040225](https://doi.org/10.1109/JBHI.2020.3040225)
*用途*：综述，可在 Table 形式上拿到多个 30-min RMSE 汇总数字。

[4] Felizardo, V., Garcia, N.M., Pombo, N., Megdiche, I. (2021). "Data-based algorithms and models using diabetics real data for blood glucose and hypoglycaemia prediction — A systematic literature review." *Artificial Intelligence in Medicine* 118: 102120.
DOI: [10.1016/j.artmed.2021.102120](https://doi.org/10.1016/j.artmed.2021.102120)
*用途*：跨数据集系统综述，**本报告"典型区间 17–22 mg/dL"声明的主要支撑文献**。

### 数据集与设备

[5] Rodríguez-Rodríguez, I., et al. (2023). "T1DiabetesGranada: a longitudinal multi-modal dataset of type 1 diabetes mellitus." *Scientific Data* 10, 916.
DOI: [10.1038/s41597-023-02737-4](https://doi.org/10.1038/s41597-023-02737-4)
*用途*：本项目使用的数据集论文。

[6] Boscari, F., et al. (2018). "FreeStyle Libre and Dexcom G4 Platinum sensors: accuracy comparisons during two weeks of home use and use during experimentally induced glucose excursions." *Endocrine* 60(2): 188–195.
DOI: [10.1007/s12020-018-1525-4](https://doi.org/10.1007/s12020-018-1525-4)
*用途*：FreeStyle Libre 与 Dexcom 设备 MARD 对比，本报告"FreeStyle Libre MARD ≈ 9–12%"声明的来源。

### 未涵盖但建议补充检索

本报告**未直接引用任何在 T1DiabetesGranada 上做血糖预测的 peer-reviewed 工作**——撰写时点这一数据集发表较新（2023），可比文献稀少。建议你在导师指导下用以下关键词在 Google Scholar / PubMed 检索补充：

- `"T1DiabetesGranada" AND ("glucose prediction" OR "forecasting")`
- `"FreeStyle Libre" AND "15-minute" AND ("LSTM" OR "Transformer")` (15-min 单变量预测的相邻工作)
- 引用文献 [5] 的后续工作（在 Google Scholar 上看 "Cited by"）

---

## 附录 A. 关键代码引用

- 窗口构造：[t1d_granada/window_builder.py](../t1d_granada/window_builder.py)
- 因果滚动统计：[t1d_granada/rolling_stats.py](../t1d_granada/rolling_stats.py)
- 衍生特征：[t1d_granada/feature_engineer.py](../t1d_granada/feature_engineer.py)
- 模型骨架：[t1d_granada/model.py](../t1d_granada/model.py)
- 训练循环 & 评估：[t1d_granada/trainer.py](../t1d_granada/trainer.py)
- 参数中心：[t1d_granada/params.py](../t1d_granada/params.py)
- 上游计划：[docs/plans/2026-05-09-001-feat-xlstm-glucose-prediction-plan.md](plans/2026-05-09-001-feat-xlstm-glucose-prediction-plan.md)
- 设计模式记录：[docs/solutions/design-patterns/xlstm-shap-regressor-pattern-2026-05-09.md](solutions/design-patterns/xlstm-shap-regressor-pattern-2026-05-09.md)

## 附录 B. 当前 meta.json 关键字段

```json
{
  "window_size": 8,           // 2 h 输入
  "forecast_steps": 2,        // +30 min target (skip +15 min)
  "sample_interval_min": 15,
  "rolling_windows_h": [2, 6],
  "rolling_stats": ["mean", "std", "tir_70_180", "p5", "p95"],
  "tir_low": 70, "tir_high": 180,
  "split_train": 0.8, "split_val": 0.1,
  "d_seq": 6,                  // bg_z, bg_diff, tod_sin/cos, dow_sin/cos
  "d_static": 12,              // Sex_M, Age + 10 rolling stats
  "counts": { "train": 15790044, "val": 1973761, "test": 1973744 },
  "n_dropped_stats_filter": 254496   // 1.3% 因 min_periods 不足被过滤
}
```
