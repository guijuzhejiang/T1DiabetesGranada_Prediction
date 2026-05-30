# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in the `T1DiabetesGranada_Prediction/` subproject.

## 项目概述

基于 [T1DiabetesGranada 数据集](https://www.nature.com/articles/s41597-023-02737-4)（每 15 分钟一次的连续 CGM）预测 30 分钟后血糖。栈：**PyTorch + xLSTM 2.0.5 + Optuna + MLflow + SHAP**。模型把 2 小时（8 个 15 分钟点）的序列特征 + 滚动统计静态特征喂给 `xLSTMRegressor`，输出标量回归。

与仓库根的 BrisT1D LightGBM 解决方案完全独立，不共享代码或数据。

## 常用命令

工作目录是本子目录（`T1DiabetesGranada_Prediction/`，脚本依赖相对路径 `./settings.json`）。conda 环境 `py312_cu121`，已装 `xlstm 2.0.5`、`shap`、`optuna`、`mlflow`，`numpy<2.3`（numba/shap 兼容）。

```bash
# 四阶段
python ./prepare_data.py   # CSV → 窗口 + 特征 + scaler，落到 data/processed/WINDOW_SIZE_<P.WINDOW_SIZE>/
CUDA_VISIBLE_DEVICES=1 python ./train_optuna.py   # Optuna 搜索 → hyperparameter_tuning/best_params.json + study.pkl + best_trial_meta.json
CUDA_VISIBLE_DEVICES=1 python ./train.py          # 用 best_params 在 train 上拟合 + val 早停 → model/xlstm_best.pt + test 指标
python ./predict.py --history ... --patient_id ... --last_ts ...   # 单条/批量推理
python ./shap_analyze.py   # SHAP 计算 + 4 类图 → reports/shap/

# 测试
pytest tests/ -q
```

所有可调参数集中在 [t1d_granada/params.py](t1d_granada/params.py)，**没有 CLI flag**——改 `params.py` 即可调整 `WINDOW_SIZE`、`PATIENCE`、`STUDY_NAME`、`OPTUNA_STORAGE`、`OPTUNA_SEARCH_SPACE`、`DEFAULT_HP`、`FINAL_FIT_EPOCHS_OVERRIDE` 等。`train.py` 例外：`best_params.json` 不存在时，可通过 `--embedding-dim / --num-blocks / --static-embedding-dim / ...` 等 fallback CLI（默认值取自 `P.DEFAULT_HP`）直接训练。

`batch_size` 不进搜索空间，固定使用 `P.BATCH_SIZE`。

## 预处理产物 `data/processed/WINDOW_SIZE_<N>/` 结构

`prepare_data.py` 一次性物化训练所需的全部张量到 **`data/processed/WINDOW_SIZE_<P.WINDOW_SIZE>/`**（如 `WINDOW_SIZE_4/`、`WINDOW_SIZE_8/`），`train_optuna.py` / `train.py` / `shap_analyze.py` 直接 `np.load(..., mmap_mode='r')` 消费，**不会再回到 CSV**。

按 `WINDOW_SIZE` 分子目录的好处：改 `params.py` 里 `WINDOW_SIZE` 重跑 prepare_data 不会覆盖之前 window 的产物，可在不同上下文长度间快速切换。

**统一通过 [t1d_granada/utils.py](t1d_granada/utils.py) 的 `processed_data_dir(cfg)` 拿路径**——它返回 `cfg["PROCESSED_DATA_DIR"] / f"WINDOW_SIZE_{P.WINDOW_SIZE}"`。新增 / 改动消费 npy 的脚本必须走这个 helper，**不要直接拼 `cfg["PROCESSED_DATA_DIR"]`**，否则会读到错的 / 不存在的目录。

| 文件 | 形状 / 内容 |
|---|---|
| `{train,val,test}_seq.npy` | `(N, T=WINDOW_SIZE, D_seq=6)` float32 — 输入窗时序特征（默认 T=8 即 2 小时） |
| `{train,val,test}_static.npy` | `(N, D_static=12)` float32 — 患者属性 + 滚动统计 |
| `{train,val,test}_target.npy` | `(N,)` float32 — z-score 归一化的 +30 分钟血糖标量 |
| `scaler.pkl` | `Scaler` 对象，12 字段（`bg`、`Age`、10 个滚动统计），仅在 train 上 fit |
| `meta.json` | 形状、特征名、计数、配置；**单一可信来源**（看下面字段不要再 hardcode） |

按 80/10/10 per-patient chronological 划分；典型计数 train≈16.27M / val≈2.03M / test≈2.03M。

**seq 通道含义（按 `meta.json` 的 `seq_feature_names` 顺序）**：

```
[0] bg_z       # z-score 归一化血糖（按训练集统计的 mean/std）
[1] bg_diff    # 一阶差分（与上一时间点的 bg_z 之差，t=0 处补 0）
[2] tod_sin    # 一天内时刻三角编码 sin(2π·秒数/86400)
[3] tod_cos    # 一天内时刻三角编码 cos(...)
[4] dow_sin    # 星期几三角编码 sin(2π·((天数+3)%7)/7)
[5] dow_cos    # 星期几三角编码 cos(...)
```

**8 个时间点的物理偏移（相对 last_input_ts）**：`t=0` 是最早，`t=7` 是 last_input_ts。15 分钟间隔 ⇒ 偏移 `[-105min, -90min, -75min, -60min, -45min, -30min, -15min, 0]`。target 在 `last_input_ts + 30min`（跳过 +15min，索引 9 = `WINDOW_SIZE + FORECAST_STEPS - 1`）。

**static 列含义（`meta.json` 的 `static_cols` 顺序）**：

```
[0]  Sex_M                 # 性别独热（M=1 / F=0）
[1]  Age                   # z-score 归一化年龄（按训练集 fit）
[2]  bg_2h_mean            ┐
[3]  bg_2h_std             │
[4]  bg_2h_tir_70_180      │ 2 小时滚动统计 × 5 = 5 列
[5]  bg_2h_p5              │ 严格因果：窗口 [t-2h, t)，不含 t
[6]  bg_2h_p95             ┘
[7]  bg_6h_mean            ┐
[8]  bg_6h_std             │
[9]  bg_6h_tir_70_180      │ 6 小时滚动统计 × 5 = 5 列
[10] bg_6h_p5              │
[11] bg_6h_p95             ┘
```

`tir_70_180` = time-in-range，窗内血糖落在 [70, 180] mg/dL 的比例。所有滚动统计也都 z-score 归一化，但 **`bg_2h_*` / `bg_6h_*` 的 scaler 是按它们各自分布 fit 的，不是 `bg` 那个**——`bg` 的 scaler 只用来反归一化 target，预测阶段才用得到。

**target z-score ↔ mg/dL 转换**：

```python
pred_z = model(seq, static)              # (B,)
pred_mg = scaler.inverse("bg", pred_z)   # mg/dL，可读尺度，用于 RMSE/MAE/R² 报告
```

**训练时如何用**（`xLSTMRegressor.forward(seq, static)`）：

```
seq:    (B, 8, 6)  ─► Linear(6→E) ─► LayerNorm ─► xLSTMBlockStack ─► [:, -1, :] ─┐
                                                                                  ├─► concat ─► MLP ─► squeeze ─► (B,) z-score
static: (B, 12) ─────────────────────────────────────────────────────────────────┘
```

Loss 在 z-score 域（`MSELoss`），但 `_validate()` 反归一化到 mg/dL 报 RMSE / MAE / R²。

**直观查看前 N 行**：[scripts/dump_processed_head.py](scripts/dump_processed_head.py) 把 9 个 npy 各取头 N 行落到 `data/processed/WINDOW_SIZE_<N>/head20/*.csv`（列名带物理含义，例如 `t-45_bg_z`、`bg_2h_tir_70_180`、`target_mg_dL_at_+30min`）。脚本通过 `processed_data_dir()` 自动拿到当前 `WINDOW_SIZE` 的子目录。

```bash
PYTHONPATH=. python scripts/dump_processed_head.py
```

## 关键模块

| 文件 | 作用 |
|---|---|
| [t1d_granada/window_builder.py](t1d_granada/window_builder.py) | 用 `sliding_window_view` 矢量化构造 (8-step input, +30min target) 窗口 |
| [t1d_granada/rolling_stats.py](t1d_granada/rolling_stats.py) | 2h/6h 滚动统计；**`closed='left'` 严格因果**（窗口 `[t-w, t)` 排除锚行） |
| [t1d_granada/feature_engineer.py](t1d_granada/feature_engineer.py) | 派生特征（`bg_diff`、time-of-day/day-of-week 三角编码） |
| [t1d_granada/model.py](t1d_granada/model.py) | `xLSTMRegressor`：`Linear→LayerNorm→xLSTMBlockStack→last-step→concat static→MLP→squeeze` |
| [t1d_granada/trainer.py](t1d_granada/trainer.py) | 训练循环 + early stopping + Optuna 剪枝 + denormalize 评估 |
| [t1d_granada/shap_analysis.py](t1d_granada/shap_analysis.py) | `_SHAPWrapper(unsqueeze(-1))` + `GradientExplainer` 多输入列表形式 |

## Documented Solutions

[docs/solutions/](docs/solutions/) — 按类目组织（如 `design-patterns/`、`tooling-decisions/`），每个文件带 YAML frontmatter（`module`、`tags`、`problem_type`）记录本子项目已解决问题、可复用模式和决策（含 xLSTM/SHAP 集成、`slstm_backend='vanilla'` 选择、因果滚动统计契约、`numpy<2.3` 环境约束）。在文档化的领域里实现或调试时相关。

[docs/plans/](docs/plans/) — 设计与计划文档（含 `2026-05-09-001-feat-xlstm-glucose-prediction-plan.md` 上游计划）。

## 关键决策（务必遵守）

- **`slstm_backend='vanilla'`**：训练 / 推理 / SHAP 三条路径共用同一后端，避免 Triton 依赖和 SHAP autograd 冲突。详见 `docs/solutions/design-patterns/xlstm-shap-regressor-pattern-2026-05-09.md`。
- **`closed='left'` 严格因果**：所有 rolling 统计都不能包含锚行；改这块代码必须配套跑因果注入测试（在 `last_input_ts` 之后注入 9999，断言预测不变）。
- **`numpy<2.3`** 必须保留在 [requirements.txt](requirements.txt)，否则 `import shap` 会被 numba 报错挡住。
- **`from t1d_granada import utils as utils_mod`** 的 import 形式：让 `pytest.monkeypatch` 能替换 `load_settings`。直接 `from t1d_granada.utils import load_settings` 会失去可测性。
- **processed 数据按 `WINDOW_SIZE` 分子目录**：所有读 / 写 `data/processed/` 的脚本必须通过 `utils.processed_data_dir(cfg)` 拿路径，**禁止**直接 `Path(cfg["PROCESSED_DATA_DIR"])` 后再拼文件名——否则切 `WINDOW_SIZE` 时会读到错目录。test fixture 也遵循同样的子目录约定（写到 `tmp_path/data/processed/WINDOW_SIZE_<T>/`）。
