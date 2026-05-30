# T1DiabetesGranada · xLSTM 30-min Blood Glucose Prediction

> 中文版：[README.zh.md](README.zh.md)

A regression model that takes the last 2 hours of multimodal CGM data (8 samples × 15 min) from a type-1 diabetic patient and predicts blood glucose **30 minutes ahead** (mg/dL). Built on the [T1DiabetesGranada dataset](https://www.nature.com/articles/s41597-023-02737-4) and **xLSTM 2.0.5**.

> This subproject is independent from the BrisT1D LightGBM solution at the repo root — **no shared code, data, or stack**.

---

## Stack

| Component | Version / Choice | Purpose |
|---|---|---|
| Python | 3.12 | conda env `py312_cu121` |
| PyTorch | ≥ 2.4 (CUDA 12.x) | training framework |
| xlstm | **2.0.5** | sLSTM/mLSTM blocks (`slstm_backend='vanilla'`, no Triton) |
| Optuna | ≥ 3.5 | TPE + MedianPruner search, persisted to SQLite |
| MLflow | ≥ 2.10 | experiment tracking (params/metrics/artifacts) |
| SHAP | ≥ 0.45 | feature attribution via `GradientExplainer` |
| numpy | **< 2.3** | required for numba/shap compatibility, **do not upgrade** |

Full list in [requirements.txt](requirements.txt).

---

## Task Definition

| Item | Value |
|---|---|
| Input sequence length | 8 steps × 15 min = 2 hours |
| Forecast target | `bg` (mg/dL) at `last_input_ts + 30 min` |
| Sequence features (D_seq=6) | `bg_z`, `bg_diff`, `tod_sin/cos`, `dow_sin/cos` |
| Static features (D_static=12) | `Sex_M`, `Age_z` + 2h/6h rolling stats × 5 (mean/std/tir_70_180/p5/p95) |
| Training loss | MSE in z-score space |
| Validation / report metrics | RMSE / MAE / R² in mg/dL (after denormalization) |
| Splits | per-patient chronological **80 / 10 / 10** |

> Rolling stats are strictly causal: `closed='left'` ⇒ window `[t-w, t)` **excludes** the anchor row, eliminating future leakage.

---

## Project Layout

```
T1DiabetesGranada_Prediction/
├── prepare_data.py        # Stage 1: raw CSV → npy + scaler + meta
├── train_optuna.py        # Stage 2: Optuna hyperparameter search
├── train.py               # Stage 3: final fit on best_params
├── predict.py             # Stage 4: single / batch inference
├── shap_analyze.py        # Stage 5 (optional): SHAP attribution + 4 plot types
├── settings.json          # Path config (data / model / mlruns / reports)
├── requirements.txt
├── t1d_granada/           # Core package
│   ├── params.py          # ★ All tunable parameters live here ★
│   ├── window_builder.py  # Vectorized window construction
│   ├── rolling_stats.py   # Strictly causal 2h/6h rolling stats
│   ├── feature_engineer.py# Derived features (bg_diff, time-of-day, ...)
│   ├── scaler.py          # 12-field z-score scaler (fit on train only)
│   ├── dataset.py         # PyTorch Dataset / DataLoader
│   ├── model.py           # xLSTMRegressor
│   ├── trainer.py         # Training loop + early stop + tqdm
│   ├── shap_analysis.py   # SHAP wrapper + plotting
│   └── utils.py           # settings / seed / timer / make_dir
├── tests/                 # pytest unit + smoke tests
├── docs/
│   ├── plans/             # Design documents
│   └── solutions/         # Solved problems / decisions (with frontmatter)
└── scripts/
    └── dump_processed_head.py  # Dump first N rows of npy as CSVs for inspection
```

---

## Quick Start

### 1. Environment

```bash
conda create -n py312_cu121 python=3.12 -y
conda activate py312_cu121
pip install -r requirements.txt
```

> ⚠️ `numpy<2.3` is a hard constraint — otherwise `import shap` will fail due to numba conflicts.

### 2. Prepare Data

Place the two raw T1DiabetesGranada CSVs into the directory pointed to by `RAW_DATA_DIR` in `settings.json`:

```
data/
├── Glucose_measurements.csv
└── Patient_info.csv
```

```bash
cd T1DiabetesGranada_Prediction
python prepare_data.py
```

Materialized outputs (under `data/processed/`):

| File | Shape / Content |
|---|---|
| `{train,val,test}_seq.npy` | `(N, 8, 6)` float32 |
| `{train,val,test}_static.npy` | `(N, 12)` float32 |
| `{train,val,test}_target.npy` | `(N,)` float32 (z-score) |
| `scaler.pkl` | `Scaler` object (fit on train only) |
| `meta.json` | shapes, feature names, counts, config — **single source of truth** |

Typical scale: train ≈ 16.27M / val ≈ 2.03M / test ≈ 2.03M.

To inspect the physical meaning of each row:

```bash
PYTHONPATH=. python scripts/dump_processed_head.py
# → data/processed/head20/*.csv with columns like t-45_bg_z, target_mg_dL_at_+30min
```

### 3. Hyperparameter Search

```bash
CUDA_VISIBLE_DEVICES=1 python train_optuna.py
```

Outputs:

```
hyperparameter_tuning/
├── best_params.json        # Best hyperparameters (consumed by train.py)
├── best_trial_meta.json    # trial number + last_epoch + best_val_rmse
├── study.pkl               # Optuna study object
└── xlstm_search.db         # SQLite store (for dashboard replay)
```

Re-running the same command after an interruption **resumes automatically** (`load_if_exists=True`).

### 4. Final Fit + Test Evaluation

```bash
CUDA_VISIBLE_DEVICES=1 python train.py
```

Outputs:

- `model/xlstm_best.pt` — best-on-val checkpoint (contains `state_dict / hp / meta / max_epochs / stopped_at_epoch / best_val_rmse`)
- Console report of test-set RMSE / MAE / R² (mg/dL)

> Training uses `train`, monitoring uses `val` to drive early stopping (`P.PATIENCE`), and `test` is touched only once for final evaluation — zero leakage end-to-end.

### 5. Inference

```bash
# Single sample
python predict.py --history hist.csv --patient_id P001 \
                  --last_ts "2025-03-15 08:30:00" --sex M --birth_year 1985

# Batch
python predict.py --history hist.csv --input batch.csv --output out.csv
```

`hist.csv` must cover at least `[last_ts - 6h, last_ts]` of CGM data; otherwise the 6h rolling stats fail `min_periods` and the sample is skipped.

### 6. SHAP Attribution (optional)

```bash
python shap_analyze.py
# → reports/shap/{summary.csv, feature_importance.png, timestep_importance.png, ...}
```

---

## Configuration (`t1d_granada/params.py`)

**All tunable parameters live in [t1d_granada/params.py](t1d_granada/params.py) — there are no CLI flags.** Edit this file directly:

```python
# ----- Data / windowing -----
WINDOW_SIZE        = 8         # input sequence length (steps)
FORECAST_STEPS     = 2         # +30 min target offset (2 × 15-min steps)
SAMPLE_INTERVAL_MIN = 15
ROLLING_WINDOWS    = [2, 6]    # hours
ROLLING_STATS      = ["mean", "std", "tir_70_180", "p5", "p95"]

# ----- Splits -----
SPLIT_TRAIN, SPLIT_VAL = 0.8, 0.1   # test = 1 - train - val

# ----- Training -----
BATCH_SIZE     = 512
NUM_WORKERS    = 8
MAX_EPOCHS     = 20            # upper bound; actual stop decided by early stopping
PATIENCE       = 3             # stop if val_rmse fails to improve for N epochs
WEIGHT_DECAY   = 1e-5
WARMUP_RATIO   = 0.05          # cosine schedule + linear warmup
GRAD_CLIP      = 1.0

# ----- Optuna -----
N_TRIALS                = 25
OPTUNA_N_STARTUP_TRIALS = 5
OPTUNA_N_WARMUP_STEPS   = 5
STUDY_NAME              = "xlstm_search"
OPTUNA_STORAGE          = None  # None=default sqlite, "memory"=in-memory, str=custom RDB URL

# ----- train.py -----
FINAL_FIT_EPOCH_MULT       = 1.2     # max_epochs = (last_epoch+1) * MULT
FINAL_FIT_EPOCHS_OVERRIDE  = None    # int=force max_epochs, bypassing the MULT formula
FINAL_FIT_RUN_NAME         = "xlstm_final_fit"

# ----- Reproducibility -----
SEED = 42

# ----- Search space -----
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

Path configuration lives in [settings.json](settings.json): `PROCESSED_DATA_DIR / MODEL_DIR / HYPERPARAMETER_TUNING_DIR / MLFLOW_TRACKING_URI / REPORTS_DIR / SHAP_REPORTS_DIR`.

---

## Experiment Tracking

### MLflow

```bash
mlflow ui --backend-store-uri ./mlruns
# → http://localhost:5000
```

Experiment layout (nested runs):

```
xlstm_search                    ← experiment
├── xlstm_search_parent         ← parent run (study-level params + best_val_rmse_overall)
│   ├── trial_0                 ← nested run (trial hyperparameters + per-epoch curves)
│   ├── trial_1
│   └── ...
└── ...

xlstm_final_fit                 ← experiment (train.py)
└── xlstm_final_fit             ← run (test metrics + training curves + pytorch model artifact)
```

> Per-epoch curves (`train_loss / val_rmse / val_mae / val_r2 / lr / epoch_time`) live on the **trial run**, not the parent. Expand the parent in the UI to find the child trials.

### Optuna Dashboard

```bash
optuna-dashboard sqlite:///hyperparameter_tuning/xlstm_search.db
# → http://localhost:8080
```

Or in Python:

```python
import optuna
study = optuna.load_study(
    study_name="xlstm_search",
    storage="sqlite:///hyperparameter_tuning/xlstm_search.db",
)
print(study.trials_dataframe())
```

---

## Data Contract Cheat Sheet

### Sequence feature order (matches `meta.json` `seq_feature_names`)

| Index | Name | Meaning |
|---|---|---|
| 0 | `bg_z` | z-score normalized blood glucose (using train statistics) |
| 1 | `bg_diff` | first-order difference (relative to prior `bg_z`; padded 0 at t=0) |
| 2-3 | `tod_sin / cos` | time-of-day trigonometric encoding |
| 4-5 | `dow_sin / cos` | day-of-week trigonometric encoding (`(day+3) % 7` aligns Monday to 0) |

Time offsets: `t=0..7` corresponds to `[-105min, -90min, -75min, -60min, -45min, -30min, -15min, 0]` relative to `last_input_ts`; target lives at `last_input_ts + 30min`.

### Static feature order (matches `meta.json` `static_cols`)

```
[0]   Sex_M                     # one-hot (M=1, F=0)
[1]   Age_z                     # z-score normalized
[2-6]  bg_2h_{mean,std,tir_70_180,p5,p95}    # 2-hour rolling stats
[7-11] bg_6h_{mean,std,tir_70_180,p5,p95}    # 6-hour rolling stats
```

Each rolling-stat field is normalized using its **own distribution** — **not** the `bg` scaler.

### z-score ↔ mg/dL Conversion

```python
pred_z = model(seq, static)              # (B,)
pred_mg = scaler.inverse("bg", pred_z)   # mg/dL, human-readable scale
```

---

## Key Decisions (Do Not Break)

- **`slstm_backend='vanilla'`**: training, inference, and SHAP all share the same backend, avoiding Triton dependencies and SHAP autograd conflicts. See [docs/solutions/design-patterns/xlstm-shap-regressor-pattern-2026-05-09.md](docs/solutions/design-patterns/xlstm-shap-regressor-pattern-2026-05-09.md).
- **`closed='left'` strict causality**: rolling stats must never include the anchor row. Any change here must be paired with a causality-injection test (inject 9999 *after* `last_input_ts` and assert prediction is unchanged).
- **`numpy<2.3`** must stay pinned in [requirements.txt](requirements.txt) — `import shap` will be blocked by numba otherwise.
- **`from t1d_granada import utils as utils_mod`** import style: required so `pytest.monkeypatch` can swap out `load_settings`. `from t1d_granada.utils import load_settings` would lose testability.
- **`train.py` does not merge train+val**: merging leaves no held-out monitor for early stopping, and using test as monitor leaks. Keeping `val` separate for early stopping is the safest design.

---

## Tests

```bash
pytest tests/ -q
```

Coverage:
- Unit tests (`test_dataset.py / test_feature_engineer.py / test_model.py / test_rolling_stats.py / test_scaler.py / test_shap_analysis.py / test_trainer.py / test_window_builder.py`)
- End-to-end smoke (`test_train_smoke.py / test_predict.py`)

---

## Further Reading

- [README.zh.md](README.zh.md) — Chinese version of this document
- [CLAUDE.md](CLAUDE.md) — guidance for Claude Code (data schema, module map, key decisions)
- [docs/plans/](docs/plans/) — design and planning documents
- [docs/solutions/](docs/solutions/) — categorized solved problems / decisions (xLSTM/SHAP integration, `slstm_backend='vanilla'` rationale, causal rolling stats contract, `numpy<2.3` constraint, etc.)

## Citations

- T1DiabetesGranada dataset: Rodríguez-Rodríguez, I. et al. *Sci Data* 10, 916 (2023). <https://www.nature.com/articles/s41597-023-02737-4>
- xLSTM: Beck, M. et al. *NeurIPS* (2024). <https://arxiv.org/abs/2405.04517>
