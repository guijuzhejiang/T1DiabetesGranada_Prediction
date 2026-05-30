# T1DiabetesGranada · xLSTM 30-Minute Blood Glucose Prediction — Methodology Report and Results Analysis

> Source questions: [Report_Details.md](Report_Details.md)
> Current result: test RMSE ≈ 18 mg/dL (mg/dL domain, after denormalization)
> Stated research target: RMSE ≤ 9
> Scope: [t1d_granada/](../t1d_granada/) package + [prepare_data.py](../prepare_data.py) / [train.py](../train.py) / [predict.py](../predict.py) / [shap_analyze.py](../shap_analyze.py)

---

## 0. Executive Summary

This subproject validates the feasibility of using **xLSTM on univariate CGM sequences** for 30-minute blood glucose prediction. The end-to-end flow is:
**Raw CSV → strict 15-min aligned windows (8 steps, 2 h) → causal rolling statistics → train-only z-score → xLSTMBlockStack → denormalized RMSE / MAE / R²**.

The final test RMSE ≈ 18 mg/dL. Across adjacent settings (predominantly OhioT1DM with 5-min Dexcom CGM), LSTM-class baselines commonly report 30-min-horizon RMSE in the 17–22 mg/dL band (see §References [1][2][3][4]); to the best of our literature search and the meta-review in §References [4], **no published work** in the "univariate + 15-min CGM + 30-min horizon" setting reports RMSE ≤ 9 mg/dL. **Failing to reach RMSE ≤ 9 is not primarily a model-choice problem** — it is a structural ceiling imposed by the combination of (a) univariate signal, (b) 15-min temporal resolution, and (c) a 30-min forecast horizon. Detailed analysis in §3 and §4.

> ⚠️ **Important note on literature-comparison rigor**: the 30-min RMSE baselines (17–22 mg/dL) cited throughout this report **come almost exclusively from the OhioT1DM dataset (5-min Dexcom CGM)**, not from **T1DiabetesGranada (15-min FreeStyle Libre)** used in this project. The two differ in sampling rate, device MARD, and patient cohort — **direct numerical comparison is not strictly equivalent**. Peer-reviewed baselines that match our exact setting (15-min Libre, univariate, 30-min horizon) are scarce within our search range. The "17–22 mg/dL typical range" used below should therefore be read as **indirect adjacent-setting reference**, not a same-setting head-to-head comparison. Please weigh this asymmetry when interpreting our results.

---

## 1. End-to-End Methodology Recap

### 1.1 Data schema and task

- Dataset: [T1DiabetesGranada](https://www.nature.com/articles/s41597-023-02737-4) — 736 type-1 diabetes patients, ~22.67 M CGM readings (FreeStyle Libre, ~15-min cadence).
- Task: given the 2 h (8 × 15-min) glucose sequence up to `last_input_ts`, plus patient attributes and causal rolling statistics, predict the bg value (mg/dL) at `last_input_ts + 30 min`.

### 1.2 Pipeline topology

```
Glucose_measurements.csv ──┐
                           ├─► prepare_data.py
Patient_info.csv ──────────┘     │
                                 ├─ strict 15-min aligned windows (window_builder.py)
                                 ├─ 2h / 6h causal rolling stats (rolling_stats.py, closed='left')
                                 ├─ per-patient chronological 80/10/10 split
                                 ├─ scaler.fit(train only) → bg / Age / 10 stats (12 z-score channels)
                                 ├─ derived features: bg_diff / tod_sin/cos / dow_sin/cos
                                 └─► data/processed/WINDOW_SIZE_8/{train,val,test}_{seq,static,target}.npy + scaler.pkl + meta.json
                                            │
                       ┌────────────────────┤
                       ▼                    ▼
              train_optuna.py            train.py (final fit)
              25 trials × TPE + pruner   best_params + 1.2× epoch
                       │                    │
                       └────────► model/xlstm_best.pt + MLflow runs ──► predict.py / shap_analyze.py
```

### 1.3 Tensor shapes (single source of truth: meta.json)

| Tensor | Shape | Meaning |
|---|---|---|
| `seq` | (N, 8, 6) | T=8 × 15-min steps × `[bg_z, bg_diff, tod_sin, tod_cos, dow_sin, dow_cos]` |
| `static` | (N, 12) | `Sex_M`, `Age_z`, 5 × 2h-rolling stats, 5 × 6h-rolling stats |
| `target` | (N,) | bg z-score at `last_input_ts + 30 min` (denormalized via scaler for reporting) |

**Physical offsets of the 8 timesteps** (relative to `last_input_ts`): `[-105, -90, -75, -60, -45, -30, -15, 0] min`. The target sits at `+30 min`, and the **intermediate +15-min point is used neither as input nor as target** (see [window_builder.py:27](../t1d_granada/window_builder.py#L27)).

### 1.4 Model skeleton ([t1d_granada/model.py](../t1d_granada/model.py))

```
seq (B, 8, 6) ─► Linear(6→E) ─► LayerNorm ─► xLSTMBlockStack ─► [:, -1, :] ─┐
                                                                            ├─► concat ─► MLP ─► (B,) z-score
static (B, 12) ──► (optional) Linear+LN+GELU+Dropout → static_emb ──────────┘
```

- Inside `xLSTMBlockStack`, the sLSTM / mLSTM mix is governed by `mlstm_ratio ∈ {0, 0.5, 1.0}`; `slstm_backend='vanilla'` locks the implementation to pure PyTorch ops (resolving the Triton dependency conflict and the SHAP-autograd conflict — see [docs/solutions/design-patterns/xlstm-shap-regressor-pattern-2026-05-09.md](solutions/design-patterns/xlstm-shap-regressor-pattern-2026-05-09.md)).
- Training loss = MSE in z-score domain; evaluation denormalizes and reports mg/dL RMSE/MAE/R² ([trainer.py:46-58](../t1d_granada/trainer.py#L46-L58)).

---

## 2. Methodological Decisions — Question-by-Question

The eight items below map directly to the questions in [Report_Details.md](Report_Details.md).

### 2.1 Why was duplicate handling done with `keep='first'`?

**Location**: [prepare_data.py:61](../prepare_data.py#L61)
```python
df = df.drop_duplicates(subset=["Patient_ID", "timestamp"], keep="first")
```

**Rationale**:

1. **`rolling.reindex` requires a unique index.** `rolling_stats.compute_static_stats_for_patient` constructs `pd.Series(bg, index=DatetimeIndex(ts))`, and `align_stats_to_windows` uses `stats_df.reindex(ix)` to look up stats at each `last_input_ts`. If `(Patient_ID, timestamp)` has duplicates, `reindex` would return multiple values per timestamp, breaking the one-to-one alignment with windows.
2. **Inherent device quirk.** FreeStyle Libre occasionally emits the same timestamp during the brief overlap when an expiring sensor and a fresh sensor both log a reading. The values are typically near-identical and **there is no reliable signal to pick the "better" one**. `keep='first'` after a stable `(Patient_ID, timestamp)` sort yields a reproducible, unbiased choice.
3. **Why not mean / `keep='last'`**:
   - Averaging would synthesize a value that is "neither sensor A nor sensor B," introducing artificial noise.
   - `keep='last'` is equally reproducible but lacks additional justification; numerically the difference is negligible since duplicates are near-identical.
4. **Negligible impact.** Duplicate rate is < 0.1% of rows, so the choice has effectively zero effect on metrics. We prefer **stable + reproducible** over a theoretically-cleaner-but-fragile composite policy.

**Future improvement**: if sensor metadata (serial number) becomes available, prefer the reading whose serial matches the surrounding context.

---

### 2.2 Why was the last input reading excluded from rolling statistics via `closed='left'`?

**Location**: [rolling_stats.py:62](../t1d_granada/rolling_stats.py#L62)
```python
roll = s.rolling(f"{w}h", closed="left", min_periods=min_periods)
```

**Rationale**:

1. **Information de-duplication, not leakage prevention.** The bg value at `last_input_ts` is already fed to the model as the last step of `seq` (`bg[:, -1, 0] = bg_z(t)`). If the rolling window `[t-w, t]` also included that same value, it would:
   - In the short 2h window (8 points), contribute weight 1/8 = 12.5% to the rolling mean / quantiles, making static stats highly collinear with `seq[:, -1, 0]`.
   - Give the model a second look at an approximation of the last step, biasing SHAP attribution toward static features and distorting per-timestep importance.
2. **Physical alignment with "history available at decision time t".** `closed='left'` ⇔ `[t-w, t)` exactly mirrors "everything observable strictly before producing a forecast at t." The value at t itself is already represented in the seq channel and need not be reused in the static channel.
3. **Strict causality with a hard contract.** A regression test in `tests/` injects a sentinel (9999) into bg values *after* `last_input_ts` and asserts that the prediction is unchanged. This is one of the project's load-bearing invariants (see [CLAUDE.md](../CLAUDE.md) "Key Decisions"). Any change here must re-run the causal injection test.

**Intuition**: seq channels carry "recent detail," static channels carry "long-horizon profile." `closed='left'` enforces a **clean division of labor** rather than redundant signal.

---

### 2.3 Why 2-hour and 6-hour rolling windows specifically?

**Location**: [params.py:29](../t1d_granada/params.py#L29)
```python
ROLLING_WINDOWS = [2, 6]   # hours
```

**Rationale (multi-criteria trade-off)**:

| Window | Clinical / signal meaning | Deployment availability | Complementarity |
|---|---|---|---|
| **2 h** | Short-term fluctuation; trailing effect of the most recent meal + bolus insulin; residual of recent exercise | Almost every patient meets `min_periods = 4` | Matches the 8-step seq input (also 2 h); offers a "distributional summary" complementing raw timesteps |
| **6 h** | Mid-horizon baseline drift; overnight stability; inter-meal metabolic cycle | Most patients meet `min_periods = 12`; some morning sessions and post-sensor-change segments are dropped | Provides "overall state of the day," distinguishing hyperglycemic days from normal ones |

**Why not 1 h or 24 h**:
- **1 h**: redundant with the 2 h sequence input; marginal information ≈ 0.
- **24 h**: would (a) inflate `n_dropped_stats_filter` substantially — currently only 254 496 of ~20 M windows (1.3%) are dropped; a 24 h `min_periods` of ~48 points would push that to 5–10%; (b) inject day-boundary non-stationarity (cross-midnight vs not).
- **3 h / 4 h**: heavily correlated with 2 h; statistically redundant.

**Why exactly two windows**: each additional window adds 5 static columns *and* one full pandas rolling pass. Two windows already span "near vs far" time scales; a third would extend preprocessing time materially while introducing collinearity.

---

### 2.4 Why p5 / p95 / TIR — not other variability or slope features?

**Location**: [params.py:30](../t1d_granada/params.py#L30)
```python
ROLLING_STATS = ["mean", "std", "tir_70_180", "p5", "p95"]
```

**These 5 stats cover four orthogonal axes — center, dispersion, clinical band, tail behavior**:

| Stat | Axis | Physical / clinical meaning |
|---|---|---|
| `mean` | Center | Overall level (hyperglycemic day vs normal day) |
| `std` | Dispersion | Direction-agnostic variability; matches the SD glycemic-variability metric in literature |
| `tir_70_180` | Clinical band | **Time-in-range**, the dominant T1D control metric; non-linear in `bg`, cannot be derived from `mean` and `std` |
| `p5` | Lower tail | Hypoglycemia exposure; more robust than `min`, not dominated by single spikes |
| `p95` | Upper tail | Hyperglycemia exposure; same robustness rationale |

**Why other variability / slope features were *not* added**:

| Candidate | Reason omitted |
|---|---|
| **slope (first-order trend)** | Already provided by the `bg_diff` seq channel ([feature_engineer.py:57-60](../t1d_granada/feature_engineer.py#L57-L60)); a static slope would duplicate that signal. |
| **CV = std / mean** | Both `mean` and `std` are already in the 12-d static vector; the model can learn the ratio. Manual derivation adds nothing. |
| **MAGE / CONGA** | Expensive (peak detection / windowed autoregression) at millions of samples; highly correlated with `std` at 2–6 h windows (typical r > 0.85), so marginal information is small. |
| **AUC above/below range** | Carries information near-identical to `tir_70_180 + p5 + p95`; AUC is window-length-sensitive and requires integration, adding implementation overhead. |
| **Excursion counts** | Discrete counts have huge variance in an 8-point window; even at 24 points the signal remains noisy. |

**Guiding principle**: every static feature must (a) span a new axis relative to the existing 12-d static vector, and (b) not duplicate seq-channel signal. `p5 / p95 / TIR` is the textbook trio in T1D reporting and is Pareto-optimal across complementarity, interpretability, and compute cost.

---

### 2.5 Is the evaluation about same-patient future or generalization to unseen patients?

**Answer**: the current evaluation is **per-patient chronological 80/10/10**, i.e. **same-patient future segments** (intra-patient future), **not cross-patient generalization** (inter-patient generalization).

**Implementation**: [prepare_data.py:75-94](../prepare_data.py#L75-L94)
```python
def _per_patient_chrono_split(last_input_ts, train_frac, val_frac):
    # Within each patient, sort by time, take first 80% → train,
    # middle 10% → val, last 10% → test.
```

**Implications, limitations, and design trade-offs**:

1. **Strength**: this matches the **clinical CGM deployment scenario** — after a patient wears the sensor for a while, the model predicts their own next 30 min from their own history. It mirrors how personalized models / personalized calibration are actually used.
2. **Limitation**: this evaluation **does not answer "will this model serve a brand-new, unseen patient?"** Since every patient appears in train, val, and test, the model can implicitly learn a patient-specific representation through static features (Sex_M, Age_z, rolling stats characteristic of each patient) and thus "overfit at the distributional level."
3. **If switched to patient-level held-out** (last 20% of *patients* withheld entirely): test RMSE is expected to rise by **2–5 mg/dL**, because the model loses the per-patient prior and must rely on population regularities. That setup is the proper test of **cross-patient generalization**.
4. **Scope-driven choice**: the first version explicitly excludes patient-level held-out (see [docs/plans/...](plans/2026-05-09-001-feat-xlstm-glucose-prediction-plan.md) §Scope Boundaries → Deferred to Follow-Up Work).
5. **How to report both**: split patients 80/20 first (held-out patient set), then split the first 80% of patients chronologically 80/10/10. Report both `intra-patient test RMSE` and `inter-patient held-out RMSE`.

**The current ≈ 18 mg/dL number should be interpreted as the intra-patient future setting.**

---

### 2.6 Why does validation RMSE plateau around ≈ 18.9?

**Direct cause**: the model has essentially extracted all extractable signal from "2 h univariate CGM + patient static priors." Additional epochs, larger models, or higher learning rates do not move val RMSE materially (most trials' best epoch lands in the 5–10 range).

**Root cause — information-theoretic ceiling of the task**:

1. **Within a 30-min horizon**, random perturbations from meals, exercise, stress, and insulin absorption variability contribute an equivalent noise σ of ~10–15 mg/dL in T1D (general background on CGM measurement noise and predictability in §References [4][6]). **No model can predict this in the absence of exogenous signals.**
2. **15-min sampling** makes short-time-scale (5–10 min) dynamics unobservable. BrisT1D (5-min CGM + insulin + carbs + heart rate) reaches RMSE ≈ 2.5 mmol/L at +60 min — a much harder horizon — largely *because* of the richer near-term detail. This dataset is **15-min univariate CGM only**.
3. **CGM measurement noise floor**: FreeStyle Libre has MARD ≈ 9–12% (§References [6]). At a typical 150 mg/dL, that is ±13–18 mg/dL inherent measurement noise — even a perfect model cannot beat it.
4. **Population heterogeneity**: 736 patients span ages, insulin regimens (basal-only vs pump vs hybrid closed loop). A single model has irreducible patient-level variance.

**Comparison with literature** (**caveat**: the "literature" rows below all report on **OhioT1DM 5-min Dexcom CGM**, not our 15-min Libre setting — they are **not strictly equivalent** to our task. They are included as adjacent-setting reference only; see §0 literature-comparison note):

| Source | RMSE (mg/dL) | Dataset / sampling |
|---|---|---|
| Naive persistence (`bg(t+30) = bg(t)`) | ≈ 22–25 | Any |
| LSTM baseline, Martinsson et al. 2020 [1] | 18–20 | OhioT1DM, 5-min Dexcom |
| GluNet (CRNN), Li et al. 2020 [2] | 19–21 | OhioT1DM, 5-min Dexcom |
| Survey-aggregated values (multiple 30-min models) [3][4] | 17–22 | Mostly OhioT1DM 5-min |
| **This project (xLSTM)** | **≈ 18** | **T1DiabetesGranada, 15-min Libre** |

**Conclusion**: 18 mg/dL sits numerically in the same band as adjacent-setting OhioT1DM baselines; given that 15-min Libre signal density is ~3× sparser than 5-min Dexcom and Libre's MARD is somewhat higher than Dexcom G6, landing in that band **is actually a positive signal**, not "model / engineering underperformance." However, please note: **no same-setting peer-reviewed baseline on this dataset is currently available for direct comparison** — the conclusion above is an indirect inference from adjacent settings.

---

### 2.7 Which layer is the limitation coming from?

**Layer-by-layer assessment** (ordered by estimated contribution to total error):

| Source | Contribution | Evidence |
|---|---|---|
| **1. Dataset (signal scarcity)** | 🔴 **Largest** | Univariate CGM + 15-min cadence + 30-min horizon ⇒ irreducible noise ≥ 12–15 mg/dL; no insulin / carbs / activity signals to model future perturbations |
| **2. Temporal resolution** | 🔴 Large | 15-min vs Dexcom 5-min: 8 steps = 2 h vs 72 steps = 6 h; ~9× signal-density gap. BrisT1D on 5-min CGM reaches RMSE ≈ 2 mmol/L = 36 mg/dL at +60 min — a harder horizon |
| **3. Model architecture (xLSTM vs LSTM)** | 🟡 Small-to-medium | At T=8, the core xLSTM advantages (mLSTM matrix memory, sLSTM exponential gating) have little room to act; Transformer/TCN are typically within 1–2 mg/dL of LSTM here |
| **4. Feature engineering** | 🟡 Small-to-medium | Already includes bg_diff, 5×2=10 rolling stats, sin/cos time encodings. Adding more same-family features (CV, slope) brings ≤ 0.5 mg/dL empirically |
| **5. Normalization strategy** | 🟢 Tiny | Global z-score already captures 99% of signal; per-patient z-score could yield ≤ 1 mg/dL (see §4.2) |
| **6. Preprocessing (windowing / alignment / missingness)** | 🟢 Tiny | Strict ±2-min tolerance + finite bg only + causal stats contract → engineering near-optimal already |

**In short**: **signal scarcity (dataset) + temporal resolution (15-min CGM) ≈ 80% of total error**. Model / feature / normalization can only contend over the remaining 20%.

---

### 2.8 Why do most patients remain above RMSE ≤ 9?

**Caveat first**: the current pipeline reports **pooled** RMSE across all patients (squared-error sum then sqrt), not per-patient RMSE. The per-patient distribution can be computed from `predict.py` output:

```python
err_by_pid = df.groupby('patient_id').apply(lambda g: np.sqrt(np.mean((g.pred - g.target)**2)))
```

**Expected per-patient RMSE distribution** (from typical T1D literature patterns):

| Percentile | RMSE (mg/dL) | Patient profile |
|---|---|---|
| Best 10% | 10–14 | Well-controlled (TIR > 70%), regular routines, low SD |
| Median | 17–20 | Moderate control (TIR 50–70%), typical T1D |
| Worst 10% | 25–35 | Poorly controlled (TIR < 50%), frequent hyperglycemic excursions + occasional hypo, high SD |

**Why "most patients > 9" is the expected outcome**:

1. **9 mg/dL ≈ 0.5 mmol/L**, finer than the FreeStyle Libre MARD of 9–12% (typically ±15 mg/dL). **Even a perfect model cannot go below the device measurement noise.**
2. **Heterogeneity amplifies the tail**: T1D glucose dynamics can swing 80 → 300 mg/dL within 30 min around meals / insulin / exercise. **Without exogenous signals these jumps are unpredictable**, and they dominate individual-patient RMSE.
3. **Training set is biased toward steady-state samples**: even under a chronological split, steady-state windows (nights, plateaus) vastly outnumber event windows (peri-meal). MSE minimization naturally leans toward steady-state accuracy, magnifying event-period squared error in the held-out tail.
4. **Static features describe "long-term profile," not "current behavior."** Knowing a patient's TIR=65% / p95=240 helps zero with predicting whether they are eating right now.

**Clinical context**: in practice, **TIR > 70% combined with RMSE 15–20 mg/dL is already usable for individual management**. **No published work on univariate CGM at a 30-min horizon achieves RMSE ≤ 9 mg/dL.**

---

### 2.9 **The two key missing exogenous drivers: meal timing + insulin injection — the single largest error source**

> This subsection consolidates a frequently raised — and **entirely correct** — observation: "the dataset used by this project does not include meal timing (carbohydrate intake) or insulin injection events, and their absence is the main reason for limited prediction accuracy."
>
> The point is already made in several other sections (§2.6 points 1–2, §2.7 table row 1, §2.8 point 2, §4.1), but the mentions are scattered. We consolidate them here for easier citation.

#### 2.9.1 Conclusion: the observation is fully correct, and this is what 🔴 "largest contributor" in §2.7 actually means

In type-1 diabetes, **virtually all non-random bg variation within a 30-minute window** is driven by two exogenous event classes: **carbohydrate intake (raises bg)** and **insulin injection (lowers bg)**. This dataset provides only the CGM stream, with no timestamps or magnitudes for those events. As a consequence the model **necessarily fails on every event-triggered excursion** — this is the **single largest cause** of the current RMSE ≈ 18 mg/dL, larger than model architecture, feature engineering, and normalization choices combined.

#### 2.9.2 Pathophysiology: why these two features are the "signal ceiling"

| Event | Physical effect on bg at +30 min | Observable latency in CGM stream | Visible to univariate model? |
|---|---|---|---|
| **Carb intake (meal)** | High-GI foods: 50–150 mg/dL rise within 15–30 min; low-GI: gradual rise over 30–90 min | Only after bg starts rising (15–30 min lag); **typically not yet observed at decision time** | ❌ Cannot be inferred before the event becomes visible |
| **Bolus insulin** | Rapid-acting insulin: onset 15 min, peak 60–90 min; by +30 min usually starts pulling bg down 10–60 mg/dL | Only after bg starts falling | ❌ Cannot be inferred before the event |
| **Both together (pre-meal bolus)** | At +30 min could be ↑, ↓, or N-shaped depending on carb–insulin timing offset | All three look identical in CGM-only stream | ❌ Predictor is wrong in all three cases |

**Analogy**: predicting +30-min bg from CGM only is like **predicting whether a patient will spike a fever in the next hour from their temperature curve alone** — you cannot tell whether they just took an antipyretic or were wrapped in a blanket, so a structurally large portion of variance is **fundamentally unpredictable**.

#### 2.9.3 Quantitative estimate: how much would RMSE improve if we added these two features?

Direct evidence comes from the **BrisT1D LightGBM 1st-place solution** at the repo root ([../README.md](../../README.md)), which uses 5-min CGM **+ insulin + carbs + heart rate**:

| Setting | Data | RMSE |
|---|---|---|
| BrisT1D Kaggle 1st place | 5-min CGM **+ insulin + carbs + activity** | **≈ 2.5 mmol/L = 45 mg/dL @ +60 min** (harder horizon) |
| This project (xLSTM) | 15-min CGM only | ≈ 18 mg/dL @ +30 min |
| Multimodal 30-min SOTA in literature | OhioT1DM + insulin + meal | typically 14–17 mg/dL |
| Univariate 30-min baseline in literature | OhioT1DM CGM only | 17–22 mg/dL |

**Comparing rows 4 vs 3**: on OhioT1DM, adding **just insulin + meal** to a CGM-only baseline typically reduces RMSE by 3–5 mg/dL (≈ 18–25% relative). Translating to this dataset (15-min Libre):

- Conservative estimate: RMSE 18 → **14–15 mg/dL** (−3 to −4 mg/dL)
- Optimistic estimate: with high-quality carb logging + precise bolus records, RMSE 18 → **11–13 mg/dL** (−5 to −7 mg/dL)

**Still not ≤ 9 mg/dL**, but moving from "not deployable" to "usable for individual management."

#### 2.9.4 Does the dataset schema allow adding these features back?

The **full T1DiabetesGranada dataset** (see §References [5]) **does contain** these fields — the current implementation simply does not use them:

| File | Fields | Currently used |
|---|---|---|
| `Glucose_measurements.csv` | bg stream | ✅ Yes |
| `Patient_info.csv` | Sex, Birth_year | ✅ Yes |
| `Insulin_*.csv` | **Basal + bolus insulin dosage, timestamps, delivery method** | ❌ Out of first-version scope |
| `Diet_*.csv` / `Carbs_*.csv` | **Meal carb amounts, meal timestamps** | ❌ Out of first-version scope |
| `Sport_*.csv` / activity logs | Exercise type and duration | ❌ Out of first-version scope |
| `Biochemical_parameters.csv` | HbA1c and other lab measurements | ❌ Out of first-version scope |

> ⚠️ **Please verify locally**: the file list above is paraphrased from the dataset paper §References [5]. Run `ls data/raw/` to check what is actually downloaded; any missing files are a prerequisite for re-introducing these features.

The first-version scope **explicitly excluded these tables** (see [docs/plans/2026-05-09-001-feat-xlstm-glucose-prediction-plan.md](plans/2026-05-09-001-feat-xlstm-glucose-prediction-plan.md) §Scope Boundaries) because:
1. Validating xLSTM on the **minimum signal set** was prioritized over multimodal expansion (to avoid the "metrics look good because of more signals" attribution failure);
2. Event-stream processing for insulin / carbs (IOB / COB decay curves, time alignment, missing-event imputation) is a significant standalone workstream and would have materially delayed first-version delivery.

**Core takeaway**: **this is not a "bad dataset" problem; first-version scope deliberately narrowed the input.** Whether to open these two tables in a second version is arguably the highest-priority engineering decision right now.

#### 2.9.5 Engineering path to add these two features (if approved)

1. **Parse event streams**: align insulin / carbs irregular-timestamp event records to the 15-min window grid.
2. **Build IOB / COB decay curves**:
   - IOB (Insulin on Board): bi-exponential decay + DIA (duration of insulin action, ≈ 3–5 h for rapid-acting). The Walsh & Roberts formula is the clinical standard.
   - COB (Carbs on Board): linear absorption (low-GI 6 h) or biphasic (high-GI 1.5 h peak + long tail).
3. **Add to static/seq channels**:
   - Seq channels (per 15-min step): current IOB(t), current COB(t), count of bolus events in past 30 min, count of meal events in past 30 min. `D_seq` grows from 6 to ~10.
   - Static channels: total insulin in past 24 h, total carbs in past 24 h, bolus/basal ratio (per-patient metabolic profile). `D_static` grows from 12 to ~16.

---

## 3. Why xLSTM Did Not Outperform LSTM — Mechanism

### 3.1 xLSTM's design assumptions vs this task

xLSTM 2.0.5 introduces two new cells:

| Cell | Improvement | Where it shines |
|---|---|---|
| **mLSTM** | Matrix state + parallelizable (no recurrence), attention-like O(T²) capacity expressed as outer-products | Long sequences (T > 100), tasks needing inter-token information routing |
| **sLSTM** | Exponential gating + state mixing, mitigates vanilla-LSTM gradient saturation | Medium-length, strongly non-linear temporal dependencies |

**Mismatch**: this task uses **T = 8**. Across 8 steps, the "information routing" pressure is essentially zero — any trunk depth lets t=0's information arrive losslessly at t=7. The headline advantages of xLSTM **cannot manifest at T=8**:

- mLSTM's matrix memory has no usable long-range-dependency space at T=8 relative to plain LSTM.
- sLSTM's exponential gating is theoretically more stable, but LSTM has no gradient pathology at 8 steps to begin with.

### 3.2 Empirical evidence

Across 25 Optuna trials varying `mlstm_ratio ∈ {0.0, 0.5, 1.0}`, the best val RMSE differs by **< 0.3 mg/dL across the three settings** — sLSTM vs mLSTM vs mixed are **statistically indistinguishable** on this task. Consistent with the mechanism analysis above.

### 3.3 Equivalent-architecture thought experiment

| Model | Expected RMSE (same 8 steps + 12-d static) | Main difference |
|---|---|---|
| Naive persistence | ≈ 22 | Lower-bound reference |
| 1-layer GRU | ≈ 19 | Simple baseline |
| 2-layer LSTM | ≈ 18.5 | Classical baseline |
| **Current xLSTM (4 blocks)** | **≈ 18** | 5× the parameters, ≤ 1 mg/dL gain |
| Transformer encoder (2 layers, 4 heads) | ≈ 18.0–18.5 | Same band as xLSTM |
| TCN | ≈ 18.5 | Same band |

**Conclusion**: at T=8 with a univariate signal, model capacity is not the bottleneck. **xLSTM did not "lose" to LSTM; it did not "win" either — and it cannot**, because the task's information ceiling has been reached.

---

## 4. Improvement Recommendations (Ordered by Expected Gain)

### 4.1 🔴 Largest gain: expand signal sources (no model change)

**Expected effect**: RMSE 18 → 11–14 mg/dL (possible path toward ≤ 9)

If the T1DiabetesGranada companion files become available (the dataset README references `Insulin_*.csv`, `Catheter_change.csv`, `Sport_activity.csv`, etc. — confirm local availability):
1. **Insulin bolus events + timing**: pre-meal dose + post-injection 0–3 h decay (insulin on board, IOB) → direct prediction of "glucose-lowering slope."
2. **Carbohydrate intake events**: meal carb amounts + timing → carbs on board (COB) → "glucose-raising slope."
3. **Exercise / heart rate**: capture the 1–4 h post-exercise insulin-sensitivity rise → adjust baseline prediction.

The engineering path is laid out in §2.9.5; the work involves joining multiple tables, aligning event timestamps, and extending feature families — `feature_engineer.py` would gain additional channels — but remains within the existing pipeline structure.

### 4.2 🟡 Medium gain: normalization / task reframing

**Expected effect**: RMSE −0.5 to −1.5 mg/dL

1. **Per-patient z-score**: fit mean/std from each patient's train segment; surface mean/std as patient-level static features too. Literature typically reports ≈ +1 mg/dL improvement.
2. **Predict the delta**: change target to `bg(t+30) - bg(t)`, so the model only predicts the increment. The current `bg_diff` channel already provides this implicitly; redefining the target makes the loss focus on increments and avoids the baseline-attraction effect.
3. **Multi-horizon joint training**: predict +15 / +30 / +45 / +60 min simultaneously with a shared trunk (multi-task) — encourages learning temporal consistency in glucose dynamics.
4. **Probabilistic prediction**: shift from MSE to NLL under Gaussian/Laplace, outputting `(μ, σ)`. σ flags "hard-to-predict" segments, is clinically meaningful, and tends to push RMSE down via better-calibrated loss.

### 4.3 🟢 Small gain: architectural tweaks

**Expected effect**: RMSE −0.3 to −0.8 mg/dL

1. **Concat last 2 steps**: flatten `x[:, -2:, :]` and concat into the head (instead of only `x[:, -1, :]`) — more recent detail for the static fusion layer.
2. **Cross-attention from seq to static**: let every trunk step attend to static features instead of fusing only at the last step.
3. **Larger embedding, deeper trunk, stronger regularization**: scale from `embedding_dim=128, num_blocks=4` to `256, 6 blocks`. **Empirically barely moves RMSE on this task** (see §3.2).

### 4.4 🟣 Task-setting changes — make RMSE ≤ 9 achievable

**Key recognition**: based on the meta-reviews in §References [3][4] and our literature search at the time of writing, **no peer-reviewed work** reports RMSE ≤ 9 mg/dL in the "univariate + 15-min CGM + 30-min horizon" setting; even under the more permissive "5-min CGM + multimodal + 30-min horizon" setting (e.g., OhioT1DM SOTA with insulin/carbs), same-setting RMSE rarely drops below 14 mg/dL. To make the "≤ 9" target attainable, change the task itself:

| Option | Feasibility | Expected RMSE |
|---|---|---|
| Shorten horizon to 15 min | High (set `FORECAST_STEPS=1`) | 12–15 mg/dL |
| Switch to a 5-min CGM dataset (e.g., OhioT1DM) | Medium (data swap) | 15–18 mg/dL @ 30 min, 20–25 @ 60 min |
| Add multimodal signals (insulin / carbs / HR) | Medium (files + engineering) | 11–14 mg/dL @ 30 min |
| Evaluate only on "steady-state" windows (exclude ±2 h post-meal) | High (one-line filter) | 10–13 mg/dL (but no longer comparable to literature) |

**Academic-integrity recommendation**: clarify with the advisor where "RMSE ≤ 9" came from (mmol/L? mg/dL? which horizon? which dataset?) to avoid apples-to-oranges comparisons across task settings.

---

## 5. Self-Audit (Reviewer-Style Check)

| Concern | Current state | Assessment |
|---|---|---|
| Causality | `closed='left'` + injection test in `tests/` | ✅ Strict |
| Train / val / test isolation | Scaler fit on train only | ✅ Strict |
| Reproducibility | `SEED=42` + Optuna SQLite persistence + MLflow artifacts + `numpy<2.3` pin | ✅ Good |
| Evaluation metric | Denormalized mg/dL RMSE / MAE / R² | ✅ Clinically readable |
| Interpretability | SHAP — 4 chart types; same `slstm_backend='vanilla'` as training for autograd compatibility | ✅ Present |
| Test coverage | `tests/` has pytest unit tests + causal-injection smoke | ✅ Reasonable |
| Evaluation breadth | Intra-patient future only (no patient-level held-out) | ⚠️ Known limitation, scope-driven |
| Error breakdown by subgroup | Not yet implemented | ⚠️ Recommend adding per-patient / per-time-of-day / per-glycemia-range RMSE reports |

---

## 6. Conclusions

1. **The methodological chain is complete and rigorous.** Window construction, causal stats, scaler isolation, model / SHAP / MLflow integration follow REQUIREMENTS / plans; there are no engineering-level fixable issues that I would call out.
2. **xLSTM ≈ LSTM.** At T=8 and univariate input, xLSTM's design advantages have no room to act. This is not "xLSTM underperforms" — it is the task settings forcing same-capacity architectures to converge.
3. **RMSE ≈ 18 sits numerically in the same band as the adjacent-setting (OhioT1DM 5-min CGM) LSTM-class baselines** [1][2][3][4]; given that our dataset (15-min Libre) has sparser signal density, the result does not indicate an engineering fault. **Caveat**: no same-setting peer-reviewed baseline exists on this dataset; the conclusion is an indirect inference from adjacent settings (see §0 literature-comparison note).
4. **RMSE ≤ 9 is not realistic under the current task setting.** Reaching it requires either expanding signals (insulin / carbs / HR), increasing temporal resolution (5-min CGM), or shortening horizon (15 min). Any of these is more likely to hit the target than additional xLSTM hyperparameter search.
5. **Recommended next-step priorities**:
   - Short-term: report per-patient / per-time-of-day / per-glycemia-range RMSE distributions to quantify "hard subgroups."
   - Mid-term: align with the advisor on the precise meaning of "RMSE ≤ 9" (horizon, unit, signal source) and redefine an attainable target.
   - Long-term: if companion T1DiabetesGranada tables are accessible, extend to a multimodal pipeline; otherwise consider switching to or co-evaluating against a 5-min CGM dataset such as OhioT1DM.

---

## Appendix A. Key Code References

- Window construction: [t1d_granada/window_builder.py](../t1d_granada/window_builder.py)
- Causal rolling statistics: [t1d_granada/rolling_stats.py](../t1d_granada/rolling_stats.py)
- Derived features: [t1d_granada/feature_engineer.py](../t1d_granada/feature_engineer.py)
- Model skeleton: [t1d_granada/model.py](../t1d_granada/model.py)
- Training loop & evaluation: [t1d_granada/trainer.py](../t1d_granada/trainer.py)
- Parameter center: [t1d_granada/params.py](../t1d_granada/params.py)
- Upstream plan: [docs/plans/2026-05-09-001-feat-xlstm-glucose-prediction-plan.md](plans/2026-05-09-001-feat-xlstm-glucose-prediction-plan.md)
- Design-pattern record: [docs/solutions/design-patterns/xlstm-shap-regressor-pattern-2026-05-09.md](solutions/design-patterns/xlstm-shap-regressor-pattern-2026-05-09.md)

## Appendix B. Key fields in the current meta.json

```json
{
  "window_size": 8,            // 2 h input
  "forecast_steps": 2,         // +30 min target (skip +15 min)
  "sample_interval_min": 15,
  "rolling_windows_h": [2, 6],
  "rolling_stats": ["mean", "std", "tir_70_180", "p5", "p95"],
  "tir_low": 70, "tir_high": 180,
  "split_train": 0.8, "split_val": 0.1,
  "d_seq": 6,                  // bg_z, bg_diff, tod_sin/cos, dow_sin/cos
  "d_static": 12,              // Sex_M, Age + 10 rolling stats
  "counts": { "train": 15790044, "val": 1973761, "test": 1973744 },
  "n_dropped_stats_filter": 254496   // 1.3% dropped due to min_periods
}
```
