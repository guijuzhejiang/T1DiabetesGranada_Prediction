---
title: xLSTM regressor with SHAP GradientExplainer-compatible scalar output
date: 2026-05-09
category: design-patterns
module: t1d_granada
problem_type: design_pattern
component: tooling
severity: high
applies_when:
  - "Building a PyTorch sequence regressor on top of NX-AI's `xlstm` package (xLSTMBlockStack)"
  - "Need SHAP GradientExplainer attribution over a multi-input scalar-output model"
  - "Mixing sLSTM/mLSTM blocks where Triton/CUDA backends conflict with autograd-based explainers"
  - "Same model must run on CPU, GPU, and through SHAP without code branching"
  - "Causal feature engineering required (no future leak in rolling stats fed alongside the sequence)"
tags: [xlstm, shap, pytorch, time-series, gradient-explainer, slstm-backend, causal-features]
---

# xLSTM regressor with SHAP GradientExplainer-compatible scalar output

## Context

This project trains an xLSTM-based regressor for 30-minute glucose forecasting on the T1DiabetesGranada dataset. It exercises a thin slice of the modern DL stack that is unusually error-prone:

- **xLSTM 2.0.5** as a sequence backbone — the library is recent, with sparse community examples for regression heads.
- **Multi-input model**: `(seq: (B, T, D_seq), static: (B, D_static)) → scalar`. Both streams must flow through SHAP for clinical interpretability.
- **`shap.GradientExplainer`** drives feature attribution; it has rigid expectations about input/output tensor shapes that PyTorch regressors with `squeeze(-1)` heads do not satisfy out of the box.
- **Causal feature engineering**: rolling glucose stats (2h/6h windows) feed in as static features at each window's `last_input_ts`, which makes target leakage trivially easy if the rolling window is configured wrong.
- **Brittle dependency floor**: numba (pulled in by shap) does not yet support numpy 2.3+, so a fresh install in 2026 fails at `import shap` with a confusing error.

The patterns below were arrived at after running into each failure mode at least once. Future xLSTM + SHAP integrations should start from this set rather than rediscover them.

## Guidance

### 1. xLSTM block stack composition for a regressor

Build per-block configs sharing `embedding_dim` / `context_length` / `num_heads`, choose sLSTM positions via a ratio function, wrap in `xLSTMBlockStackConfig`, instantiate `xLSTMBlockStack`. The forward pass shape is the load-bearing detail: `Linear → LayerNorm → trunk → take last timestep → concat static → MLP → squeeze(-1)`.

From [T1DiabetesGranada_Prediction/t1d_granada/model.py](../../../t1d_granada/model.py):

```python
def _slstm_indices(num_blocks: int, ratio: float) -> list[int]:
    if ratio >= 1.0 - 1e-6:
        return []
    if ratio <= 1e-6:
        return list(range(num_blocks))
    return [i for i in range(num_blocks) if i % 2 == 1]
```

```python
slstm_at = _slstm_indices(num_blocks, mlstm_ratio)
mlstm_block = mLSTMBlockConfig(
    mlstm=mLSTMLayerConfig(
        num_heads=num_heads, conv1d_kernel_size=conv_kernel_size, dropout=dropout,
    )
)
slstm_block = (
    sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            num_heads=num_heads, conv1d_kernel_size=conv_kernel_size,
            backend=slstm_backend, dropout=dropout,
        ),
        feedforward=FeedForwardConfig(dropout=dropout),
    )
    if slstm_at else None
)
cfg = xLSTMBlockStackConfig(
    mlstm_block=mlstm_block, slstm_block=slstm_block,
    context_length=context_length, num_blocks=num_blocks,
    embedding_dim=embedding_dim, slstm_at=slstm_at, dropout=dropout,
)
self.trunk = xLSTMBlockStack(cfg)
```

```python
def forward(self, seq, static):                # (B, T, d_seq), (B, d_static)
    x = self.input_proj(seq)                   # (B, T, E)
    x = self.input_norm(x)                     # LayerNorm — required: bg_z and sin/cos differ in scale
    x = self.trunk(x)                          # (B, T, E)
    last = x[:, -1, :]                         # (B, E) — the regression "summary"
    h = torch.cat([last, static], dim=-1)      # (B, E + d_static)
    out = self.head(h).squeeze(-1)             # (B,)
    return out
```

Two non-obvious constraints to bake in at construction:

- `embedding_dim % num_heads == 0` — validate explicitly; xLSTM raises a less-helpful message deeper in the stack.
- `context_length` should equal the input window size `T` (here `WINDOW_SIZE = 8`).

### 2. Pin `slstm_backend='vanilla'`

Every model instantiation in `train.py` and tests passes `slstm_backend="vanilla"`. The reasoning:

- Vanilla is pure-PyTorch ops → identical code path on CPU, GPU, and through `shap.GradientExplainer`.
- The CUDA/Triton sLSTM backends are faster but introduce two failure modes: (a) hard Triton dependency on GPU, (b) gradients through custom CUDA ops do not compose cleanly with SHAP's autograd traversal. Explanations either crash or silently produce wrong values.
- One model is used for training **and** explanation. Any backend split forces dual code paths and breaks the "explainer sees the trained model byte-for-byte" property.

The throughput cost is small relative to the simplification.

### 3. SHAP wrapper for scalar-output multi-input PyTorch regressors

Symptom-without-fix: handing a regressor with `out.squeeze(-1)` to `shap.GradientExplainer` and calling `shap_values` raises `IndexError: too many indices for tensor of dimension 1` — the explainer slices `outputs[:, idx]` internally, which fails on `(B,)`.

Fix is a thin wrapper that re-adds the trailing dim. From [T1DiabetesGranada_Prediction/t1d_granada/shap_analysis.py](../../../t1d_granada/shap_analysis.py):

```python
class _SHAPWrapper(nn.Module):
    """SHAP's GradientExplainer slices `outputs[:, idx]`, so the wrapped model must
    return a (B, n_out) tensor, not (B,). Our regressor squeezes the last dim --
    here we re-add it so SHAP works."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        out = self.inner(seq, static)
        if out.ndim == 1:
            out = out.unsqueeze(-1)
        return out
```

Multi-input is expressed as a list of tensors to the explainer (not a tuple, not a stacked tensor):

```python
explainer = shap.GradientExplainer(wrapped, [bg_seq_t, bg_static_t])
```

The returned shap values come back with a trailing `n_outputs=1` axis on each input that must be squeezed before downstream reshaping:

```python
chunk_shap = explainer.shap_values([fg_seq_t[s:e], fg_static_t[s:e]])
seq_part = np.asarray(chunk_shap[0])
static_part = np.asarray(chunk_shap[1])
if seq_part.ndim == 4 and seq_part.shape[-1] == 1:
    seq_part = seq_part.squeeze(-1)
if static_part.ndim == 3 and static_part.shape[-1] == 1:
    static_part = static_part.squeeze(-1)
```

Process foreground samples in chunks (`batch_size=64`) — `GradientExplainer` runs one backprop per `(foreground_sample, background_sample)` pair, so memory grows quickly without chunking.

Three setup invariants the wrapper depends on:

- `model.eval()` before wrapping (dropout, batchnorm running stats).
- Background and foreground tensors on the same device as the model.
- Vanilla sLSTM backend (see #2).

### 4. Causal rolling stats: `closed='left'`

For any rolling statistic computed at anchor row `t` and used as a model input, set `closed='left'` so the window is `[t - w, t)` — anchor row excluded. From [T1DiabetesGranada_Prediction/t1d_granada/rolling_stats.py](../../../t1d_granada/rolling_stats.py):

```python
roll = s.rolling(f"{w}h", closed="left", min_periods=min_periods)
```

The docstring locks down the contract:

> Causality: uses `closed='left'`, so the row at time t reflects bg values in [t - w, t) -- it never includes t itself. This guarantees that when we look up stats at the last-input timestamp T, the rolling window only sees data strictly before T.

`min_periods = max(1, points_per_window // 2)` rejects anchors with too little history; rows with NaN stats should be filtered out of the training set rather than imputed.

Verify causality with an injection test: set the bg value at `last_input_ts` (and after) to absurd values, recompute features, and assert predictions are bit-identical. If they shift, leakage is occurring.

### 5. `numpy<2.3` env pin

`shap >= 0.45` imports `numba`; `numba 0.61.x` raises `"Numba needs NumPy 2.2 or less. Got NumPy 2.3"` on numpy 2.3+. Pin in `requirements.txt` until numba catches up:

```
numpy<2.3
numba>=0.61.2
```

This is a fresh-install footgun, not a runtime bug — local `pip freeze` will show success right up until someone provisions a new env.

### 6. Optuna search space: prefer `categorical` to avoid `step > 0` errors

A spec like `{"type": "float", "low": 0.0, "high": 0.3, "step": 0.0}` raises `step > 0 must hold`. The current space dodges this by using a positive grid step for the dropout sweep:

```python
"dropout": {"type": "float", "low": 0.0, "high": 0.4, "step": 0.05},
```

For non-zero-grid sweeps where `step` would be ambiguous, prefer `categorical` outright (`{"type": "categorical", "choices": [0.0, 0.1, 0.2, 0.3]}`). It works cleanly with `MedianPruner` and avoids the step-validation gotcha.

### 7. Testable `load_settings`: import the module, not the function

`from t1d_granada.utils import load_settings` binds a local name in the importer's module that pytest's `monkeypatch.setattr("t1d_granada.utils.load_settings", ...)` cannot reach — the patch updates the `utils` module but the consumer already cached the reference.

Fix: import the module, look up the attribute at call time. From [T1DiabetesGranada_Prediction/train.py](../../../train.py):

```python
from t1d_granada import utils as utils_mod
from t1d_granada.utils import make_dir, set_seed, timer  # pure helpers — fine to import directly
...
cfg = utils_mod.load_settings()                          # patchable
```

Rule of thumb: helpers that are stable and pure can be imported by name; functions that tests need to swap (config loaders, side-effecting I/O, time/date) must be reached through the module.

## Why This Matters

- **Without #2 + #3**, SHAP analysis crashes on a model that trains fine. The bug surfaces only at attribution time, often after a long training run, with a confusing tensor-shape error.
- **Without #4**, the model silently learns to peek at the target through rolling stats. Unit tests pass, validation RMSE looks great, real-world deployment fails because the "future leak" is gone at inference time.
- **Without #5**, fresh installs in 2026+ fail at `import shap` with a numba error that appears unrelated to the code being run.
- **Without #1's** exact shape (LayerNorm before stack, last-timestep extraction, MLP head, `squeeze(-1)`), training either diverges or the regression head fails to learn the target magnitude.
- **Without #7**, testing the configuration boundary requires touching real `settings.json` or pytest fixtures that override env vars — fragile and slow.

## When to Apply

- Any new project using **xLSTM 2.0.5+** for regression or any sequence-to-scalar task.
- Any **PyTorch model** that needs `shap.GradientExplainer` with multi-input + scalar output (the wrapper pattern is generic — it applies to any squeezed regressor).
- Any **time-series feature engineering** where rolling stats are computed on the same series being predicted and used as model input.
- Any **conda or pip environment** installing `shap >= 0.45` on Python 3.12+ before numba supports numpy 2.3.
- Any **module with side-effecting helpers** (config loading, env, clocks) that needs to be unit-testable via monkeypatch.

## Examples

### SHAP wrapper — before vs after

Before (raises `IndexError`):

```python
import shap
explainer = shap.GradientExplainer(model, [bg_seq_t, bg_static_t])
sv = explainer.shap_values([fg_seq_t, fg_static_t])
# IndexError: too many indices for tensor of dimension 1
# (model returns (B,), explainer does outputs[:, idx])
```

After (works, with the wrapper from `shap_analysis.py`):

```python
class _SHAPWrapper(nn.Module):
    def __init__(self, inner): super().__init__(); self.inner = inner
    def forward(self, seq, static):
        out = self.inner(seq, static)
        return out.unsqueeze(-1) if out.ndim == 1 else out

wrapped = _SHAPWrapper(model).to(device).eval()
explainer = shap.GradientExplainer(wrapped, [bg_seq_t, bg_static_t])
chunk_shap = explainer.shap_values([fg_seq_t[s:e], fg_static_t[s:e]])
seq_part, static_part = np.asarray(chunk_shap[0]), np.asarray(chunk_shap[1])
if seq_part.ndim == 4 and seq_part.shape[-1] == 1:
    seq_part = seq_part.squeeze(-1)        # (B, T, D_seq)
if static_part.ndim == 3 and static_part.shape[-1] == 1:
    static_part = static_part.squeeze(-1)  # (B, D_static)
```

The test in [T1DiabetesGranada_Prediction/tests/test_shap_analysis.py](../../../tests/test_shap_analysis.py) exercises this end-to-end and asserts shapes:

```python
shap_seq, shap_static = compute_shap(
    model, bg_seq, bg_static, fg_seq, fg_static,
    device=torch.device("cpu"), batch_size=8,
)
assert shap_seq.shape == (32, 4, 6)
assert shap_static.shape == (32, 12)
assert not np.isnan(shap_seq).any()
```

### Causal rolling stats — leak vs no leak

Leaky (default `closed='right'` includes the anchor row):

```python
# At anchor t, window is (t - w, t] — INCLUDES t itself.
roll = s.rolling(f"{w}h", min_periods=mp)        # closed='right' default
mean_t = roll.mean()                             # mean_t[t] sees s[t]
# When `last_input_ts == t`, the static feature mean_t[t] reveals the bg value
# at the boundary — model can short-circuit through it.
```

Non-leaky (current implementation):

```python
roll = s.rolling(f"{w}h", closed="left", min_periods=mp)
# At anchor t, window is [t - w, t) — EXCLUDES t.
# Stats at last_input_ts only depend on strictly earlier values.
```

A regression test that injects garbage at and after `last_input_ts`, then asserts the model's prediction is unchanged, will catch any future regression of this contract.

## Related

- Plan: [T1DiabetesGranada_Prediction/docs/plans/2026-05-09-001-feat-xlstm-glucose-prediction-plan.md](../../plans/2026-05-09-001-feat-xlstm-glucose-prediction-plan.md) — the upstream design doc this work implemented; resolves plan items D1 (sLSTM backend selection) and F1 (SHAP wrapper implementation path).
