"""U1 smoke test: 验证关键库都能干净导入, 关键参数与工具可用。"""
import importlib

import pytest


@pytest.mark.parametrize(
    "mod",
    ["numpy", "pandas", "sklearn", "torch", "xlstm", "optuna", "mlflow", "shap", "numba", "matplotlib"],
)
def test_critical_imports(mod):
    importlib.import_module(mod)


def test_numpy_pin_below_2_3():
    import numpy as np

    major, minor = np.__version__.split(".")[:2]
    assert (int(major), int(minor)) < (2, 3), (
        f"numpy {np.__version__} ≥ 2.3 will break shap/numba; "
        "pin `numpy<2.3` per plan U1."
    )


def test_xlstm_version():
    import xlstm

    assert xlstm.__version__.startswith("2.0"), f"expected xlstm 2.0.x, got {xlstm.__version__}"


def test_torch_cuda_available():
    import torch

    assert torch.cuda.is_available(), "training requires GPU; cuda not available"


def test_params_module_loads():
    from t1d_granada.params import (
        WINDOW_SIZE,
        FORECAST_STEPS,
        ROLLING_WINDOWS,
        ROLLING_STATS,
        TIR_LOW,
        TIR_HIGH,
        OPTUNA_SEARCH_SPACE,
    )

    assert WINDOW_SIZE == 8
    assert FORECAST_STEPS == 2
    assert ROLLING_WINDOWS == [2, 6]
    assert len(ROLLING_STATS) == 5
    assert TIR_LOW == 70 and TIR_HIGH == 180
    assert "embedding_dim" in OPTUNA_SEARCH_SPACE


def test_set_seed_reproducible():
    from t1d_granada.utils import set_seed
    import numpy as np
    import torch

    set_seed(42)
    np_a = np.random.rand(3)
    torch_a = torch.rand(3)

    set_seed(42)
    np_b = np.random.rand(3)
    torch_b = torch.rand(3)

    assert np.allclose(np_a, np_b)
    assert torch.allclose(torch_a, torch_b)


def test_load_settings_resolves_paths():
    from t1d_granada.utils import load_settings

    cfg = load_settings()
    # Absolute paths should not start with "./"
    for k, v in cfg.items():
        if isinstance(v, str):
            assert not v.startswith("./"), f"{k} not resolved: {v}"
    assert "GLUCOSE_FILE" in cfg
    assert "MLFLOW_TRACKING_URI" in cfg
