"""Regression tests for lazy modelling-backend imports.

On macOS both torch and xgboost bundle their own OpenMP (libomp) runtime.
Loading both into a single process and running heavy compute in each corrupts
memory and segfaults (see ``otk predict --model transformer``). The fix defers
backend imports so a run only loads the backend it uses.

These tests assert the import isolation that prevents the crash. They run in a
subprocess because ``sys.modules`` is process-global state.
"""

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def _run(code: str) -> str:
    """Run ``code`` in a fresh interpreter and return stdout."""
    env = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert result.returncode == 0, f"subprocess failed:\n{result.stdout}\n{result.stderr}"
    return result.stdout


def test_import_otk_loads_no_backend():
    """``import otk`` must not load torch or xgboost."""
    out = _run(
        "import sys, otk; "
        "print('torch', 'torch' in sys.modules, 'xgboost', 'xgboost' in sys.modules)"
    )
    assert "torch False" in out
    assert "xgboost False" in out


def test_xgb_model_attr_loads_only_xgboost():
    """Resolving an XGBoost model must not drag in torch."""
    out = _run(
        "import sys, otk; otk.XGBNewModel; "
        "print('torch', 'torch' in sys.modules, 'xgboost', 'xgboost' in sys.modules)"
    )
    assert "torch False" in out
    assert "xgboost True" in out


def test_predict_module_does_not_import_torch():
    """Importing the predictor module must not import torch."""
    out = _run(
        "import sys; import otk.predict.predictor; "
        "print('torch', 'torch' in sys.modules)"
    )
    assert "torch False" in out


def test_cli_module_does_not_import_torch():
    """Importing the CLI module must not import torch."""
    out = _run(
        "import sys; import otk.cli; "
        "print('torch', 'torch' in sys.modules)"
    )
    assert "torch False" in out


def test_xgb_predict_cli_runs_without_torch(tmp_path):
    """``otk predict --model xgb_new`` must succeed without loading torch."""
    repo_root = Path(__file__).resolve().parent.parent
    sample = repo_root / "otk_api" / "sample_data" / "sample.csv"
    if not sample.exists():
        import pytest

        pytest.skip("sample data not available")

    output = tmp_path / "predictions.csv"
    env = {
        **os.environ,
        "PYTHONPATH": str(SRC),
        "OTK_MODELS_DIR": str(repo_root / "otk_api" / "models"),
    }
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import sys, runpy\n"
        "sys.argv = ['otk', 'predict', '--input', %r, '--output', %r, '--model', 'xgb_new']\n"
        "import otk.cli\n"
        "try:\n"
        "    otk.cli.cli()\n"
        "except SystemExit:\n"
        "    pass\n"
        "print('torch', 'torch' in sys.modules)\n" % (str(sample), str(output))
    )
    result = subprocess.run(
        [sys.executable, str(probe)], capture_output=True, text=True, env=env, timeout=300
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert "torch False" in result.stdout
    assert output.exists()
