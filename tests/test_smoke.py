"""Step 1 check: the package installs and imports. Real tests arrive with the engine in Step 2."""
import importlib

import pytest

import anomaly_auditor


def test_version_is_set():
    assert anomaly_auditor.__version__ == "0.1.0"


@pytest.mark.parametrize("module", ["schema", "cleaning", "features", "engine", "explain"])
def test_engine_modules_import(module):
    importlib.import_module(f"anomaly_auditor.{module}")


def test_core_libraries_available():
    for name in ("pandas", "numpy", "sklearn", "scipy", "joblib"):
        importlib.import_module(name)
