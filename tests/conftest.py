# tests/conftest.py
"""
Script purpose: Configure shared pytest environment and reusable test data.

This file locates the local R installation before rpy2-dependent bridge tests
are imported, injects R paths and locale settings into the process environment,
and suppresses noisy rpy2 console logging. It also provides session-scoped
fixtures for synthetic MS matrices and real package demo data so normalization,
dataset-builder, and pipeline tests can reuse consistent inputs.
"""

import os
import sys
import logging
import subprocess
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Dynamic R Environment Configuration, ready for rpy2.
#
# Respect an explicitly provided R_HOME first, then fall back to the standalone
# system R installation. Bridge tests should not depend on conda R modules.
if "R_HOME" in os.environ:
    os.environ["R_HOME"] = str(Path(os.environ["R_HOME"]))
elif sys.platform == "win32":
    import winreg

    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\R-core\R") as key:
            os.environ["R_HOME"] = winreg.QueryValueEx(key, "InstallPath")[0]
    except FileNotFoundError:
        r_home_out = subprocess.check_output(["R", "RHOME"], text=True)
        os.environ["R_HOME"] = r_home_out.strip()
else:
    r_home_out = subprocess.check_output(["R", "RHOME"], text=True)
    os.environ["R_HOME"] = r_home_out.strip()

# Inject R binary paths into the system PATH.
r_bin_candidates = [
    Path(os.environ["R_HOME"]) / "bin" / "x64",
    Path(os.environ["R_HOME"]) / "bin",
]
for r_bin_path in r_bin_candidates:
    if r_bin_path.exists():
        os.environ["PATH"] = str(r_bin_path) + os.pathsep + os.environ.get("PATH", "")

os.environ["LANGUAGE"] = "en"
os.environ["LC_ALL"] = "C"

# Suppress rpy2 console output globally
from rpy2.rinterface_lib.callbacks import logger as rpy2_logger  # noqa: E402

rpy2_logger.setLevel(logging.ERROR)


@pytest.fixture(scope="session")
def mock_ms_data() -> pd.DataFrame:
    """Generate a shared mock metabolomics dataset for normalization tests.

    The scope="session" ensures this dataframe is only built once per test run,
    saving computation time across multiple test modules.

    Returns:
        pd.DataFrame: Heteroscedastic mock MS data with NaNs.
    """
    np.random.seed(12345)
    n_features, n_samples = 500, 20
    nan_ratio = 0.25

    base_sig = np.logspace(2, 5, n_features).reshape(-1, 1)
    multi_noise = base_sig * np.random.normal(0, 0.15, (n_features, n_samples))
    add_noise = np.random.normal(50, 10, (n_features, n_samples))

    mat = base_sig + multi_noise + add_noise

    if nan_ratio > 0:
        total_elements = n_features * n_samples
        n_nans = int(total_elements * nan_ratio)
        nan_idx = np.random.choice(total_elements, size=n_nans, replace=False)
        mat.flat[nan_idx] = np.nan

    cols = [f"Sample_{i+1}" for i in range(n_samples)]
    idx = [f"Met_{i+1}" for i in range(n_features)]

    return pd.DataFrame(mat, index=idx, columns=cols)


@pytest.fixture(scope="session")
def real_project_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Load real project data from the package data directory."""
    from pimqc.io_utils import load_pipeline_config

    data_dir = Path(__file__).parents[1] / "src" / "pimqc" / "data"

    meta_path = data_dir / "project_meta.csv"
    int_path = data_dir / "project_intensity.csv"
    param_path = data_dir / "pipeline_parameters.toml"

    meta_df = pd.read_csv(meta_path)
    int_df = pd.read_csv(int_path, index_col=0)

    pipeline_params = load_pipeline_config(str(param_path))

    return meta_df, int_df, pipeline_params
