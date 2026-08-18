"""Configure R only for optional cross-language reference tests.

The reference suite compares Python algorithms with their R counterparts. This
configuration locates R, prepares rpy2's Windows ABI mode, and quiets R console
logging without affecting unit or integration test collection.
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pimqc.constants import DEFAULT_RANDOM_SEED


def _resolve_r_home() -> Path:
    """Resolve R_HOME or skip the optional reference suite when unavailable."""
    configured = os.environ.get("R_HOME")
    if configured:
        return Path(configured)

    if sys.platform == "win32":
        import winreg

        try:
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\R-core\R",
            ) as key:
                return Path(winreg.QueryValueEx(key, "InstallPath")[0])
        except FileNotFoundError:
            pass

    executable = shutil.which("R")
    if executable is None:
        pytest.skip("R is not installed; reference tests are optional.")
    result = subprocess.run(
        [executable, "RHOME"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip())


R_HOME = _resolve_r_home()
os.environ["R_HOME"] = str(R_HOME)

# Make the R shared library discoverable before importing rpy2.
for r_bin_path in (R_HOME / "bin", R_HOME / "bin" / "x64"):
    if r_bin_path.exists():
        os.environ["PATH"] = (
            f"{r_bin_path}{os.pathsep}{os.environ.get('PATH', '')}"
        )

os.environ["LANGUAGE"] = "en"
if sys.platform == "win32":
    os.environ.pop("LC_ALL", None)
    os.environ.pop("LC_CTYPE", None)
    if (R_HOME / "bin" / "x64").exists():
        os.environ.setdefault("R_ARCH", "/x64")
    os.environ.setdefault("RPY2_CFFI_MODE", "ABI")

    import rpy2.situation as rpy2_situation

    _get_r_flags = rpy2_situation.get_r_flags

    def _get_r_flags_windows_compat(
        r_home: str,
        flags: str,
    ) -> tuple[argparse.Namespace, list[str]]:
        """Return empty linker flags when Windows Rtools is unavailable."""
        if shutil.which("make") is None:
            return argparse.Namespace(I=[], L=[], l=[]), []
        try:
            return _get_r_flags(r_home, flags)
        except IndexError:
            return argparse.Namespace(I=[], L=[], l=[]), []

    rpy2_situation.get_r_flags = _get_r_flags_windows_compat
else:
    os.environ["LC_ALL"] = "C"

try:
    from rpy2.rinterface_lib.callbacks import logger as rpy2_logger
except ImportError:
    pytest.skip("rpy2 is not installed; reference tests are optional.")

rpy2_logger.setLevel(logging.ERROR)


@pytest.fixture
def mock_ms_data() -> pd.DataFrame:
    """Generate deterministic heteroscedastic data for R normalization tests."""
    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    feature_count, sample_count = 500, 20
    base_signal = np.logspace(2, 5, feature_count).reshape(-1, 1)
    multiplicative_noise = base_signal * rng.normal(
        0.0,
        0.15,
        (feature_count, sample_count),
    )
    additive_noise = rng.normal(50.0, 10.0, (feature_count, sample_count))
    values = base_signal + multiplicative_noise + additive_noise

    missing_count = int(values.size * 0.25)
    missing_indices = rng.choice(values.size, size=missing_count, replace=False)
    values.flat[missing_indices] = np.nan
    return pd.DataFrame(
        values,
        index=[f"Met_{index + 1}" for index in range(feature_count)],
        columns=[f"Sample_{index + 1}" for index in range(sample_count)],
    )
