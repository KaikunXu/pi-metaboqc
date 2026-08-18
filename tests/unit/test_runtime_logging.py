"""Tests for single-record logging in standalone pi-metaboqc sessions."""

import os
import subprocess
import sys
from pathlib import Path


def test_init_replaces_loguru_default_sink_for_standalone_sessions() -> None:
    """Emit one record after initialization without duplicate console logs."""
    source_root = Path(__file__).parents[2] / "src"
    environment = os.environ.copy()
    existing_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(source_root), existing_path])
    )
    marker = "pimqc-single-log-record"
    command = (
        "import pimqc; "
        "from loguru import logger; "
        "pimqc.init(check_hardware=False, log_level='INFO'); "
        f"logger.info('{marker}')"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert (completed.stdout + completed.stderr).count(marker) == 1
    assert marker in completed.stdout
    assert marker not in completed.stderr


def test_pypandoc_standard_logs_are_routed_through_loguru() -> None:
    """Pandoc diagnostics must use the package sink rather than raw stderr."""
    source_root = Path(__file__).parents[2] / "src"
    environment = os.environ.copy()
    existing_path = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(source_root), existing_path])
    )
    marker = "pimqc-pandoc-log-bridge"
    command = (
        "import logging, pimqc; "
        "pimqc.init(check_hardware=False, log_level='INFO'); "
        f"logging.getLogger('pypandoc').warning('{marker}')"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.stdout.count(marker) == 1
    assert "WARNING" in completed.stdout
    assert marker not in completed.stderr
