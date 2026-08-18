"""Inspect the execution environment when explicitly requested by callers.

The module detects notebook execution and reports available hardware without
performing probes during package import. It is used by ``pimqc.init()`` so
frameworks embedding pi-metaboqc retain control over startup behavior.
"""

import platform
import sys

import psutil
from loguru import logger


def is_jupyter() -> bool:
    """Return whether the active process is a Jupyter kernel."""
    if "ipykernel" in sys.modules:
        return True
    ipython_module = sys.modules.get("IPython")
    if ipython_module is None or not hasattr(ipython_module, "get_ipython"):
        return False
    try:
        shell = ipython_module.get_ipython()
    except Exception:
        return False
    return (
        shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
    )


def print_hardware_diagnostics() -> None:
    """Log an optional summary of the current execution environment."""
    try:
        import cpuinfo

        cpu_name = cpuinfo.get_cpu_info().get("brand_raw", platform.processor())
    except ImportError:
        cpu_name = platform.processor()

    memory = psutil.virtual_memory()
    report = [
        "",
        "=" * 60,
        "System Hardware & Power Diagnostics",
        "=" * 60,
        f"OS Platform     : {platform.platform()}",
        f"CPU Model       : {cpu_name}",
        f"Logical Cores   : {psutil.cpu_count(logical=True)}",
        f"Physical Cores  : {psutil.cpu_count(logical=False)}",
        f"Total RAM       : {memory.total / (1024**3):.2f} GB",
        f"Python Version  : {platform.python_version()}",
        "=" * 60,
    ]
    logger.info("\n".join(report))
