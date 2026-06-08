# src/pimqc/io_utils.py
"""
Script purpose: Provide shared runtime, logging, and file I/O utilities.

This module centralizes config loading, Pydantic validation, path existence
checks, JSON/TOML helpers, output directory creation, folder zipping, and
project tree rendering. It also manages Loguru setup, hardware diagnostics,
Jupyter detection, progress bars, joblib/tqdm integration, print suppression,
and execution-time logging.
These helpers keep the pipeline modules consistent without duplicating common
environment and filesystem behavior.
"""

import os
import sys
import contextlib
import platform
import json
import zipfile
import psutil
import joblib

from loguru import logger
from datetime import datetime
from pathlib import Path
from types import TracebackType
import numpy as np
import pandas as pd
from functools import wraps
from itertools import islice
from typing import Any, Callable, Dict, Optional, Iterable, Iterator, ParamSpec, TypeVar
from contextlib import redirect_stdout, redirect_stderr
from tqdm import tqdm
from pydantic import ValidationError
from .config_schema import PipelineConfig

__max_threading__ = os.cpu_count()
P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

# Global state for progress bar visibility
SHOW_PROGRESS = True

PROGRESS_BAR_FORMAT = (
    "{l_bar}{bar}| {n_fmt}/{total_fmt} " "[Elapsed: {elapsed} | ETA: {remaining}]"
)


class HiddenPrints:
    """Context manager to completely suppress stdout and stderr.

    Utilizes contextlib for robust stream redirection, catching native
    print statements and warnings in Jupyter environments.
    """

    def __enter__(self) -> "HiddenPrints":
        """Enter the context manager, redirecting streams to devnull."""
        self.devnull = open(os.devnull, "w")
        self._stdout_ctx = redirect_stdout(self.devnull)
        self._stderr_ctx = redirect_stderr(self.devnull)
        self._stdout_ctx.__enter__()
        self._stderr_ctx.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the context manager and safely restore original streams."""
        self._stdout_ctx.__exit__(exc_type, exc_val, exc_tb)
        self._stderr_ctx.__exit__(exc_type, exc_val, exc_tb)
        self.devnull.close()


def print_hardware_diagnostics() -> None:
    """Gather system diagnostics and emit as a single atomic log block."""

    # =========================================================================
    # Phase 1: Data Gathering (This is where the monkey patch might trigger)
    # =========================================================================
    try:
        import cpuinfo

        raw_cpu = platform.processor()
        # Any DEBUG logs triggered by _safe_popen will happen HERE.
        cpu_name = cpuinfo.get_cpu_info().get("brand_raw", raw_cpu)
    except ImportError:
        cpu_name = platform.processor()

    os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
    logical_cores = psutil.cpu_count(logical=True)
    physical_cores = psutil.cpu_count(logical=False)

    # Calculate Memory
    mem = psutil.virtual_memory()
    total_ram = f"{mem.total / (1024**3):.2f} GB"

    # Assess Power Source
    try:
        battery = psutil.sensors_battery()
        if battery:
            plugged = "Plugged In (AC)" if battery.power_plugged else "Battery"
            power_info = f"{plugged} [{battery.percent:.0f}% remaining]"
        else:
            power_info = "Desktop / Uninterruptible Power Supply (AC)"
    except Exception:
        power_info = "Unknown"

    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}."
        f"{sys.version_info.micro}"
    )

    # =========================================================================
    # Phase 2: String Assembly
    # =========================================================================
    # Build the entire diagnostic report as a single list of strings
    report_lines = [
        "",  # Add an empty line for visual spacing from previous logs
        "=" * 60,
        " 🖥️  System Hardware & Power Diagnostics",
        "=" * 60,
        f"OS Platform     : {os_info}",
        f"CPU Model       : {cpu_name}",
        f"Logical Cores   : {logical_cores}",
        f"Physical Cores  : {physical_cores}",
        f"Total RAM       : {total_ram}",
        f"Power Source    : {power_info}",
        f"Python Version  : {py_version}",
        "=" * 60,
    ]

    # =========================================================================
    # Phase 3: Atomic Emission
    # =========================================================================
    # Log the entire assembled string in one atomic operation
    logger.info("\n".join(report_lines))


def setup_loguru_logger(level: str = "INFO") -> None:
    """
    Bulletproof Loguru configuration for pi-metaboqc.
    Uses a callable formatter to avoid KeyError and handle dynamic naming.
    """
    # 1. Remove all existing handlers (including the default one)
    logger.remove()

    # 2. Define the dynamic formatter function
    def dynamic_formatter(record: dict[str, Any]) -> str:
        """
        Custom formatter that processes the record before string interpolation.
        """
        # Extract submodule name in real-time
        # (e.g., 'pimqc.dataset_builder' -> 'dataset_builder')
        submodule_name = record["name"].split(".")[-1]

        # Inject processed name into the extra dict to ensure the format
        # string can locate it.
        record["extra"]["submodule"] = submodule_name

        # Return the formatted template string.
        # Note: loguru will handle color rendering and field filling.
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{extra[submodule]}:{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>\n"
        )

    # 3. Directly add the handler using the dynamic function as the format
    logger.add(sys.stdout, format=dynamic_formatter, level=level)


def script_location() -> str:
    """Return the location of the current .py or .ipynb file.

    Returns:
        str: Absolute path of the current working directory.
    """
    return (
        os.getcwd()
        if hasattr(__builtins__, "__IPYTHON__")
        else os.path.dirname(__file__)
    )


def is_jupyter() -> bool:
    """Check if the code is running within a Jupyter Notebook environment.

    This implementation uses a zero-dependency memory probe to avoid
    triggering IPython compatibility crashes in Python 3.13+ terminal mode.

    Returns:
        bool: True if executing within a Jupyter Notebook, False otherwise.
    """
    import sys

    # Primary check: If ipykernel is loaded, it's definitely a Jupyter/Colab env
    if "ipykernel" in sys.modules:
        return True

    # Secondary safe check: Inspect existing modules without forcing an import
    if "IPython" in sys.modules:
        try:
            ipython_module = sys.modules.get("IPython")
            if ipython_module and hasattr(ipython_module, "get_ipython"):
                shell = ipython_module.get_ipython()
                if shell is not None:
                    return shell.__class__.__name__ == "ZMQInteractiveShell"
        except Exception:
            pass

    return False


def get_custom_progress(
    iterable: Iterable[T],
    total: Optional[int] = None,
    desc: str = "Progress",
    color: Optional[str] = None,
) -> Iterable[T]:
    """Wrap an iterable with a customized tqdm progress bar.

    If the global SHOW_PROGRESS flag is False, this function acts as a
    transparent pass-through and returns the original iterable without
    triggering tqdm.

    Returns:
        Iterable[Any]: The tqdm-wrapped iterable or the original iterable.
    """
    if not SHOW_PROGRESS:
        return iterable

    valid_colors = [
        "green",
        "blue",
        "red",
        "yellow",
        "cyan",
        "magenta",
        "white",
        "black",
    ]
    tqdm_color = color if color in valid_colors else None

    return tqdm(
        iterable,
        total=total,
        desc=desc,
        ncols=120,
        colour=tqdm_color,
        bar_format=PROGRESS_BAR_FORMAT,
    )


@contextlib.contextmanager
def tqdm_joblib_env(
    total: int, desc: str = "Progress", color: Optional[str] = None
) -> Iterator[object]:
    """Context manager to patch joblib and track parallel execution with tqdm.

    If the global SHOW_PROGRESS flag is False, this yields a dummy object
    and completely bypasses the joblib callback modification, ensuring zero
    overhead during silent execution.

    Yields:
        tqdm.tqdm or DummyProgress: The initialized progress bar object.
    """
    if not SHOW_PROGRESS:
        # Define a mock object to safely satisfy the 'with' context syntax
        # without triggering any actual joblib patching or terminal output.
        class DummyProgress:
            def update(self, n: int = 1) -> None:
                pass

            def close(self) -> None:
                pass

        yield DummyProgress()
        return

    valid_colors = [
        "green",
        "blue",
        "red",
        "yellow",
        "cyan",
        "magenta",
        "white",
        "black",
    ]
    tqdm_color = color if color in valid_colors else None

    tqdm_object = tqdm(
        total=total,
        desc=desc,
        ncols=120,
        colour=tqdm_color,
        bar_format=PROGRESS_BAR_FORMAT,
        leave=True,
    )

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        """Custom callback to update tqdm on batch completion."""

        def __call__(self, *args: object, **kwargs: object) -> object:
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    # Store the original callback to safely restore it later
    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

    try:
        yield tqdm_object
    finally:
        # Safely restore the original callback and close the progress bar
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def _load_json_file(input_file_path: str) -> Dict[str, Any]:
    """Load JSON file content.

    Args:
        input_file_path (str): The absolute or relative path to the JSON file.

    Returns:
        Dict[str, Any]: Parsed JSON content.
    """
    with open(file=input_file_path, mode="r", encoding="utf-8-sig") as json_file:
        content = json.load(json_file)
    return content


def _load_toml_file(input_file_path: str) -> Dict[str, Any]:
    """Load TOML file content with environment compatibility.

    Args:
        input_file_path: The absolute or relative path to the TOML file.

    Returns:
        Dict[str, Any]: Parsed TOML content.

    Raises:
        ImportError: If the 'tomli' package is missing in Python < 3.11.
    """
    try:
        import tomllib  # Built-in in Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # Fallback for Python < 3.11
        except ImportError:
            logger.critical("Missing 'tomli'. Please run `pip install tomli`.")
            raise ImportError("Missing 'tomli' package for TOML support.")

    with open(file=input_file_path, mode="r", encoding="utf-8-sig") as toml_file:
        content_str = toml_file.read()
    return tomllib.loads(content_str)


def _save_json_file(content: Dict[str, Any], output_file_path: str) -> None:
    """Save dictionary content to a JSON file.

    Args:
        content (Dict[str, Any]): Dictionary data to be saved.
        output_file_path (str): Target file path.
    """
    with open(file=output_file_path, mode="w", encoding="utf-8-sig") as json_file:
        json.dump(obj=content, fp=json_file, indent=4, allow_nan=False, sort_keys=False)


def load_pipeline_config(config_path: str) -> Dict[str, Any]:
    """Automatically detect and load the pipeline configuration file.

    Supports .json and .toml formats based on file extension.

    Args:
        config_path: Path to the configuration file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.

    Raises:
        ValueError: If the file format is not supported.
    """
    _check_file_exists(config_path)
    ext = os.path.splitext(config_path)[1].lower()

    # 1. Parse raw configuration based on file extension
    if ext == ".json":
        raw_config = _load_json_file(config_path)
    elif ext == ".toml":
        raw_config = _load_toml_file(config_path)
    else:
        err_msg = f"Unsupported config format: {ext}. Use json/toml/yaml."
        logger.error(err_msg)
        raise ValueError(err_msg)

    # 2. Strict type checking and automatic default imputation via Pydantic
    try:
        validated_config = PipelineConfig.model_validate(raw_config)
    except ValidationError as e:
        logger.critical(f"Pipeline configuration validation failed in {config_path}!")
        logger.error(f"Validation Details:\n{e}")
        raise ValueError("Configuration File Error. See logs for details.")

    # 3. Log success and return the native Python dictionary
    logger.success(
        "Pipeline configuration successfully loaded and validated " "via Pydantic."
    )
    return validated_config.model_dump()


def _check_file_exists(file_path: str) -> None:
    """Check if the specified file exists. Raise an error if it doesn't.

    Args:
        file_path (str): Path of the file to check.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        logger.critical(f"No such file: {file_path}.")
        raise FileNotFoundError(f"No such file:\n\t{file_path}.")


def _check_dir_exists(dir_path: str, handle: str = "critical") -> None:
    """Check if a directory exists, and optionally create it.

    Args:
        dir_path (str): Target directory path.
        handle (str, optional): Action to take if directory is missing.
            "critical" raises an error; "makedirs" creates it.
            Defaults to "critical".

    Raises:
        FileNotFoundError:
            If the directory does not exist and handle is "critical".
    """
    if not os.path.exists(dir_path):
        if handle == "critical":
            logger.critical(f"No such directory: {dir_path}.")
            raise FileNotFoundError(f"No such directory:\n\t{dir_path}.")
        elif handle == "makedirs":
            logger.warning(
                f"No such directory, creating a new directory:\n\t{dir_path}."
            )
            os.makedirs(name=dir_path)


def _exe_time(func: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator to log the execution time of a function in HH:MM:SS.SSS format.
    Only logs if the execution time exceeds 1 second.
    """

    @wraps(func)
    def time_wrap(*args: P.args, **kwargs: P.kwargs) -> R:
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()

        # Calculate duration
        duration = end - start

        # Only log if execution time is greater than 1 second
        if duration.total_seconds() > 1:
            # Format the duration to HH:MM:SS.SSS
            # Using str(duration) handles the delta object efficiently
            exe_time = datetime.strptime(str(duration), "%H:%M:%S.%f").strftime(
                "%H:%M:%S.%f"
            )[:-3]

            logger.success(f'Execution time of "{func.__name__}": {exe_time}.')

        return result

    return time_wrap


@_exe_time
def _zip_folder(source_folder: str, output_path: Optional[str] = None) -> None:
    """Compress a target folder into a ZIP file.

    Args:
        source_folder (str): Directory to be zipped.
        output_path (Optional[str], optional): Target zip file path.
            If None, it saves in the same directory. Defaults to None.

    Raises:
        FileNotFoundError: If the source folder does not exist.
    """
    if not output_path:
        output_path = os.path.join(
            source_folder, f"{os.path.basename(source_folder)}.zip"
        )
    if not os.path.exists(path=source_folder):
        logger.error(f"No such directory:\n\t{source_folder}.")
        raise FileNotFoundError(f"No such directory:\n\t{source_folder}.")
    if os.path.exists(path=output_path):
        logger.warning("The compressed file already exists, and will be overwritten.")

    with zipfile.ZipFile(
        file=output_path, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as zipf:
        for root, _, files in os.walk(source_folder):
            for file in files:
                if not file.endswith(".zip"):
                    file_path = os.path.join(root, file)
                    archive_path = os.path.relpath(file_path, source_folder)
                    zipf.write(file_path, archive_path)
        logger.success(f"Folder compression has completed:\n\t{output_path}.")


def dir_tree(
    dir_path: Path,
    level: int = -1,
    limit_to_directories: bool = False,
    length_limit: int = 1000,
) -> str:
    """
    Return a visual tree structure of specified directory path.

    Ref:
        https://stackoverflow.com/questions/9727673/
    """

    # Set symbol for prefix components
    space = "    "
    branch = "│   "
    # Set symbol for pointers
    tee = "├── "
    last = "└── "

    # accept string coerceable to Path
    dir_path = Path(dir_path)

    # Initialize count and output variables
    files = 0
    directories = 0
    file_tree = [""]

    def inner(dir_path: Path, prefix: str = "", level: int = -1) -> Iterator[str]:
        nonlocal files, directories
        if not level:
            return  # 0, stop iterating
        if limit_to_directories:
            contents = [d for d in dir_path.iterdir() if d.is_dir()]
        else:
            contents = list(dir_path.iterdir())
        pointers = [tee] * (len(contents) - 1) + [last]
        for pointer, path in zip(pointers, contents):
            if path.is_dir():
                yield prefix + pointer + path.name
                directories += 1
                extension = branch if pointer == tee else space
                yield from inner(path, prefix=prefix + extension, level=level - 1)
            elif not limit_to_directories:
                yield prefix + pointer + path.name
                files += 1

    file_tree.append(dir_path.name)
    iterator = inner(dir_path, level=level)
    for line in islice(iterator, length_limit):
        file_tree.append(line)
    if next(iterator, None):
        file_tree.append(f"... length_limit, {length_limit}, reached, counted:")
    file_tree.append(
        f"\n{directories} directories" + (f", {files} files" if files else "")
    )
    return "\n".join(file_tree)


def find_ambiguous_attrs(d: dict[str, object], path: str = "") -> None:
    """
    Recursively scan the dictionary to find all complex types that would trigger
    truth value ambiguous conditions (which causes error when concat dataframe).
    """

    for k, v in d.items():
        current_path = f"{path} -> '{k}'" if path else f"'{k}'"

        if isinstance(v, (np.ndarray, pd.Series, pd.DataFrame, pd.Index)):
            print(f"🚨Variable: {current_path} | Type: {type(v).__name__}")

        elif isinstance(v, dict):
            find_ambiguous_attrs(v, current_path)
