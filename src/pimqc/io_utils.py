#src/pimqc/io_utils.py
"""
Purpose of script: Utility functions for data I/O.
"""
import os
import sys
import json
import ruamel.yaml
import zipfile
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Union
from loguru import logger
import joblib

__max_threading__ = joblib.cpu_count(only_physical_cores=False)


class HiddenPrints:
    """Context manager to suppress stdout and stderr."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

def script_location() -> str:
    """Return the location of the current .py or .ipynb file.

    Returns:
        str: Absolute path of the current working directory.
    """
    return (
        os.getcwd() if hasattr(__builtins__, "__IPYTHON__") 
        else os.path.dirname(__file__)
    )

def is_jupyter() -> bool:
    """Check if the code is running within a Jupyter Notebook environment.

    Returns:
        bool: True if the code is executing within a Jupyter Notebook or 
        JupyterLab environment, False otherwise.
    """
    try:
        from IPython.core.getipython import get_ipython
        
        shell = get_ipython()
        if shell is not None:
            return shell.__class__.__name__ == 'ZMQInteractiveShell'
        return False
    except (NameError, ImportError):
        return False

def get_custom_progress(
    iterable, total, desc="Progress", color="green", bar_length=80):
    """
    Unified progress bar adapter using tqdm for both CLI and Jupyter.
    
    Args:
        iterable: Iterable object (e.g., range or generator).
        total: Total number of iterations.
        desc: Description text on the left of the progress bar.
        color: Color of the progress bar (matplotlib colors are ignored by tqdm).
        bar_length: Physical length/width of the progress bar.
    """
    from tqdm import tqdm
    
    tqdm_color = color if color in [
        "green", "blue", "red", "yellow", "cyan", "magenta", "white", "black"
    ] else None
    
    custom_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [ETA: {remaining}]"
    
    return tqdm(
        iterable, 
        total=total,
        desc=desc, 
        ncols=bar_length, 
        colour=tqdm_color,
        bar_format=custom_format,
        leave=True
    )


def _load_json_file(input_file_path: str) -> Dict[str, Any]:
    """Load JSON file content.

    Args:
        input_file_path (str): The absolute or relative path to the JSON file.

    Returns:
        Dict[str, Any]: Parsed JSON content.
    """
    with open(
        file=input_file_path, mode="r", encoding="utf-8-sig") as json_file:
        content = json.load(json_file)
    return content

def _save_json_file(content: Dict[str, Any], output_file_path: str) -> None:
    """Save dictionary content to a JSON file.

    Args:
        content (Dict[str, Any]): Dictionary data to be saved.
        output_file_path (str): Target file path.
    """
    with open(
        file=output_file_path, mode="w", encoding="utf-8-sig") as json_file:
        json.dump(
            obj=content, fp=json_file, indent=4, allow_nan=False,
            sort_keys=False)

def _load_yaml_file(input_file_path: str) -> Any:
    """Load YAML file content while preserving comments.

    This function uses ruamel.yaml to parse the YAML file, ensuring that
    any comments are kept intact in the loaded ordered dictionary structure.

    Args:
        input_file_path: The absolute or relative path to the YAML file.

    Returns:
        Any: Parsed YAML content with comments preserved.
    """
    from ruamel.yaml import YAML
    yaml_parser = YAML()
    with open(
        file=input_file_path, 
        mode="r", 
        encoding="utf-8-sig"
    ) as yaml_file:
        content = yaml_parser.load(stream=yaml_file)
    return content

def _save_yaml_file(content: Any, output_file_path: str) -> None:
    """Save content to a YAML file while preserving its original comments.

    Args:
        content: Data to be saved (usually parsed by ruamel.yaml).
        output_file_path: Target file path.
    """
    from ruamel.yaml import YAML
    yaml_parser = YAML()
    with open(file=output_file_path, mode="w") as yaml_file:
        yaml_parser.dump(data=content, stream=yaml_file)

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
                f"No such directory, creating a new directory:\n\t{dir_path}.")
            os.makedirs(name=dir_path)

def _exe_time(func: Callable) -> Callable:
    """
    Decorator to log the execution time of a function in HH:MM:SS.SSS format.
    """
    def time_wrap(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        end = datetime.now()
        exe_time = datetime.strptime(
            str(end - start), "%H:%M:%S.%f").strftime("%H:%M:%S.%f")[:-3]
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
            source_folder, f"{os.path.basename(source_folder)}.zip")
    if not os.path.exists(path=source_folder):
        logger.error(f"No such directory:\n\t{source_folder}.")
        raise FileNotFoundError(f"No such directory:\n\t{source_folder}.")
    if os.path.exists(path=output_path):
        logger.warning(
            "The compressed file already exists, and will be overwritten.")

    with zipfile.ZipFile(
        file=output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_folder):
            for file in files:
                if not file.endswith(".zip"):
                    file_path = os.path.join(root, file)
                    archive_path = os.path.relpath(file_path, source_folder)
                    zipf.write(file_path, archive_path)
        logger.success(f"Folder compression has completed:\n\t{output_path}.")