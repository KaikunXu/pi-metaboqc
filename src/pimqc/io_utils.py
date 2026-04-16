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
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
from loguru import logger
import joblib

__max_threading__ = joblib.cpu_count(only_physical_cores=False)


# pi-metaboqc/src/pimqc/io_utils.py

import os
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import Any

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
        self, exc_type: Any, exc_val: Any, exc_tb: Any
    ) -> None:
        """Exit the context manager and safely restore original streams."""
        self._stdout_ctx.__exit__(exc_type, exc_val, exc_tb)
        self._stderr_ctx.__exit__(exc_type, exc_val, exc_tb)
        self.devnull.close()

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
    iterable: Any, total: int, desc: str = "Progress", 
    color: str = None, bar_length: int = 80, position: int = 0
) -> Any:
    """Unified progress bar adapter using tqdm for both CLI and Jupyter.
    
    Args:
        iterable: Iterable object to be wrapped.
        total: Total number of iterations.
        desc: Description text on the left of the progress bar.
        color: Color of the progress bar.
        bar_length: Physical length/width of the progress bar.
        position: Specify the line offset to print this bar (useful for
            parallel multi-bar rendering).
            
    Returns:
        A tqdm wrapped iterable.
    """
    from tqdm import tqdm  # Ensure strict usage of default tqdm
    
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
        leave=True,
        position=position
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

def dir_tree(
    dir_path: Path,
    level: int=-1,
    limit_to_directories: bool=False,
    length_limit: int=1000
):
    """
    Return a visual tree structure of specified directory path.
    
    Ref:
        https://stackoverflow.com/questions/9727673/
    """
    
    # Set symbol for prefix components
    space="    "
    branch="│   "
    # Set symbol for pointers
    tee="├── "
    last="└── "
    
    # accept string coerceable to Path
    dir_path = Path(dir_path)
    
    # Initialize count and output variables
    files = 0
    directories = 0
    file_tree = [""]
    
    def inner(dir_path: Path, prefix: str="", level=-1):
        nonlocal files, directories
        if not level: 
            return # 0, stop iterating
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
                yield from inner(path, prefix=prefix+extension, level=level-1)
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
        f"\n{directories} directories" + (f", {files} files" if files else ""))
    return "\n".join(file_tree)