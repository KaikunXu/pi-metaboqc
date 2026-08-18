"""Provide small, domain-independent filesystem operations.

The helpers create validated output directories and render compact Unicode
directory trees for documentation or diagnostics. Configuration parsing and
runtime setup intentionally live in their own modules.
"""

from pathlib import Path
from itertools import islice

from loguru import logger


def ensure_directory(dir_path: str | Path) -> Path:
    """Create a directory and its parents when they do not yet exist."""
    path = Path(dir_path)
    if not path.exists():
        logger.info(f"Creating output directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise NotADirectoryError(f"Path exists but is not a directory: {path}")
    return path


def dir_tree(
    dir_path: str | Path,
    level: int = -1,
    limit_to_directories: bool = False,
    length_limit: int = 1000,
) -> str:
    """Return a compact Unicode tree for a directory."""
    space = "    "
    branch = "│   "
    tee = "├── "
    last = "└── "
    root = Path(dir_path)
    files = 0
    directories = 0

    def inner(path: Path, prefix: str = "", depth: int = -1):
        nonlocal files, directories
        if not depth:
            return
        contents = sorted(
            (
                item
                for item in path.iterdir()
                if not limit_to_directories or item.is_dir()
            ),
            key=lambda item: (not item.is_dir(), item.name.lower()),
        )
        # Only the final sibling uses the closing branch; its descendants then
        # inherit whitespace instead of a continuing vertical guide.
        pointers = [tee] * max(0, len(contents) - 1) + (
            [last] if contents else []
        )
        for pointer, item in zip(pointers, contents):
            yield prefix + pointer + item.name
            if item.is_dir():
                directories += 1
                extension = branch if pointer == tee else space
                yield from inner(item, prefix + extension, depth - 1)
            else:
                files += 1

    lines = ["", root.name]
    iterator = inner(root, depth=level)
    lines.extend(islice(iterator, length_limit))
    if next(iterator, None):
        lines.append(f"... length limit ({length_limit}) reached")
    suffix = f"\n{directories} directories"
    if files:
        suffix += f", {files} files"
    lines.append(suffix)
    return "\n".join(lines)
