"""Load, validate, and normalize pipeline configuration files.

This module owns the format-specific JSON and TOML readers and converts their
contents into the shared Pydantic pipeline schema. Keeping configuration I/O
separate prevents unrelated filesystem helpers from becoming a grab-bag API.
"""

import json
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import ValidationError

from ..config.schema import PipelineConfig


def _load_json(path: Path) -> dict[str, Any]:
    """Return raw configuration data from a JSON file."""
    with path.open(mode="r", encoding="utf-8-sig") as stream:
        return json.load(stream)


def _load_toml(path: Path) -> dict[str, Any]:
    """Return raw configuration data from a TOML file."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError as exc:
            raise ImportError(
                "TOML support on Python < 3.11 requires the 'tomli' package."
            ) from exc

    return tomllib.loads(path.read_text(encoding="utf-8-sig"))


def load_pipeline_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate a JSON or TOML pipeline configuration."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"No such file:\n\t{path}.")

    loaders = {".json": _load_json, ".toml": _load_toml}
    # Resolve the parser strictly from the suffix so unsupported formats never
    # reach schema validation with misleading field errors.
    try:
        loader = loaders[path.suffix.lower()]
    except KeyError as exc:
        message = (
            f"Unsupported config format: {path.suffix.lower()}. "
            "Use JSON or TOML."
        )
        logger.error(message)
        raise ValueError(message) from exc

    # Both input formats cross the same schema boundary and therefore produce
    # an identical normalized dictionary for downstream stages.
    try:
        validated = PipelineConfig.model_validate(loader(path))
    except ValidationError as exc:
        logger.error(
            f"Pipeline configuration validation failed in {path}:\n{exc}"
        )
        raise ValueError(
            "Configuration File Error. See logs for details."
        ) from exc

    logger.success("Pipeline configuration loaded and validated.")
    return validated.model_dump()
