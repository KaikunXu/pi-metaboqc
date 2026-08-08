"""Configuration package exports for validated pipeline settings.

The package exposes the Pydantic schema and shared stage-resolution helper used
to merge defaults, TOML sections, and explicit runtime overrides. Keeping these
exports together gives processing stages one stable configuration entry point.
"""

from .schema import PipelineConfig
from .resolution import resolve_stage_config

__all__ = ["PipelineConfig", "resolve_stage_config"]
