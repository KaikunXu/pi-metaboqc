"""Shared resolution of stage defaults, pipeline settings, and overrides.

resolve_stage_config applies a single precedence rule across stages: built-in
defaults are extended by the relevant TOML section and then by explicit,
non-None runtime overrides. The helper prevents configuration behavior from
drifting between filtering, correction, imputation, and normalization.
"""

from __future__ import annotations

from collections.abc import Mapping


def resolve_stage_config(
    pipeline_params: Mapping[str, object] | None,
    section_name: str,
    defaults: Mapping[str, object],
    explicit_overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Resolve one stage's settings without mutating caller-owned mappings.

    Precedence is stable across every stage: class defaults, then the named
    pipeline section, then explicit non-``None`` constructor arguments.
    """

    resolved = dict(defaults)
    if pipeline_params is not None:
        section = pipeline_params.get(section_name)
        if section is not None:
            if not isinstance(section, Mapping):
                raise TypeError(
                    f"Pipeline section '{section_name}' must be a mapping."
                )
            resolved.update(section)

    if explicit_overrides is not None:
        resolved.update(
            {
                key: value
                for key, value in explicit_overrides.items()
                if value is not None
            }
        )
    return resolved


__all__ = ["resolve_stage_config"]
