"""Tests for consistent stage configuration precedence."""

from pimqc.config import resolve_stage_config


def test_resolve_stage_config_applies_documented_precedence() -> None:
    resolved = resolve_stage_config(
        pipeline_params={"Stage": {"shared": "toml", "from_toml": 2}},
        section_name="Stage",
        defaults={"shared": "default", "from_default": 1},
        explicit_overrides={"shared": "explicit", "ignored": None},
    )

    assert resolved == {
        "shared": "explicit",
        "from_default": 1,
        "from_toml": 2,
    }
