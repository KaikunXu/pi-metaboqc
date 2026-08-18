"""Tests for call-time configuration precedence in processing runners."""

from functools import cached_property
from pathlib import Path

import pytest

from pimqc.processing.stage import StageResult, StageRunner


class _Processor:
    """Minimal processor that exposes stage configuration and a cached value."""

    def __init__(self) -> None:
        """Initialize a processor with the pipeline-resolved setting."""
        self.attrs = {"method": "from_pipeline"}

    @cached_property
    def configured_method(self) -> str:
        """Expose the currently configured method through a cache."""
        return self.attrs["method"]


class _Runner(StageRunner[_Processor, str]):
    """Minimal runner used to exercise shared lifecycle behavior."""

    def compute(self) -> StageResult[str]:
        """Return the processor setting after runtime resolution."""
        return StageResult(data=self.processor.configured_method)

    def export(self, result: StageResult[str]) -> None:
        """Avoid filesystem side effects in the lifecycle unit test."""

    def render(self, result: StageResult[str]) -> None:
        """Avoid visualization side effects in the lifecycle unit test."""


class _FailingRunner(_Runner):
    """Runner that fails during computation for lifecycle regression tests."""

    def compute(self) -> StageResult[str]:
        """Raise before any artifact directory may be created."""
        raise RuntimeError("calculation failed")


def test_runtime_override_wins_over_pipeline_value_and_clears_cache() -> None:
    """Apply notebook overrides after pipeline settings and clear caches."""
    processor = _Processor()
    assert processor.configured_method == "from_pipeline"

    result = _Runner(
        processor,
        output_dir=None,
        runtime_overrides={"method": "from_notebook"},
        allowed_override_keys={"method"},
    ).run()

    assert result.data == "from_notebook"
    assert processor.attrs["method"] == "from_notebook"


def test_runtime_override_rejects_unknown_configuration_key(
    tmp_path: Path,
) -> None:
    """Reject misspelled notebook settings before creating artifacts."""
    output_dir = tmp_path / "not-created"
    with pytest.raises(TypeError, match="Unsupported runtime override"):
        _Runner(
            _Processor(),
            output_dir=output_dir,
            runtime_overrides={"methdo": "typo"},
            allowed_override_keys={"method"},
        ).run()

    assert not output_dir.exists()


def test_compute_failure_does_not_create_output_directory(
    tmp_path: Path,
) -> None:
    """Create artifact directories only after successful computation."""
    output_dir = tmp_path / "not-created"

    with pytest.raises(RuntimeError, match="calculation failed"):
        _FailingRunner(_Processor(), output_dir=output_dir).run()

    assert not output_dir.exists()


def test_runner_records_explicit_render_context() -> None:
    """Expose the lifecycle processor through the structured stage result."""
    processor = _Processor()
    result = _Runner(processor, output_dir=None).run()

    assert result.render_context["processor"] is processor
