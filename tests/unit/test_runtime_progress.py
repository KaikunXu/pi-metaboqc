"""Regression tests for scoped Joblib runtime configuration."""

import os

import joblib
import pytest

from pimqc.runtime import configure_joblib_cpu_limit, joblib_execution_context


def _query_openmp_threads() -> int:
    """Execute scikit-learn's thread query inside a Joblib worker."""
    from sklearn.utils._openmp_helpers import _openmp_effective_n_threads

    return int(_openmp_effective_n_threads())


def test_loky_context_limits_nested_worker_threads() -> None:
    """Loky workers must not auto-probe physical cores for inner threads."""
    with joblib_execution_context("loky"):
        backend, _ = joblib.parallel.get_active_backend()

        assert backend.__class__.__name__ == "LokyBackend"
        assert backend.inner_max_num_threads == 1
        assert int(os.environ["LOKY_MAX_CPU_COUNT"]) >= 1

    assert int(os.environ["LOKY_MAX_CPU_COUNT"]) >= 1


def test_loky_context_preserves_user_cpu_constraint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit application-level Loky limit must not be overwritten."""
    monkeypatch.setenv("LOKY_MAX_CPU_COUNT", "3")

    with joblib_execution_context("loky"):
        assert os.environ["LOKY_MAX_CPU_COUNT"] == "3"

    assert os.environ["LOKY_MAX_CPU_COUNT"] == "3"


def test_cpu_limit_falls_back_without_physical_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing physical-core data uses a deterministic logical fallback."""
    monkeypatch.delenv("LOKY_MAX_CPU_COUNT", raising=False)
    monkeypatch.setattr(
        "pimqc.runtime.progress.psutil.cpu_count",
        lambda logical: 12 if logical else None,
    )

    assert configure_joblib_cpu_limit() == 6
    assert os.environ["LOKY_MAX_CPU_COUNT"] == "6"


def test_threading_context_does_not_apply_loky_only_limit() -> None:
    """The threading backend remains available without unsupported options."""
    with joblib_execution_context("threading"):
        backend, _ = joblib.parallel.get_active_backend()

        assert backend.__class__.__name__ == "ThreadingBackend"


def test_loky_workers_skip_physical_core_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit inner limits avoid Joblib's Windows physical-core warning."""
    monkeypatch.setenv("PYTHONWARNINGS", "error::UserWarning")

    with joblib_execution_context("loky"):
        thread_counts = joblib.Parallel(n_jobs=2)(
            joblib.delayed(_query_openmp_threads)() for _ in range(2)
        )

    assert thread_counts == [1, 1]
