"""Regression tests for stable distribution-distance calculations."""

import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd
from loguru import logger

from pimqc.constants import DEFAULT_RANDOM_SEED
from pimqc.statistics.metrics import calc_jsd_similarity
from pimqc.statistics.sample_structure import calc_sample_structure_arrays


def test_jsd_handles_constant_and_variable_distributions_without_warning() -> (
    None
):
    """Near-zero variance must not collapse the KDE density to all zeros."""
    variable = np.linspace(19.65, 19.70, 100)
    constant = np.full(100, 18.42)

    with np.errstate(all="raise"):
        result = calc_jsd_similarity(variable, constant)

    assert np.isfinite(result["jsd"])
    assert result["jsd"] > 0.0


def test_jsd_identical_constant_distributions_are_equal() -> None:
    """Two identical point distributions have zero Jensen-Shannon distance."""
    constant = np.full(20, 7.5)

    with np.errstate(all="raise"):
        result = calc_jsd_similarity(constant, constant)

    assert result["jsd"] == 0.0


def test_jsd_empty_distribution_is_undefined() -> None:
    """An empty comparison has no defined distribution distance."""
    result = calc_jsd_similarity(np.array([]), np.array([1.0, 2.0]))

    assert np.isnan(result["jsd"])


def test_jsd_logs_and_uses_histogram_when_kde_degenerates() -> None:
    """A zero-mass KDE degrades explicitly to a shared-bin histogram."""
    messages: list[str] = []
    sink_id = logger.add(
        lambda message: messages.append(str(message)),
        format="{message}",
    )
    try:
        with patch(
            "pimqc.statistics.metrics._numba_gaussian_kde",
            return_value=np.zeros(200),
        ):
            result = calc_jsd_similarity(
                np.array([0.0, 0.5, 1.0]),
                np.array([1.0, 1.5, 2.0]),
            )
    finally:
        logger.remove(sink_id)

    assert np.isfinite(result["jsd"])
    assert any("shared-bin histograms" in message for message in messages)


def test_sample_trustworthiness_reuses_local_distance_ranks() -> None:
    """Global trustworthiness is the mean of deterministic sample terms."""
    rng = np.random.default_rng(DEFAULT_RANDOM_SEED)
    raw = pd.DataFrame(
        rng.lognormal(mean=4.0, sigma=0.5, size=(12, 8)),
        columns=[f"S{i}" for i in range(8)],
    )
    transformed = raw * rng.lognormal(mean=0.0, sigma=0.03, size=raw.shape)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        geometry = calc_sample_structure_arrays(raw, transformed)["geometry"]

    local = geometry["sample_neighborhood_trustworthiness"]
    assert len(local) == raw.shape[1]
    assert np.isfinite(geometry["neighborhood_trustworthiness"])
    assert np.isclose(
        geometry["neighborhood_trustworthiness"],
        local.mean(),
    )
