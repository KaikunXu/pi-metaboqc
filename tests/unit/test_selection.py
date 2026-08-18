"""Tests for reusable candidate-ranking helpers."""

import numpy as np
import pandas as pd

from pimqc.statistics.selection import rank_candidates, selection_margin


def test_rank_candidates_applies_score_and_deterministic_tie_breakers() -> None:
    """Rank finite candidates with stable, ordered tie breakers."""
    candidates = pd.DataFrame(
        {
            "method": ["B", "A", "Unavailable"],
            "score": [0.8, 0.8, np.nan],
            "error": [0.2, 0.1, 0.0],
        }
    )

    ranked = rank_candidates(
        candidates,
        score_column="score",
        tie_breakers=(("error", True), ("method", True)),
    )

    assert ranked["method"].tolist() == ["A", "B"]
    assert selection_margin(ranked["score"]) == 0.0
