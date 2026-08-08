"""Stable, domain-neutral ranking helpers for evaluated candidates.

The helpers apply eligibility masks, deterministic tie-breakers, and score
margins to candidate tables produced by correction, imputation, or
normalization. Keeping the ranking policy outside individual stages prevents
selection semantics from drifting between AUTO workflows.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

TieBreaker = tuple[str, bool]


def rank_candidates(
    candidates: pd.DataFrame,
    score_column: str,
    tie_breakers: Sequence[TieBreaker] = (),
    eligible_mask: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Return eligible candidates ranked by score and deterministic tie-breakers.

    Scores must be finite and are ranked high-to-low. Each tie-breaker is a
    ``(column, ascending)`` pair, which makes domain-specific secondary
    criteria explicit without embedding any correction or imputation logic.

    """

    if score_column not in candidates:
        raise KeyError(f"Candidate score column is missing: {score_column}")

    ranked = candidates.copy()
    numeric_score = pd.to_numeric(ranked[score_column], errors="coerce")
    valid_mask = pd.Series(np.isfinite(numeric_score), index=ranked.index)

    if eligible_mask is not None:
        valid_mask &= pd.Series(eligible_mask, index=ranked.index).fillna(False)

    ranked = ranked.loc[valid_mask].copy()
    if ranked.empty:
        return ranked

    ranked[score_column] = numeric_score.loc[ranked.index]
    sort_columns = [score_column]
    ascending = [False]
    for column, column_ascending in tie_breakers:
        if column not in ranked:
            raise KeyError(f"Candidate tie-break column is missing: {column}")
        sort_columns.append(column)
        ascending.append(column_ascending)

    return ranked.sort_values(
        by=sort_columns,
        ascending=ascending,
        kind="mergesort",
    )


def selection_margin(scores: pd.Series | np.ndarray) -> float:
    """Return the finite top-two score gap, or NaN when it is undefined."""

    values = pd.to_numeric(pd.Series(scores), errors="coerce")
    values = values[np.isfinite(values)].sort_values(ascending=False)
    if len(values) < 2:
        return float("nan")
    return float(values.iloc[0] - values.iloc[1])


__all__ = ["TieBreaker", "rank_candidates", "selection_margin"]
