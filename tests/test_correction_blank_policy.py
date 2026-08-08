"""Regression tests for Blank-safe correction fitting and projection."""

import numpy as np
import pandas as pd

from pimqc.processing.correction import RUVCorrector, SERRFCorrector, WaveICA2Corrector


def _blank_policy_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    columns = [f"S{i}" for i in range(8)]
    data = pd.DataFrame(
        [
            [100.0, 10.0, 120.0, 130.0, 140.0, 150.0, 160.0, 10.0],
            [98.0, 1.0, 118.0, 128.0, 138.0, 148.0, 158.0, 1.0],
            [200.0, 5.0, 220.0, 230.0, 240.0, 250.0, 260.0, 5.0],
        ],
        index=["F1", "F2", "F3"],
        columns=columns,
    )
    qc_mask = np.array([True, False, True, False, True, False, True, False])
    blank_mask = np.array([False, True, False, False, False, False, False, True])
    order = np.arange(1, 9, dtype=float)
    return data, qc_mask, blank_mask, order


def test_serrf_blank_prediction_does_not_use_raw_blank_correlates() -> None:
    data, qc_mask, blank_mask, order = _blank_policy_data()
    corr = np.array(
        [[1.0, 0.99, 0.50], [0.99, 1.0, 0.50], [0.50, 0.50, 1.0]]
    )
    engine = SERRFCorrector(
        n_estimators=30, cv_folds=2, n_corr_features=1, random_state=7, n_jobs=1
    )

    protected = engine.fit_transform(
        data,
        batch_array=np.repeat("B1", data.shape[1]),
        qc_mask=qc_mask,
        order_array=order,
        corr_mat=corr.copy(),
        blank_mask=blank_mask,
    )["SERRF"][0]
    legacy = engine.fit_transform(
        data,
        batch_array=np.repeat("B1", data.shape[1]),
        qc_mask=qc_mask,
        order_array=order,
        corr_mat=corr.copy(),
    )["SERRF"][0]

    np.testing.assert_allclose(
        protected.loc[:, qc_mask].to_numpy(), legacy.loc[:, qc_mask].to_numpy()
    )
    assert not np.allclose(
        protected.loc[:, blank_mask].to_numpy(), legacy.loc[:, blank_mask].to_numpy()
    )
    assert np.isfinite(protected.loc[:, blank_mask].to_numpy()).all()


def test_ruviii_fits_only_nonblank_rows_and_projects_blanks() -> None:
    data, qc_mask, blank_mask, _ = _blank_policy_data()
    changed_blanks = data.copy()
    changed_blanks.loc[:, blank_mask] *= 1_000.0
    engine = RUVCorrector(k=2)

    original = engine.fit_transform(
        data,
        qc_mask=qc_mask,
        control_features=data.index,
        blank_mask=blank_mask,
    )["RUV-III"][0]
    changed = engine.fit_transform(
        changed_blanks,
        qc_mask=qc_mask,
        control_features=changed_blanks.index,
        blank_mask=blank_mask,
    )["RUV-III"][0]

    np.testing.assert_allclose(
        original.loc[:, ~blank_mask].to_numpy(),
        changed.loc[:, ~blank_mask].to_numpy(),
    )
    assert list(original.columns) == list(data.columns)
    assert np.isfinite(original.loc[:, blank_mask].to_numpy()).all()


def test_waveica2_excludes_blank_signal_and_predicts_frozen_artifact() -> None:
    rng = np.random.default_rng(31)
    columns = [f"S{i}" for i in range(12)]
    data = pd.DataFrame(
        rng.lognormal(mean=5.0, sigma=0.2, size=(5, len(columns))), columns=columns
    )
    blank_mask = np.array(
        [False, False, True, False, False, False, False, False, True, False, False, True]
    )
    changed_blanks = data.copy()
    changed_blanks.loc[:, blank_mask] *= 100.0
    kwargs = {
        "order_array": np.arange(1, len(columns) + 1, dtype=float),
        "batch_array": np.repeat("B1", len(columns)),
        "blank_mask": blank_mask,
    }

    original = WaveICA2Corrector(n_components=3, cutoff=0.0, random_state=11)
    original_df = original.fit_transform(data, **kwargs)["WaveICA 2.0"][0]
    changed = WaveICA2Corrector(n_components=3, cutoff=0.0, random_state=11)
    changed_df = changed.fit_transform(changed_blanks, **kwargs)["WaveICA 2.0"][0]

    np.testing.assert_allclose(
        original_df.loc[:, ~blank_mask].to_numpy(),
        changed_df.loc[:, ~blank_mask].to_numpy(),
    )
    np.testing.assert_allclose(
        changed_df.loc[:, blank_mask].to_numpy()
        - original_df.loc[:, blank_mask].to_numpy(),
        changed_blanks.loc[:, blank_mask].to_numpy()
        - data.loc[:, blank_mask].to_numpy(),
    )
    assert list(original_df.columns) == columns
