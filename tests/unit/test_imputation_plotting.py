"""Regression tests for imputation dashboard visualization geometry."""

import matplotlib.pyplot as plt
import numpy as np

from pimqc.plotting.imputation import ImputationPlotter


def test_scorecard_cell_borders_do_not_cross_grouped_header() -> None:
    """Keep matrix column borders out of the semantic group-header row."""
    metric_values = {
        "KNN": (0.87, 0.96, 0.98, 0.96, 0.89),
        "LLS": (0.86, 0.96, 0.98, 0.95, 0.96),
    }
    results = {
        method: (
            {
                "JSD_Score": values[0],
                "Wasserstein_Score": values[1],
                "Trustworthiness": values[2],
                "Distance_Rank_Preservation": values[3],
                "Distance_Scale_Preservation": values[4],
                "Auto_Score": float(np.mean(values)),
            },
            np.array([]),
            np.array([]),
        )
        for method, values in metric_values.items()
    }

    figure, axis = plt.subplots()
    visualizer = object.__new__(ImputationPlotter)
    visualizer.runtime_font_fallbacks = visualizer.VECTOR_FONT_FALLBACKS
    visualizer.plot_imputation_preservation_scorecard(
        results,
        selected_method="KNN",
        ax=axis,
    )

    header_center = -0.86
    vertical_borders = []
    for line in axis.lines:
        x_data = np.asarray(line.get_xdata(), dtype=float)
        y_data = np.asarray(line.get_ydata(), dtype=float)
        if x_data.size == 2 and np.isclose(x_data[0], x_data[1]):
            vertical_borders.append(line)
            assert not (
                float(np.min(y_data)) <= header_center <= float(np.max(y_data))
            )

    assert len(vertical_borders) == 6
    assert [patch.get_width() for patch in axis.patches] == [2.0, 3.0]
    plt.close(figure)


def test_nrmse_scatter_describes_feature_strata_without_point_cutoff_lines() -> (
    None
):
    """Describe the feature-median split without implying point-level limits."""
    true_values = np.array([8.0, 10.0, 12.0, 14.0])
    predicted_values = np.array([8.2, 9.8, 11.5, 14.4])
    metrics = {
        "NRMSE_Total": 0.10,
        "NRMSE_Low": 0.12,
        "NRMSE_High": 0.08,
        "Threshold": 10.5,
        "Threshold_Quantile": 0.25,
    }
    figure, axis = plt.subplots()
    visualizer = object.__new__(ImputationPlotter)

    visualizer._plot_nrmse_scatter(
        true_vals=true_values,
        pred_vals=predicted_values,
        metrics=metrics,
        show_colorbar=False,
        ax=axis,
    )

    annotation_text = "\n".join(text.get_text() for text in axis.texts)
    assert annotation_text.splitlines() == [
        "NRMSE (total): 0.1000",
        "NRMSE (low): 0.1200",
    ]
    assert len(axis.lines) == 1
    plt.close(figure)
