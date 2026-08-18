"""Regression tests for side-effect-free runtime and dataframe finalization."""

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pimqc.constants import DEFAULT_RANDOM_SEED
from pimqc.core import MetaboInt
from pimqc.processing.assessment import MetaboIntAssessor
from pimqc.processing.correction import MetaboIntCorrector
from pimqc.processing.filtering import MetaboIntFilter
from pimqc.processing.imputation import MetaboIntImputer
from pimqc.processing.normalization import MetaboIntNormalizer
from pimqc.processing.stage import StageResult


def _minimal_metabo_frame() -> MetaboInt:
    columns = pd.MultiIndex.from_tuples(
        [
            ("S1", "QC", "B1", 1),
            ("S2", "Sample", "B1", 2),
        ],
        names=["Sample Name", "Sample Type", "Batch", "Inject Order"],
    )
    return MetaboInt(
        [[1.0, 2.0], [3.0, 4.0]],
        index=["F1", "F2"],
        columns=columns,
    )


def test_metaboint_construction_does_not_reset_numpy_global_rng() -> None:
    """Constructing pandas subclasses must not alter application RNG state."""
    # This test intentionally exercises NumPy's legacy global RNG to ensure
    # MetaboInt construction does not mutate host-application state.
    np.random.seed(DEFAULT_RANDOM_SEED)
    expected_first = np.random.random()
    expected_second = np.random.random()

    np.random.seed(DEFAULT_RANDOM_SEED)
    assert np.random.random() == expected_first
    _minimal_metabo_frame()
    assert np.random.random() == expected_second


def test_visualizer_construction_does_not_patch_matplotlib_globals() -> None:
    """Visualizers must leave Axes construction and rcParams untouched."""
    from pimqc.plotting.assessment import AssessmentPlotter

    original_axes_init = matplotlib.axes.Axes.__init__
    original_font_type = mpl.rcParams["pdf.fonttype"]
    frame = _minimal_metabo_frame()
    AssessmentPlotter(frame)

    assert matplotlib.axes.Axes.__init__ is original_axes_init
    assert mpl.rcParams["pdf.fonttype"] == original_font_type


def test_vector_save_keeps_svg_text_editable(tmp_path: Path) -> None:
    """Save SVG labels as text while restoring host rcParams afterwards."""
    from pimqc.plotting.assessment import AssessmentPlotter

    original_svg_fonttype = mpl.rcParams["svg.fonttype"]
    visualizer = AssessmentPlotter(_minimal_metabo_frame())
    figure, axis = plt.subplots()
    axis.set_title("Editable title")
    axis.set_xlabel("Selectable x label")
    axis.set_ylabel("Selectable y label")
    axis.plot([0.0, 1.0], [0.0, 1.0])
    output_path = tmp_path / "editable_vector.svg"

    visualizer.save_and_close_fig(
        figure,
        file_path=str(output_path),
        save_format="svg",
    )

    svg_text = output_path.read_text(encoding="utf-8")
    assert "<text" in svg_text
    assert "Editable title" in svg_text
    assert "Selectable x label" in svg_text
    assert "Selectable y label" in svg_text
    assert mpl.rcParams["svg.fonttype"] == original_svg_fonttype


def test_patchwork_save_keeps_svg_and_pdf_text_editable(
    tmp_path: Path,
) -> None:
    """Preserve text elements and Unicode maps in dashboard vector exports."""
    import patchworklib as pw

    from pimqc.plotting.assessment import AssessmentPlotter

    pw.clear()
    visualizer = AssessmentPlotter(_minimal_metabo_frame())
    brick = pw.Brick(figsize=(3.0, 2.0), label="editable_vector_brick")
    brick.set_title("Editable dashboard title")
    brick.set_xlabel("Selectable dashboard label")
    brick.plot([0.0, 1.0], [0.0, 1.0])
    output_base = tmp_path / "editable_dashboard"

    visualizer.save_and_show_pw(
        brick,
        file_path=str(output_base),
        show_plot=False,
        save_format=["svg", "pdf"],
    )

    svg_text = output_base.with_suffix(".svg").read_text(encoding="utf-8")
    pdf_bytes = output_base.with_suffix(".pdf").read_bytes()
    assert "<text" in svg_text
    assert "Editable dashboard title" in svg_text
    assert "Selectable dashboard label" in svg_text
    assert b"/Subtype /CIDFontType2" in pdf_bytes
    assert b"/ToUnicode" in pdf_bytes


def test_subclasses_share_metaboint_finalization_policy() -> None:
    """Stage subclasses preserve attrs and stats through pandas operations."""
    source = _minimal_metabo_frame()
    source.attrs["custom_state"] = {"values": [1, 2]}
    source.stats["audit"] = {"retained": 2}

    stage_types = (
        MetaboIntAssessor,
        MetaboIntCorrector,
        MetaboIntFilter,
        MetaboIntImputer,
        MetaboIntNormalizer,
    )
    for stage_type in stage_types:
        stage = stage_type(source)
        sliced = stage.iloc[:, :1]
        assert sliced.attrs["custom_state"] == {"values": [1, 2]}
        assert sliced.stats["audit"] == {"retained": 2}
        assert sliced.attrs is not stage.attrs
        assert sliced.stats is not stage.stats


def test_derived_metaboint_preserves_attrs() -> None:
    """Pandas-derived objects keep custom labels and seeds by default."""
    source = _minimal_metabo_frame()
    source.attrs["sample_dict"] = {
        "Actual sample": "Biological",
        "Blank sample": "Solvent",
        "QC sample": "Pooled QC",
    }
    source.attrs["global_seed"] = 73

    derived = MetaboInt(source)

    assert derived.attrs["sample_dict"] == source.attrs["sample_dict"]
    assert derived.attrs["global_seed"] == 73

    fresh = MetaboInt([[2]], index=["f"], columns=["s"])
    fresh.attrs["sample_dict"]["QC sample"] = "QC altered"
    assert source.attrs["sample_dict"]["QC sample"] == "Pooled QC"


def test_inplace_dataframe_mutation_invalidates_derived_cache() -> None:
    """Do not retain QC subsets after their source columns are removed."""
    source = _minimal_metabo_frame()
    assert source._qc.columns.get_level_values("Sample Name").tolist() == [
        "S1"
    ]

    source.drop(columns=[("S1", "QC", "B1", 1)], inplace=True)

    assert source._qc.empty


def test_stage_result_keeps_data_metrics_and_candidates_separate() -> None:
    """The stage contract separates transformation output from audit data."""
    frame = _minimal_metabo_frame()
    result = StageResult(
        data=frame,
        metrics={"score": 0.8},
        candidates=[{"method": "A"}],
    )
    assert result.data is frame
    assert result.metrics["score"] == 0.8
    assert result.candidates == [{"method": "A"}]


def test_auto_imputation_keeps_request_and_selection_separate() -> None:
    """Preserve AUTO in report metrics after selecting a MAR candidate."""
    columns = pd.MultiIndex.from_tuples(
        [
            ("QC1", "QC", "B1", 1),
            ("QC2", "QC", "B1", 2),
            ("S1", "Sample", "B1", 3),
            ("S2", "Sample", "B1", 4),
        ],
        names=["Sample Name", "Sample Type", "Batch", "Inject Order"],
    )
    source = MetaboInt(
        [[np.nan, 2.0, 2.0, 4.0], [4.0, 4.0, 4.0, 4.0]],
        index=["F1", "F2"],
        columns=columns,
    )
    source.attrs["idx_mar"] = ["F1"]
    source.attrs["idx_mnar"] = []
    imputer = MetaboIntImputer(
        source,
        mar_method="Auto",
        mnar_method="row-wise",
    )
    selected_metrics = {
        "NRMSE_Low": 0.1,
        "NRMSE_High": 0.2,
        "NRMSE_Total": 0.15,
        "JSD_Total": 0.1,
        "Wasserstein_Total": 0.2,
        "Wasserstein_Normalized": 0.1,
    }
    candidate_cache = {"LLS": (selected_metrics, np.array([1.0]), np.array([1.0]))}

    with (
        patch.object(
            imputer,
            "_select_best_imputation_method",
            return_value=("LLS", candidate_cache),
        ),
        patch.object(
            imputer,
            "_apply_isolated",
            side_effect=lambda frame, _method, **_kwargs: frame.fillna(1.0),
        ),
    ):
        result = imputer.transform_imputation()

    selection = result.metrics["selection"]
    assert selection["requested_method"] == "Auto"
    assert selection["selected_method"] == "LLS"
    assert selection["is_auto"] is True
    assert result.metadata["requested_method"] == "Auto"
    assert result.metadata["is_auto"] is True


def test_import_does_not_replace_subprocess_popen() -> None:
    """The public package must not monkey-patch the host subprocess module."""
    import pimqc

    assert pimqc is not None
    assert subprocess.Popen.__module__ == "subprocess"


def test_import_does_not_load_optional_r_bridge() -> None:
    """Keep R and rpy2 outside the normal Python package import path."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import pimqc; print('rpy2' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"
