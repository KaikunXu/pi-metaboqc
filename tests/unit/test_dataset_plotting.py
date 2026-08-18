"""Regression tests for dataset-construction plotting metadata."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pimqc.core import MetaboInt
from pimqc.dataset import build_dataset
from pimqc.plotting.dataset import DatasetPlotter


def test_dataset_plotter_uses_resolved_metaboint_attrs() -> None:
    """Use labels resolved on the dataset, without reparsing configuration."""
    columns = pd.MultiIndex.from_tuples(
        [("S1", "Pool", "Run-A", 1)],
        names=["Specimen", "Class", "Run", "Sequence"],
    )
    dataset = MetaboInt(
        [[1.0]],
        columns=columns,
        sample_name="Specimen",
        sample_type="Class",
        batch="Run",
        inject_order="Sequence",
        sample_dict={
            "Actual sample": "Study",
            "Blank sample": "Solvent",
            "QC sample": "Pool",
        },
    )

    metadata = DatasetPlotter(dataset)._get_plot_metadata()

    assert metadata == {
        "batch_column": "Run",
        "sample_type_column": "Class",
        "inject_order_column": "Sequence",
        "qc_label": "Pool",
        "actual_label": "Study",
        "blank_label": "Solvent",
    }


def test_build_dataset_renders_dashboard_with_output_dir(
    tmp_path: Path,
) -> None:
    """Keep dashboard export inside the dataset execution entry point."""
    metadata = pd.DataFrame(
        {
            "Sample Name": ["S1"],
            "Sample Type": ["QC"],
            "Batch": ["Batch-1"],
            "Inject Order": [1],
        }
    )
    intensity = pd.DataFrame(
        {"S1": [1.0]},
        index=pd.Index(["Feature-1"], name="Metabolite"),
    )

    with patch.object(
        DatasetPlotter,
        "plot_dataset_dashboard",
        return_value=object(),
    ) as plot_dashboard, patch.object(
        DatasetPlotter,
        "save_and_show_pw",
    ) as save_dashboard:
        build_dataset(
            meta_info=metadata,
            int_df=intensity,
            output_dir=tmp_path,
        )

    plot_dashboard.assert_called_once_with()
    save_dashboard.assert_called_once()
    assert Path(
        save_dashboard.call_args.kwargs["file_path"]
    ).name == "Global_Acquisition_Overview.svg"
