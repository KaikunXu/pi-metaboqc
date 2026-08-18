"""Tests for user-facing filesystem and configuration utility output."""

from pathlib import Path

import pytest

from pimqc.io import dir_tree, load_pipeline_config


def test_dir_tree_uses_standard_tree_symbols(tmp_path: Path) -> None:
    """Render standard Unicode branches for nested directory entries."""
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "result.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "root.txt").write_text("ok", encoding="utf-8")

    tree = dir_tree(tmp_path)

    assert "├── nested" in tree
    assert "└── root.txt" in tree
    assert "│   └── result.txt" in tree


def test_unsupported_configuration_error_lists_supported_formats(
    tmp_path: Path,
) -> None:
    """Describe JSON and TOML as the only supported configuration formats."""
    config_path = tmp_path / "pipeline_parameters.yaml"
    config_path.write_text("key: value\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Use JSON or TOML"):
        load_pipeline_config(str(config_path))
