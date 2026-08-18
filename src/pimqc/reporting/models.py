"""Structured, serializable inputs for narrative report generation.

The report contract keeps metrics, resolved configuration, metadata, and
rendered-asset references independent from live pandas objects and processing
engines. It is the portable boundary shared by Markdown, HTML, and PDF
rendering, and it provides a reproducible JSON snapshot for later auditing.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _to_json_value(value: Any) -> Any:
    """Return a JSON-compatible snapshot without retaining live objects."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _to_json_value(value.item())
    if hasattr(value, "tolist"):
        return _to_json_value(value.tolist())
    raise TypeError(
        "Report inputs must be JSON-serializable; received "
        f"{type(value).__name__}."
    )


@dataclass(frozen=True)
class ReportInput:
    """Capture the complete, portable report contract for one pipeline run.

    Data matrices and live processing objects are deliberately excluded. The
    report consumes only its resolved configuration, stage metrics, metadata,
    and a manifest of rendered assets.
    """

    pipeline_metrics: Mapping[str, Any]
    qa_metrics: Mapping[str, Any]
    metadata: Mapping[str, Any]
    resolved_config: Mapping[str, Any]
    asset_manifest: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return an isolated JSON-compatible report snapshot."""
        return {
            "pipeline_metrics": _to_json_value(self.pipeline_metrics),
            "qa_metrics": _to_json_value(self.qa_metrics),
            "metadata": _to_json_value(self.metadata),
            "resolved_config": _to_json_value(self.resolved_config),
            "asset_manifest": _to_json_value(self.asset_manifest),
        }

    def write_json(self, path: str | Path) -> Path:
        """Persist the exact structured input used to render a report."""
        destination = Path(path)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return destination


__all__ = ["ReportInput"]
