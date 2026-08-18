"""Configure pi-metaboqc console logging through an explicit runtime hook.

Importing the package does not add, remove, or replace Loguru sinks.
Applications may call this module through ``pimqc.init()`` to install the
package formatter. Standalone scripts and notebooks replace existing Loguru
sinks by default so each record appears once; embedding applications can opt
to preserve their own sinks explicitly.
"""

import logging
import sys
from typing import Any

from loguru import logger

_SINK_ID: int | None = None


class _LoguruBridge(logging.Handler):
    """Route one standard-library logger through the configured Loguru sink."""

    def emit(self, record: logging.LogRecord) -> None:
        """Forward a standard logging record at its corresponding level."""
        message = record.getMessage().strip()
        if not message:
            return

        # WeasyPrint can relay this harmless Windows package-registration
        # diagnostic through Pandoc even when PDF creation succeeds. Keep it
        # available at DEBUG without presenting it as a pipeline warning.
        if (
            "GLib-GIO-WARNING" in message
            and "Microsoft.WindowsAppRuntime" in message
            and "AppxManifest.xml" in message
        ):
            level = "DEBUG"
        else:
            try:
                level = logger.level(record.levelname).name
            except ValueError:
                level = record.levelno

        logger.opt(exception=record.exc_info).log(level, message)


def _configure_external_log_bridges() -> None:
    """Make report-converter diagnostics obey the package Loguru policy."""
    pandoc_logger = logging.getLogger("pypandoc")
    pandoc_logger.handlers[:] = [_LoguruBridge()]
    pandoc_logger.setLevel(logging.DEBUG)
    pandoc_logger.propagate = False


def _format_record(record: dict[str, Any]) -> str:
    """Return the package console format for one Loguru record."""
    record["extra"]["submodule"] = record["name"].split(".")[-1]
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[submodule]}:{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>\n"
    )


def configure_logging(
    level: str = "INFO",
    preserve_existing_sinks: bool = False,
) -> int:
    """Configure the package console sink without duplicate log records.

    Args:
        level: Minimum severity emitted by the package console sink.
        preserve_existing_sinks: Keep Loguru sinks configured by an embedding
            application. Standalone scripts and notebooks should use the
            default ``False`` value to replace Loguru's default stderr sink.
    """
    global _SINK_ID
    # Remove the package-owned sink before replacing all sinks or adding a new
    # one, so repeated explicit configuration cannot duplicate records.
    if _SINK_ID is not None:
        try:
            logger.remove(_SINK_ID)
        except ValueError:
            pass
        _SINK_ID = None

    if not preserve_existing_sinks:
        # Loguru installs a default stderr sink. Leaving it active while adding
        # the package stdout sink renders every notebook and CLI record twice.
        logger.remove()

    _SINK_ID = logger.add(
        sys.stdout,
        format=_format_record,
        level=level.upper(),
    )
    _configure_external_log_bridges()
    return _SINK_ID
