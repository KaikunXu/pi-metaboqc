"""Shared plotting lifecycle and utility exports.

The package exposes BaseMetaboVisualizer, which supplies consistent figure
management, export behavior, and presentation styling for all domain-specific
visualization classes.
"""

from .base import BaseMetaboVisualizer

__all__ = ["BaseMetaboVisualizer"]
