"""Package-wide constants for deterministic pi-metaboqc behavior.

The project uses one documented base seed for reproducible workflows. Public
APIs retain their established ``global_seed`` or ``random_state`` parameter
names, while internal fallbacks import this value instead of repeating magic
integers. Callers can always override the default explicitly.
"""

DEFAULT_RANDOM_SEED = 0

__all__ = ["DEFAULT_RANDOM_SEED"]
