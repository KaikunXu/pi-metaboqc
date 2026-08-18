"""Declare canonical normalization identifiers and aliases.

The registry maps configuration values to normalization handlers and stable
display labels. Candidate evaluation, blank handling, and method-specific
parameters remain in the normalization stage implementation.
"""

from ..methods import MethodRegistry, MethodSpec

NORMALIZATION_METHODS = MethodRegistry(
    [
        MethodSpec("AUTO", "Auto"),
        MethodSpec(
            "ROBUST_LOG_ONLY",
            "Robust Log2",
            aliases=("RobustLogOnly", "RobustLog2Only", "Log2Only", "LogOnly"),
            handler_name="robust_log2_transform",
        ),
        MethodSpec("TIC", "TIC", handler_name="calc_tic_normalization"),
        MethodSpec(
            "MEDIAN", "Median", handler_name="calc_median_normalization"
        ),
        MethodSpec("PQN", "PQN", handler_name="calc_pqn_normalization"),
        MethodSpec("MDFC", "MDFC", handler_name="calc_mdfc_normalization"),
        MethodSpec(
            "QUANTILE", "Quantile", handler_name="calc_quantile_normalization"
        ),
        MethodSpec("VSN", "VSN", handler_name="calc_vsn_normalization"),
    ]
)
