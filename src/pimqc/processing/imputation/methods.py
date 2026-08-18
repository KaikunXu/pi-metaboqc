"""Declare canonical missing-value imputation identifiers and aliases.

The registry provides consistent dispatch and display labels for MAR methods
without flattening their distinct parameters or benchmark behavior. MNAR routing
continues to be controlled independently by the imputation processor.
"""

from ..methods import MethodRegistry, MethodSpec

IMPUTATION_METHODS = MethodRegistry(
    [
        MethodSpec("AUTO", "Auto", aliases=("Best",)),
        MethodSpec(
            "MINPROB",
            "MinProb",
            aliases=("Prob",),
            handler_name="impute_by_minprob",
        ),
        MethodSpec("KNN", "KNN", handler_name="impute_by_knn"),
        MethodSpec("LLS", "LLS", handler_name="impute_by_lls"),
        MethodSpec("BPCA", "BPCA", handler_name="impute_by_bpca"),
        MethodSpec(
            "QRILC", "QRILC", aliases=("QRLIC",), handler_name="impute_by_qrilc"
        ),
        MethodSpec("MEDIAN", "Median", handler_name="impute_by_median"),
    ]
)
