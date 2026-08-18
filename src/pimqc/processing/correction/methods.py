"""Declare canonical signal-correction identifiers and aliases.

The registry resolves user-facing names for QC regression, SERRF, RUV-III,
WaveICA, and AUTO selection while retaining handler metadata for orchestration.
Algorithm parameters remain owned by the correction engines.
"""

from ..methods import MethodRegistry, MethodSpec

CORRECTION_METHODS = MethodRegistry(
    [
        MethodSpec("AUTO", "AUTO", aliases=("Auto",)),
        MethodSpec(
            "QC-RLSC",
            "QC-RLSC",
            aliases=("RLSC", "LOESS"),
            handler_name="RegressionCorrector",
        ),
        MethodSpec(
            "QC-RFSC",
            "QC-RFSC",
            aliases=("RFSC", "RF", "Random Forest"),
            handler_name="RegressionCorrector",
        ),
        MethodSpec(
            "QC-SVR",
            "QC-SVR",
            aliases=("QC-SVRC", "SVR"),
            handler_name="RegressionCorrector",
        ),
        MethodSpec("SERRF", "SERRF", handler_name="SERRFCorrector"),
        MethodSpec(
            "RUV-III",
            "RUV-III",
            aliases=("RUV", "RUV3"),
            handler_name="RUVCorrector",
        ),
        MethodSpec(
            "WaveICA 2.0",
            "WaveICA 2.0",
            aliases=("WaveICA2", "WaveICA20", "WaveICA-2.0"),
            handler_name="WaveICA2Corrector",
        ),
    ]
)
