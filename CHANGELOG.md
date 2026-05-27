# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://peps.python.org/pep-0440/) for versioning.

## [1.1.1a1] - 2026-05-27

### Added
- **[Imputation]** Implemented **LLS** (Local Least Squares) for missing value imputation. 

### Changed
- **[Imputation]** Renamed the **Probabilistic** method to **MinProb** to align with widely adopted academic terminology.
- **[Imputation]** Optimized the subplot grid layout of Imputer_Candidates.svg to dynamically adapt to the number of evaluated algorithms.

## [1.1.0a1] - 2026-05-26

### Added
- Added `CHANGELOG.md` to the project root directory.
- **[Correction]** Implemented **SERRF** (Systematic Error Removal using Random Forest) method for signal correction. 
- **[Correction]** Implemented **RUV-III** (Remove Unwanted Variation) method, which utilizes Singular Value Decomposition (SVD) to eliminate the need for cross-validation. 
- **[Correction]** Introduced **AUTO** mode, which dynamically evaluates multiple correction algorithms and automatically selects the optimal method based on out-of-fold QC RSD.
- **[Imputation]** Added **QRILC** (Quantile Regression Imputation of Left-Censored data) algorithm, specifically tailored for the imputation of MNAR (Missing Not At Random) features.
- **[Normalization]** Added **MDFC** (Median Difference from Control) approach for data normalization.

### Changed
- **[Correction]** Refactored `QC-RLSC`, `QC-RFSC`, and `QC-SVR` algorithms into a single, unified `RegressionCorrector` class for better maintainability.
- **[Correction]** Integrated a SERRF-like cross-validation strategy into the core regression correction pipeline to enhance stability.
- **[Imputation]** Optimized the visualization logic for NRMSE scatter plots and KDE curves.
- **[Normalization]** Accelerated the parameter calculation speed by introducing a downsampling mechanism.

### Fixed
- Fixed specific compatibility issues that caused execution errors on Windows OS and Python 3.13.
- **[Report]** Resolved a rendering issue in `AUTO` mode where reports failed to display the correct algorithm parameters and specific RSD improvement plots.

## [1.0.0a1] - 2026-05-19

### Added
- Initial alpha release of the `pi-metaboqc` package.
