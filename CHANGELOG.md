# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [PEP 440](https://peps.python.org/pep-0440/) for versioning.

## [1.1.4] - 2026-06-17

### Added

- **[Correction]** Added **WaveICA2** as a configurable correction method and included it in `AUTO` correction evaluation, with R-reference comparison support for WaveICA2, RUV-III, and SERRF.
- **[Imputation]** Added **BPCA** as a MAR imputation candidate in the mask-based NRMSE selection workflow, with configurable convergence settings and R-reference comparison support for BPCA and QRILC.
- **[Normalization]** Added `AUTO` normalization evaluation based on before-to-after QC precision, QC RLE alignment, MA intensity-bias change, and sample-structure preservation, together with deterministic shared MA evaluation positions across candidate methods.

### Changed

- **[Filtering]** Refined the missing-value classification dashboard and flowchart, including boundary-anchored arrows, clearer MAR eligibility labeling, dynamic QC intensity percentile labels, and consistent node typography for BioGroup and non-BioGroup layouts.
- **[Correction]** Updated correction dashboards, `AUTO` scorecards, method labels, and internal-standard visualizations to accommodate WaveICA2 while retaining the existing QC-RLSC, QC-RFSC, QC-SVR, SERRF, and RUV-III workflows.
- **[Imputation]** Updated MAR candidate reporting and dashboard logic to include BPCA and to bypass imputation cleanly when no missing values require filling.
- **[Normalization]** Refactored the normalization dashboard around the final `AUTO` score components, including the integrated score-contribution panel, pooled-QC precision plot, QC RLE alignment plot, MA before/after plots, and sample-structure preservation plot.
- **[Visualizer]** Consolidated legend layout, dense tick-label handling, and shared formatting utilities across dataset-building, filtering, assessment, correction, imputation, and normalization figures.
- **[Reports]** Updated brief and comprehensive report templates to reflect skipped imputation states and auto-normalization score visualizations without showing irrelevant candidate plots.
- **[Tests]** Split R bridge validation into method-specific scripts for QRILC, BPCA, VSN, quantile normalization, WaveICA2, RUV-III, and SERRF, with R-related temporary files excluded from version control.
- **[Code Quality]** Continued type-hint, formatting, and configuration cleanup across source and test modules using `ruff`, `black`, and `pyproject.toml` metadata updates.

### Fixed

- **[Dataset Builder]** Fixed acquisition-overview legend clipping and sublegend overlap for datasets with many batches or sample types.
- **[Assessment]** Fixed dense tick-label overlap and cell-annotation scaling in QC correlation heatmaps, batch-correlation heatmaps, and integrated outlier bar plots.
- **[Filtering]** Fixed flowchart arrow rendering so connectors attach to node borders rather than relying on fragile shrink offsets.
- **[Imputation]** Fixed no-missing-value datasets so imputation is reported as not required and unnecessary imputation dashboards are omitted.
- **[Normalization]** Fixed MA trend-bias jitter by decoupling deterministic score calculation from rendering-time MA plot downsampling.


## [1.1.2a1] - 2026-06-01

### Added

- **[Dataset Builder]** Added critical feature detection during dataset construction to output prompts for degraded analysis modes.
- **[Dataset Builder]** Added automatic replacement of exact zero values to prevent `ZeroDivisionError` during downstream analytical and normalization steps.
- **[Imputation]** Implemented **LLS** (Local Least Squares) for missing value imputation.
- **[Assessment]** Introduced a unified 3-mode clustering routing (`cluster="total"`, `"within-group"`, `"none"`) for QC correlation heatmaps. The `"within-group"` mode dynamically generates isolated batch forests to reveal intra-batch sub-drifts without disrupting the overarching temporal batch sequence.
- **[IO Utils]** Implemented a custom `joblib` context manager (`tqdm_joblib_env`) to seamlessly integrate `tqdm` with parallel processing backends.

### Changed
- **[Visualizer]** Optimized the marker style display mode; numeric characters are now used as marker styles when the number of groups/batches exceeds 10 to robustly handle extremely large datasets.
- **[Visualizer]** Improved VS Code compatibility by reverting PNG inline rendering to native `IPython.display.Image` mode, enabling native image toolbar (copy/save) support.  Refactored visualization engine with a three-tier configuration strategy (Method > Instance > Global) to independently control `save_format`(["pdf", "svg"] or "svg") and `display_format` ("svg" or "png").
- **[Dataset Builder]** Optimized the `Global_Acquisition_Overview` plot to automatically omit the missing value distribution subplot when no missing values are present, and enhanced the legend display format.
- **[Filter]** Optimized the display of the workflow flowchart and retained feature bar charts during degraded analysis when Blank or Bio Group metadata is missing.
- **[Correction]** Overhauled the display logic of the internal standard (IS) scatter plots across multiple correction stages for improved memory efficiency.
- **[Imputation]** Renamed the **Probabilistic** method to **MinProb** to align with widely adopted academic terminology.
- **[Imputation]** Upgraded the MAR simulation algorithm (`generate_gmm_noise_mask`). Shifted from a global GMM evaluation to a strict **batch-wise independent GMM** masking strategy, accurately capturing the localized Limit of Detection (LOD) and noise baseline fluctuations for each analytical batch. Included a robust fallback mechanism for singleton batches or batch-free datasets.
- **[Imputation]** Optimized the subplot grid layout of `Imputer_Candidates.svg` to dynamically adapt to the number of evaluated algorithms.
- **[Assessment]** Refined the rendering logic of QC correlation heatmaps (dynamic toggling of correlation values and adaptive font scaling) and outlier bar plots (automatic tick font size adjustment and tick skipping when the number of outlier bars is excessively large).
- **[Assessment]** Completely refactored the QC correlation heatmap architecture (`plot_qc_corr_heatmap`). Transitioned from Seaborn's default `clustermap` to a pure Matplotlib `GridSpec` engine, achieving pixel-perfect mathematical alignment between independent lower-triangle dendrograms and heatmap cells.

### Fixed
- **[Correction]** Fixed the "progress bar illusion" during parallel processing (e.g., QC-RFSC and SERRF corrections). The progress bar now accurately tracks actual task completion from background workers rather than just task dispatching.
- **[Visualizer]** Fixed canvas memory leakage in Patchworklib workflows by optimizing the execution sequence (Jupyter rendering prior to physical export).

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
