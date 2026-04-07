# pi-metaboqc Global Parameter Configuration Guide

This document provides a comprehensive guide to the `pipeline_parameters.json` (or `.yaml`) configuration file used in the `pi-metaboqc` metabolomics data cleaning and analysis pipeline. It details the physical meaning, data types, default values, and allowed options or ranges for all parameters controlling the pipeline's behavior.

> **⚠️ Attention:** > Pure JSON format does not support comments. Do NOT add `#` or `//` comments directly into the `pipeline_parameters.json` file, as it will cause parsing errors. If you wish to use comments, please save the configuration file in YAML format (`.yaml`).

---

## 1. Core Dataset Construction (`MetaboInt`)

Parameters in this module map the raw intensity matrix and metadata into the object-oriented `MetaboInt` multi-index data structure.

| Parameter           | Type        | Default          | Options/Range                                                | Description                                                  |
| :------------------ | :---------- | :--------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| `mode`              | `String`    | `"POS"`          | `"POS"`, `"NEG"`                                             | MS polarity mode. Used mainly for report and output file naming. |
| `sample_name`       | `String`    | `"Sample Name"`  | *Any valid column name*                                      | Column name for unique sample IDs in the metadata.           |
| `sample_type`       | `String`    | `"Sample Type"`  | *Any valid column name*                                      | Column name for sample types in the metadata.                |
| `bio_group`         | `String`    | `"Bio Group"`    | *Any valid column name*                                      | Column name for biological/clinical groups in the metadata.  |
| `batch`             | `String`    | `"Batch"`        | *Any valid column name*                                      | Column name for analytical batches in the metadata.          |
| `inject_order`      | `String`    | `"Inject Order"` | *Any valid column name*                                      | Column name for injection sequence in the metadata.          |
| `sample_dict`       | `Dict`      | *See Config*     | Keys must strictly include:<br>`"Actual sample"`, `"Blank sample"`, `"QC sample"` | Dictionary mapping user-defined sample types to pipeline standards. Must contain fixed keys for actual, blank, and QC samples. |
| `internal_standard` | `List[str]` | `[...]`          | *List of valid feature IDs*                                  | List of internal isotope standards used for Shewhart control charts and outlier flagging. |
| `outlier_marker`    | `List[str]` | `[]`             | *List of valid feature IDs*                                  | List of known outlier markers (reserved interface).          |

---

## 2. Invalid Feature & Sample Filtering (`MetaboIntFLTR`)

This module defines the thresholds for 5 feature filtering rules based on Missing Values (MV), Relative Standard Deviation (RSD), and Blank ratios, along with sample warning thresholds.

| Parameter        | Type    | Default | Options/Range | Description                                                  |
| :--------------- | :------ | :------ | :------------ | :----------------------------------------------------------- |
| `mv_global_tol`  | `Float` | `0.7`   | `0.0` - `1.0` | **[Rule 1] Global MV Tolerance.** Max allowed missing ratio globally. Features exceeding this are dropped (filters electrical noise and false-positive fragments). |
| `mv_group_tol`   | `Float` | `0.5`   | `0.0` - `1.0` | **[Rule 2] Group MV Tolerance.** Max allowed missing ratio in **AT LEAST ONE** biological group for a feature to be retained (protects high inter-group specific biomarkers). |
| `mv_qc_tol`      | `Float` | `0.8`   | `0.0` - `1.0` | **[Rule 3] QC MV Tolerance.** Max allowed missing ratio in Pooled QC samples. Features exceeding this are dropped (signals below stable instrument detection limits). |
| `rsd_qc_tol`     | `Float` | `0.7`   | `0.0` - `1.0` | **[Rule 4] QC RSD Tolerance.** Max allowed Relative Standard Deviation (RSD) in Pooled QC samples. Features exceeding this are dropped (highly unstable quantitative signals). |
| `qc_blank_ratio` | `Float` | `5.0`   | `> 0.0`       | **[Rule 5] QC/Blank Ratio Limit.** Min required ratio of QC mean vs Blank mean. *(Note: Features with a Blank mean of 0 are unconditionally retained)*. |
| `sample_mv_tol`  | `Float` | `0.5`   | `0.0` - `1.0` | **[Sample Warning] High MV Sample Threshold.** Samples where the missing feature ratio exceeds this threshold will be flagged (indicating potential needle blockage or degradation) but not forcefully removed. |

---

## 3. Signal Correction & Batch Effect (`MetaboIntSC`)

Controls the non-linear regression algorithms and hyperparameters used to fit and correct the baseline drift over injection time.

| Parameter      | Type           | Default     | Options/Range                                                | Description                                                  |
| :------------- | :------------- | :---------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| `base_est`     | `String`       | `"QC-RFSC"` | `"QC-RLSC"`, `"LOESS"`,<br>`"QC-RFSC"`, `"RF"`,<br>`"QC-SVR"`, `"SVR"` | Baseline fitting estimator. Choose from Locally Weighted Scatterplot Smoothing (LOESS), Random Forest (RF), or Support Vector Regression (SVR). |
| `span`         | `Float`        | `0.3`       | `0.0` - `1.0`                                                | Window span parameter for LOESS smoothing. Only effective when LOESS is selected. |
| `n_tree`       | `Int`          | `500`       | `> 0`                                                        | Number of decision trees for Random Forest regression. Only effective when RF is selected. |
| `random_state` | `Int`          | `12345`     | *Any valid integer*                                          | Random seed ensuring model reproducibility during multi-threading and repetitive testing. |
| `svr_kernel`   | `String`       | `"rbf"`     | `"linear"`, `"poly"`, `"rbf"`, `"sigmoid"`                   | Kernel function for the SVR algorithm (Radial Basis Function `"rbf"` is recommended). Only effective when SVR is selected. |
| `svr_c`        | `Float`        | `1.0`       | `> 0.0`                                                      | Regularization penalty parameter (C) for SVR. Only effective when SVR is selected. |
| `svr_gamma`    | `String/Float` | `"scale"`   | `"scale"`, `"auto"`, or `Float > 0.0`                        | Kernel coefficient for SVR. Only effective when SVR is selected. |

---

## 4. Quality Assessment (`MetaboIntQA`)

Controls PCA visualization, outlier detection, and statistical chart rendering strategies.

| Parameter      | Type      | Default      | Options/Range                    | Description                                                  |
| :------------- | :-------- | :----------- | :------------------------------- | :----------------------------------------------------------- |
| `corr_method`  | `String`  | `"spearman"` | `"spearman"`, `"pearson"`        | Method for computing the correlation matrix of QC samples.   |
| `mask`         | `Boolean` | `true`       | `true`, `false`                  | Whether to apply an upper-triangle mask to correlation heatmaps, displaying a cleaner lower-triangle view. |
| `boundary`     | `String`  | `"IQR"`      | `"IQR"`, `"sigma"`, `"mean-std"` | Boundary calculation method for Shewhart IS control charts (Interquartile Range or 3-Sigma). |
| `stat_outlier` | `String`  | `"All"`      | `"HT2"`, `"SPE-DModX"`, `"All"`  | Statistical criteria for defining outlier samples in PCA (can require a single condition or all conditions to be met). |

---

## 5. Data Normalization (`MetaboIntNorm`)

Defines the final data transformation strategies (Normalization) applied before downstream statistical analysis.

| Parameter       | Type      | Default | Options/Range                           | Description                                                  |
| :-------------- | :-------- | :------ | :-------------------------------------- | :----------------------------------------------------------- |
| `quantile_norm` | `Boolean` | `false` | `true`, `false`                         | Enable standalone Quantile Normalization. If true, overrides column and row normalization settings and applies global non-linear mapping. |
| `col_norm`      | `String`  | `"PQN"` | `"PQN"`, `"TIC"`, `"Median"`, `"None"`  | **Sample-wise (Column) Normalization.** Aimed at eliminating systematic errors in injection volume or extraction concentration. Options include Probabilistic Quotient Normalization (PQN), Total Ion Current Normalization (TIC), Median Normalization. |
| `row_norm`      | `String`  | `"VSN"` | `"VSN"`, `"Auto"`, `"Pareto"`, `"None"` | **Feature-wise (Row) Scaling.** Aimed at stabilizing heteroscedasticity and making features of different abundance levels comparable. Options include Variance Stabilizing Normalization (VSN), Z-Score scaling (Auto Scaling), Pareto Scaling. |