# `pi-metaboqc`: $\pi$-Metabolomics-Quality Control

[![Status](https://badgen.net/badge/status/active-development/orange)](https://github.com/KaikunXu/pi-metaboqc)
[![Status](https://badgen.net/badge/stage/alpha/red)](https://github.com/KaikunXu/pi-metaboqc)
[![License](https://badgen.net/github/license/KaikunXu/pi-metaboqc)](https://github.com/KaikunXu/pi-metaboqc/blob/main/LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**pi-metaboqc** is a high-performance, fully automated data quality control  pipeline designed specifically for large-scale, multi-batch clinical metabolomics.

Built upon a robust Object-Oriented Programming (OOP) architecture, this framework heavily subclasses `pandas.DataFrame` to create the state-aware `MetaboInt` core data structure. It seamlessly integrates the entire preprocessing lifecycle: dataset building, quality assessment, invalid feature/sample filtering, signal correction, missing value imputation and data normalization.

>⚠️ Note: This project is currently under active development. The API may change without notice, and some features might be experimental.

---

## ✨ Core Capabilities

+ **Robust DataFrame Subclassing:** The core `MetaboInt` object natively inherits from pandas DataFrames, automatically preserving custom metadata (`attrs`) and business logic across complex mathematical operations and matrix slicing.
+ **Topology-aware MV Diagnostics:** Moves beyond one-size-fits-all thresholds by intelligently distinguishing between **MAR** (Missing at Random) and **MNAR** (Missing Not at Random / LOD truncation), applying stratified imputation strategies accordingly.
+ **Anti-Extrapolation Signal Correction:** Features a rigorous "Anchor QC Bracketing" logic for SVR/LOESS batch correction. This completely eliminates the blind extrapolation and numerical explosion issues commonly seen at the beginning and end of analytical runs.
+ **Top-tier Variance Stabilization:** Integrates PQN (Probabilistic Quotient Normalization) to address sample dilution bias, and VSN (Variance Stabilizing Normalization) to eliminate feature heteroscedasticity. Outputs production-ready diagnostic plots (RLE, Mean-Variance, MA Plots).
+ **Automated Data Storytelling:** Automatically captures retention rates and statistical metrics across all pipeline stages, leveraging Jinja2 and pypandoc to generate highly readable, expert-level Markdown / Word QC reports.

## 📦 Installation

We strongly recommend installing `pi-metaboqc` within a **Conda** virtual environment. 

The comprehensive HTML/PDF reporting module in this package relies on `WeasyPrint` for high-quality academic rendering. WeasyPrint requires complex system-level C libraries (such as GTK3, Pango, and Cairo) which are notoriously difficult to configure via standard `pip` on Windows. **Using Conda resolves these graphical dependencies automatically.**

### Step 1: Create and Activate Conda Environment

```bash
conda create -n pimqc_env python=3.13
conda activate pimqc_env
```

### Step 2: Install Rendering Engine via Conda

Install `WeasyPrint` via conda-forge to ensure all necessary system graphical libraries are correctly linked before installing the Python package:

```bash
conda install -c conda-forge weasyprint
```

### Step 3: Install `pi-metaboqc`

Once the environment is ready, install the package directly from the GitHub repository using `pip`:

```bash
pip install git+https://github.com/KaikunXu/pi-metaboqc.git
```

### 💡 Note on Fallback Rendering

If you choose to skip `Step 2` or if WeasyPrint fails to load properly on your specific system, `pi-metaboqc` features a built-in multi-stage defense mechanism. It will automatically attempt to download and use TinyTeX (XeLaTeX) for academic-grade PDF rendering, or fall back to generating a fully styled HTML report. You will always get your data!


## 🚀 Quickstart & Tutorials

You only need three files to trigger the fully automated pipeline: a sample metadata table, a raw intensity matrix, and a JSON/YAML configuration file. We provide an interactive Jupyter Notebook that walk you through the entire quality control workflow:

+ **[Quickstart Tutorial](https://github.com/KaikunXu/pi-metaboqc/blob/main/examples/quickstart.ipynb)**: A End-to-End workflow guide from data building to report generation.


## 🛠️ Pipeline Workflow

Upon executing run_pipeline, the system strictly follows a refinement protocol:

+ **Building dataset:** Aligns metadata with the intensity matrix based on JSON configurations to instantiate the MetaboInt object.
+ **Missing value filtering:** Classifies MAR/MNAR and removes invalid features with high missing rates.
+ **Signal correction:** Executes Intra-batch and Inter-batch signal drift correction.
+ **Quality filtering:** Precisely trims low-quality features via Blank/pooled-QC ratio and pooled-QC RSD.
+ **Missing value imputation:** Applies stratified imputation on remaining missing values based on simulation benchmarks or predefined settings.
+ **Normalization:** Performs Sample-level and Feature-level normalization.
+ **Quality assessment (Replicated):** Generates a comprehensive data assessment report, executing during all pipeline stages.

## 📂 Project Structure

```
pi-metaboqc/
├── README.md                     # Project documentation and quickstart guide
├── pyproject.toml                # Modern Python build and dependency config
├── LICENSE                       # MIT license
├── examples/                     # Directory for tutorials and examples
│   └── quickstart.ipynb          # One-click Jupyter Notebook for onboarding
├── src/                          # Core source code directory
│   └── pimqc/                    # Core pi-metaboqc package
│       ├── __init__.py           # Package initialization file
│       ├── core_classes.py       # Core DataStructure class (MetaboInt)
│       ├── visualizer_classes.py # Core Visualization class (MetaboIntVisualizer)
│       ├── dataset_builder.py    # MetaboInt instantiation
│       ├── assessment.py         # Data quality assessment
│       ├── correction.py         # Signal drift & batch correction
│       ├── filtering.py          # High-missing & low-quality features filtering
│       ├── imputation.py         # Missing values imputation
│       ├── normalization.py      # Data normalization
│       ├── pipeline.py           # Automated pipeline orchestrator
│       ├── io_utils.py           # I/O operations
│       ├── plot_utils.py         # Plotting utilities
│       ├── pca_utils.py          # Underlying PCA dimensionality reduction
│       ├── stat_utils.py         # Shared statistical utility functions
│       ├── report_utils.py       # Automated markdown and pdf report rendering
│       ├── data/...              # Demo and testing datasets...
│       └── templates/...         # Template file for generating reports...
│── tests/                        # Unit testing and E2E stress testing...
└── ...                           # Other files required by this module...
```

## 🤝 Contributing & License

This project is licensed under the **MIT License**.
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/KaikunXu/pi-metaboqc/issues).
