# `pi-metaboqc`: $\pi$-Metabolomics-Quality Control

[![Status](https://badgen.net/badge/status/alpha/yellow)](https://github.com/KaikunXu/pi-metaboqc)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://badgen.net/github/license/KaikunXu/pi-metaboqc)](https://github.com/KaikunXu/pi-metaboqc/blob/main/LICENSE)

**pi-metaboqc** is a high-performance, fully automated data quality control pipeline designed specifically for large-scale, multi-batch clinical metabolomics.

![Pipeline of pi-metaboqc](https://github.com/KaikunXu/pi-metaboqc/raw/main/docs/pipeline_of_pi-metaboqc.png)

## ✨ Core Capabilities

* **Pure Python Ecosystem & Native Pandas Integration:** The core data structure, `MetaboInt`, natively inherits from `pandas.DataFrame`. All underlying calculations are strictly implemented using industry-standard libraries like `SciPy` and `scikit-learn`. Furthermore, classical methods that traditionally relied on R (such as Quantile Normalization and VSN) have been completely reconstructed in Python, achieving statistically equivalent results and breaking down language barriers.

* **Intelligent Missing Value Management:** Built-in heuristic algorithms automatically identify and distinguish between MAR (Missing at Random) and MNAR (Missing Not at Random) metabolite features. By evaluating statistical metrics like NRMSE (Normalized Root Mean Square Error), the pipeline auto-tunes and selects the most appropriate filtering and imputation strategies for your specific dataset.

* **Dual-Engine High-Performance Computing:** Powered by a synergistic integration of `joblib` for multi-core parallelization and `Numba` for Just-In-Time (JIT) compilation. This architecture effortlessly accelerates computationally intensive tasks—such as baseline modeling and cross-validation—to near-C speeds, drastically reducing turnaround times for massive clinical cohorts.

* **End-to-End Quality Assessment (QA):** Provides comprehensive data evaluation functions spanning the entire pipeline. From raw data import and missing value handling to signal drift correction and normalization, the distribution and quality of your data are clearly monitored and controllable at every single step.

* **Dual-Tier Automated Reporting & Publication-Ready Visualizations:** The pipeline silently captures critical retention metrics and statistical parameters across all stages, offering users the flexibility to generate either **Brief** (executive summary) or **Comprehensive** (deep-dive audit) PDF/Markdown reports with a single click. Furthermore, all diagnostic plots are natively exported in lossless **SVG** format, ensuring they are instantly ready for high-fidelity editing in Adobe Illustrator or Microsoft PowerPoint for journal submission.

## 📦 Installation

We strongly recommend installing `pi-metaboqc` within a **Conda** virtual environment using [Miniforge](https://github.com/conda-forge/miniforge) (preferred), [Miniconda](https://docs.anaconda.com/free/miniconda/), or [Anaconda](https://www.anaconda.com/download).

Generating high-fidelity HTML and PDF reports requires advanced graphical engines (`pandoc`, `weasyprint`, and `librsvg`). These tools depend on complex, system-level C libraries (e.g., GTK3, Pango) that are notoriously difficult to compile and configure via standard `pip`, particularly on Windows.

Conda effortlessly resolves these low-level dependencies. To guarantee maximum stability across all operating systems, please follow the **Standard Installation** guide below.

> ⚠️ **Note:** While we have integrated an automatic fallback download feature for missing dependencies, it has not been exhaustively tested across all edge cases. Proceeding with the Conda installation remains the most robust and officially supported approach.

### Step 1: Create and Activate Conda Environment

```bash
conda create -n metaboqc python=3.13 pip -y
conda activate metaboqc
```

### Step 2: Pre-install Graphical Engines (Recommended)

Install `pandoc`, `weasyprint` and `librsvg` via `conda-forge` to ensure all necessary system graphical libraries are correctly linked before installing the Python package:

```bash
conda install -c conda-forge pandoc weasyprint librsvg tinycss2 -y
```

### Step 3: Install `pi-metaboqc`

**For standard users:**
Install the stable release directly from PyPI:

```bash
pip install pi-metaboqc
```

Alternatively, install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/KaikunXu/pi-metaboqc.git
```

**For developers (Editable mode):**
If you plan to modify the source code or contribute to the project:

```bash
git clone https://github.com/KaikunXu/pi-metaboqc.git
cd pi-metaboqc
pip install -e .
```

## 🚀 Quickstart & Tutorials

`pi-metaboqc` is designed for zero-friction deployment. You only need three files to trigger the fully automated pipeline: a sample metadata table, a raw intensity matrix, and a TOML configuration file.

We provide execution modalities for different use cases in the `examples/` directory. **For first-time users, we strongly recommend starting with the Interactive Notebook.**

### 1. Interactive Notebook (Recommended for Onboarding)

**Interactive Tutorial (`interactive_tutorial.ipynb`)**: An end-to-end Jupyter Notebook. This is the optimal way to experience `pi-metaboqc`. It allows you to step through the pipeline, visually inspect intermediate QA diagnostic dashboards (including `model_overview` plots with Q2 metrics, natively rendered as high-fidelity SVGs), and intuitively grasp the core algorithmic logic.

Choose the access method that best suits your network environment:

* **[Pre-rendered HTML Viewer](https://raw.githack.com/KaikunXu/pi-metaboqc/main/examples/interactive_tutorial.html)**: A zero-loading, fully rendered static webpage. **Recommended for users in mainland China** to ensure all inline SVG plots and Q2 metrics are displayed instantly without any GitHub API or nbviewer rendering limits.
* **[Google Colab](https://colab.research.google.com/github/KaikunXu/pi-metaboqc/blob/main/examples/interactive_tutorial.ipynb)**: A cloud-executable environment. Best for global users who wish to run the pipeline dynamically with zero local configuration.

### 2. Headless CLI Execution (For Production & Batch Processing)

For deployment on HPC clusters or integration into larger bioinformatics workflows, utilize our robust command-line interface script (`run_pimqc.py`). 

```bash
# Navigate to the examples directory
cd examples

# Option A: Run out-of-the-box with bundled demo data
python run_pimqc.py

# Option B: Run with your own custom clinical cohort
python run_pimqc.py \
    --meta /path/to/your_meta.csv \
    --intensity /path/to/your_intensity.csv \
    --config /path/to/custom_params.toml \
    --outdir /path/to/output_directory

# Option C: Run in silent mode (For background processing)
python run_pimqc.py -q
```

> ⚠️ **Troubleshooting Note for VS Code Users:** When running the CLI script via the integrated terminal in Visual Studio Code, the IDE may occasionally fail to properly inherit full Conda environment variables. This prevents the PDF rendering engine from locating essential system-level C libraries (e.g., GTK3/Pango), causing the report generation to gracefully degrade and output an **HTML** report instead. 

> **Resolution:** You can bypass this by executing the script from a native system terminal (e.g., Anaconda Prompt, macOS Terminal). Alternatively, to permanently configure VS Code for seamless PDF rendering and resolve PowerShell restrictions, please refer to our **[VS Code Environment & Troubleshooting Guide](https://github.com/KaikunXu/pi-metaboqc/tree/main/docs/vscode_conda_troubleshooting_guide.md)**.

## 📂 Project Structure

```bash
pi-metaboqc/
├── README.md                      # Project documentation and quickstart guide
├── pyproject.toml                 # Modern Python build and dependency config
├── LICENSE                        # MIT license
├── examples/                      # Directory for tutorials and examples
│   ├── interactive_tutorial.ipynb # Interactive Jupyter Notebook for onboarding
│   └── run_pimqc.py               # Production-ready CLI execution script
├── src/                           # Core source code directory
│   └── pimqc/                     # Core pi-metaboqc package
│       ├── __init__.py            # Package initialization file
│       ├── core_classes.py        # Core DataStructure class (MetaboInt)
│       ├── visualizer_classes.py  # Core Visualization class (BaseMetaboVisualizer)
│       ├── dataset_builder.py     # MetaboInt instantiation 
│       ├── assessment.py          # Data quality assessment
│       ├── correction.py          # Signal drift & batch correction
│       ├── filtering.py           # High-missing value & low-quality features filtering
│       ├── imputation.py          # Missing values imputation
│       ├── normalization.py       # Data normalization
│       ├── pipeline.py            # Automated pipeline orchestrator
│       ├── io_utils.py            # I/O operations
│       ├── plot_utils.py          # Plotting utilities
│       ├── pca_utils.py           # Underlying PCA dimensionality reduction
│       ├── stat_utils.py          # Shared statistical utility functions
│       ├── report_utils.py        # Automated markdown and pdf report rendering
│       ├── config_schema.py       # Configuration schema and parameter validation
│       ├── templates/...          # Template file for generating reports...
│       └── data/                  # Demo data and configuration file directory
│           ├── project_meta.csv          # Demo project metadata file
│           ├── project_intensity.csv     # Demo project intensity file
│           └── pipeline_parameters.toml  # Demo pipeline parameters file
│── tests/...                      # Unit testing and E2E stress testing...
└── ...                            # Other files required by this module...
```

> *💡 **Note on Configuration:** The entire analytical workflow of `pi-metaboqc` is centrally governed by `pipeline_parameters.toml`. Users can fine-tune all analysis parameters exclusively through this file, without modifying any underlying Python code.

## 🤝 Contributing & License

This project is licensed under the **MIT License**.
Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/KaikunXu/pi-metaboqc/issues).