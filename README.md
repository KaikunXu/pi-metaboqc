# π-Metabolomics-Quality Control (`pi-metaboqc`)

[![Status](https://img.shields.io/badge/status-active--development-orange.svg)](https://github.com/KaikunXu/pi-metaboqc)

**pi-metaboqc** is a comprehensive LC-MS metabolomics data quality control module. This module provides an object-oriented pipeline for preprocessing metabolomics data, including dataset building, invalid feature/sample filtering, quality assessment, signal correction, and data normalization..

⚠️ Note: This project is currently under active development. The API may change without notice, and some features might be experimental.

## Installation

You can install the module via pip:

Option 1: Install directly from GitHub (Recommended for most users)

```bash
pip install git+https://github.com/KaikunXu/pi-metaboqc.git
``` 

Option 2: Install from source (For developers)

If you want to contribute to the project, modify the algorithm, or explore the source code, you can clone the repository and install it in "editable" mode. This means any changes you make to the local code will immediately take effect without needing to reinstall the package.

```bash
# 1. Clone the repository
git clone https://github.com/KaikunXu/pi-metaboqc.git

# 2. Navigate into the project directory
cd pi-metaboqc

# 3. Install in editable mode
pip install -e .
```
