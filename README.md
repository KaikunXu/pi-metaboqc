# π-Metabolomics-Quality Control (`pi-metaboqc`)
**pi-metaboqc** is an automated, robust pipeline for LC-MS metabolomics data pre-processing and quality control.

## Features
- **Data Matrix Generation**: Merge metadata and intensity seamlessly.
- **Missing Value Imputation & Filtering**.
- **Comprehensive Quality Assessment**: PCA, correlation heatmaps, RSD distributions, and Shewhart control charts for internal standards.
- **Signal Correction**: Pooled QC sample-based intra- and inter-batch correction.

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

## Quick Start

```Python
from pimqc.pipeline import piMetaboMain

# Execute the complete QC pipeline
piMetaboMain(
    inputFileDir="path/to/input",
    outputFileDir="path/to/output",
    jsonFileName="Pipeline_Parameters.json",
    metaFileName="Project_Meta.csv",
    intFileName="Project_Intensity.csv"
)
```

#### 3. 单元测试文件 (`tests/test_core_classes.py`)
为了保证代码在开源社区的可靠性，建议使用 `pytest` 编写测试用例。创建 `tests/` 文件夹并编写基础测试：

```python
import pandas as pd
import pytest
from pi_metaboqc.core_classes import MetaboData  # 替换了原来的 MetInt

def test_metabo_data_initialization():
    # 创建模拟数据
    data = pd.DataFrame({"Feature1": [10, 20], "Feature2": [15, 25]})
    
    # 初始化你的核心类
    met_data = MetaboData(data, pipePara={})
    
    # 断言测试属性是否正确继承
    assert met_data.shape == (2, 2)
    assert hasattr(met_data, "attrs")
```



## 分析流程示意

![Pipeline Flowchart](assets/pipeline_flowchart.png)

 ## 质控结果展示 

![QC Result Demo](assets/result_demo.png)