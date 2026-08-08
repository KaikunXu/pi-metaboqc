# π-MetaboQC: VS Code Environment & Troubleshooting Guide

**IDE-Level Configuration for Large-Scale Clinical Metabolomics QC Pipelines**

**Category:** Technical Deployment & Architecture | **Core Stack:** Conda (metaboqc) + VS Code Integrated Terminal

---

## 1. Introduction & Architectural Background

π-MetaboQC is a high-performance, automated quality control pipeline optimized for large-scale, multi-batch metabolomics data. It is distributed as `pi-metaboqc` and provides the `pimqc` Python package. To meet the non-interactive deployment requirements of High-Performance Computing (HPC) clusters and backend batch processing, the project features a robust, `argparse`-based CLI execution script (`run_pimqc.py`).

However, when using **Visual Studio Code (VS Code)** as an Integrated Development Environment (IDE) on Windows, developers frequently encounter a specific anomaly: *pure Python dependencies run perfectly, but system-level graphical or report rendering modules crash or gracefully degrade*. 

This is rarely a flaw in the code logic. Instead, it stems from partial incompatibilities between the IDE's terminal process management and Conda's virtual environment isolation mechanisms. This document comprehensively analyzes these traps and provides industrial-grade solutions.

## 2. Core Technical Dependencies

Understanding the root cause requires clarifying the underlying report rendering architecture of `run_pimqc.py`. The pipeline relies heavily on `WeasyPrint` and `librsvg` engines to generate publication-quality PDF audit reports.

| Component                            | Required Ecosystem                                        | Core Function & Manifestation                                |
| :----------------------------------- | :-------------------------------------------------------- | :----------------------------------------------------------- |
| **Python Computational Core**        | Pandas, NumPy, SciPy, Scikit-learn, Numba, Loguru         | Handles data ingestion, parallel/JIT-accelerated algorithm execution, baseline modeling, and matrix correction. |
| **Report Compilation Engine**        | WeasyPrint, Jinja2, Pandoc                                | Converts intermediate QC assets into semantic HTML/Markdown and compiles them into high-fidelity PDF reports. |
| **System-Level Dynamic C-Libraries** | **GTK3, Pango, Cairo, GObject** *(Native OS environment)* | Responsible for complex text layout, font mapping, and high-precision anti-aliased rendering of **lossless SVG vectors** into PDF media. |

## 3. Typical Issues & Resolution Strategies

### Issue 1: The VS Code Terminal Inheritance Trap (Missing C-Libraries)

> **Symptom:** Logs indicate that the pipeline experienced a "graceful degradation" during report rendering, outputting an `.html` report instead of a `.pdf`, or the terminal throws underlying C-library addressing errors like `cannot load library 'gobject-2.0-0'`. However, rendering succeeds 100% of the time in a native system terminal (e.g., Anaconda Prompt).

**Mechanism:** When executing `conda activate metaboqc` in a standard terminal, Conda not only switches the Python interpreter path but also triggers system hook scripts located in the `activate.d` directory. These scripts inject non-Python C-library paths (e.g., `Library/bin`) into the OS's global `PATH` variable.

To maximize launch speed, VS Code's integrated terminal often employs a "shortcut" mechanism—**it merely swaps the Python executable path without running Conda's full system-level environment activation sequence**. Consequently, when `WeasyPrint` attempts to call system graphics libraries, the address resolution fails.

#### 🛠️ Solution A: Force IDE Full Topology Activation (Highly Recommended)
Configure VS Code to strictly invoke the system's Shell initialization chain when launching an integrated terminal.

1. Press `Ctrl + ,` to open VS Code **Settings**.
2. Search for `python.terminal.activateEnvironment` and ensure it is **Checked**.
3. Search for `terminal.integrated.inheritEnv` and ensure it is **Checked**.
4. **Crucial Step:** Press `Ctrl + Shift + P` to open the Command Palette and execute `Developer: Reload Window`.
5. Terminate all existing integrated terminals and open a new one. You should now see the terminal explicitly print the full activation command rather than silently switching.

#### 🛠️ Solution B: Hardcode Static Paths via Workspace Settings

If Solution A fails due to OS security policies, you can explicitly inject the physical paths of the underlying C-libraries in the project's `.vscode/settings.json` (assuming installation at `D:\miniconda3`):

```json
{
    "terminal.integrated.env.windows": {
        "PATH": "D:\\miniconda3\\envs\\metaboqc\\Library\\bin;D:\\miniconda3\\envs\\metaboqc\\Scripts;${env:PATH}"
    }
}
```

> (Note: Backslashes `\` must be double-escaped `\\` in JSON. Placing `Library\bin` before `${env:PATH}` ensures the process prioritizes the GTK3/Pango dynamic link libraries deployed inside Conda.)

### Issue 2: Windows PowerShell Execution Policy Restrictions

> **Symptom:** When attempting to activate the environment or run a project shortcut within VS Code's built-in PowerShell terminal, a red security warning appears:
>
> ```
> cannot be loaded because running scripts is disabled on this system... UnauthorizedAccess
> ```

**Mechanism:** This is a native security barrier in Windows OS (Execution Policy). The default PowerShell policy is usually `Restricted`, the highest defensive posture, which prohibits the execution of any `.ps1` scripts. This directly locks out the PowerShell environment activation hooks built into Miniforge/Miniconda.

#### 🛠️ Solution: Elevate Script Trust Level for the Current User

You do not need to compromise global system security. Simply grant `RemoteSigned` permission to the "Current User" (this allows locally written scripts to run, while internet-downloaded scripts still require authoritative digital signatures).

1. Type `PowerShell` into the Windows taskbar search box.

2. Right-click **Windows PowerShell** and select **Run as Administrator** (Crucial).

3. Type the following command into the elevated console and press Enter:

   PowerShell

   ```
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```

4. When prompted by the console regarding risk, type **`Y`** and press Enter.

5. Completely close the VS Code process and restart it. Terminal script execution privileges are now fully unlocked.

## 4. Pre-Flight Environment Health Checklist

Before feeding large-scale clinical cohorts into the pipeline for formal batch processing, verify that VS Code meets the following technical benchmarks:

- [ ] **Interpreter Verification:** The bottom-right status bar displays `Python 3.1X ('metaboqc': conda)`.
- [ ] **Terminal Identifier:** The newly created integrated terminal line begins with a clear `(metaboqc)` environment tag.
- [ ] **Addressing Test:** Type `where.exe weasyprint` (Windows) or `which weasyprint` (Linux/macOS) in the terminal. The system should seamlessly echo the absolute path within the Conda environment, rather than returning a "not found" error.
