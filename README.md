# FatigueData-AM2022: Predicting the Fatigue Life of 3D-Printed Metals

This folder is a complete copy of the project for studying, running and presenting it. It does not need the
original development folder.

## Start here

1. **Read `documents/Understanding_Our_Project.pdf` first.** It explains the whole project in simple language,
   with the figures, ten viva questions and answers, and a glossary.
2. **Install the software and set up the project once** (sections 3 and 4). This needs internet and usually takes
   10–20 minutes, mostly downloading.
3. **Run the four notebooks in order: 01 → 02 → 03 → 04** (sections 5 and 6). Together they took 7.5 minutes on
   our test computer; a slower laptop can take longer.

The notebooks already show the outputs of our last full run, so you can read them before running anything.

---

## 1. What this project is about

*Fatigue* is the failure of a metal part under repeated loading, at stresses well below the stress that breaks it
in a single pull. The *fatigue life* N<sub>f</sub> is the number of load cycles a test specimen survives. For
3D-printed (additively manufactured, AM) metals, fatigue life is hard to predict. It depends on the alloy, the
printing process, the heat treatment, the surface finish and the test conditions.

We used the public database **FatigueData-AM2022** (Zhang & Xu, 2023), which collects fatigue tests reported in
published research papers. After cleaning, the study uses **12,666 room-temperature test results** from 1,533 S-N
curves in 440 papers, covering 93 alloys.

We trained five models to predict log<sub>10</sub>(N<sub>f</sub>):

- a **Basquin** curve fitted to each alloy (the classical engineering equation);
- **Ridge** regression;
- **Random Forest**;
- **LightGBM**;
- a **hybrid** that corrects the Basquin prediction with machine learning.

The question is how fair the usual way of testing such models is. We tested every model in three ways:

- **A, random points.** Test measurements are picked at random. Such models are often tested this way.
- **B, curves held out.** Whole S-N curves are kept out of training.
- **C, articles held out.** Whole papers are kept out of training. This is closest to a new lab using the model.

**Main result.** For LightGBM, the error (RMSE of log<sub>10</sub> N<sub>f</sub>) is 0.673 under A, 0.787 under B
and 0.924 under C. It is 37% higher when the model meets papers it has never seen. So random splitting makes the
models look better than they will be for a new lab.

We compare our predictions with the **measured lives reported in the papers**. We do not compare our models with
other papers' machine-learning models.

---

## 2. What is in this folder

```text
Sister_Handover/
├── README.md            this file
├── requirements.txt     the Python packages to install
├── run_all.py           optional: runs all four notebooks in order from PowerShell
├── notebooks/           the four official notebooks (always run them in order 01 → 04)
│   └── archive_original/  backup copies of the original, uncleaned notebooks (do not run these)
├── src/                 Python helper code that the notebooks import (do not move or rename it)
├── tests/               42 automatic checks; notebook 03 runs them
├── data/
│   ├── raw/             the original dataset files from Figshare (never edit these)
│   └── processed/       empty at first; notebooks 02 and 03 write the cleaned data here
├── results/
│   └── audit/           at first only the record of the original download; the notebooks add the rest
└── documents/           reading material (table below)
```

The two files in `results/audit/` (`file_provenance.json` and `figshare_api_record.json`) record where and how the
raw data were downloaded, with their checksums. Notebook 01 uses them to confirm that your copy of the data is
exactly the original. Please keep them.

### The documents

| File | What it is | When to read it |
|---|---|---|
| `Understanding_Our_Project.pdf` (and `.docx`, editable) | The beginner's guide to the whole project | First |
| `results_report.md` | The full technical results report, written by notebook 04 | After the guide, or while going through notebook 04 |
| `final_research_direction_and_contribution.md` | What the project claims and does not claim, the hypotheses and their outcome, next steps | Before the viva |
| `decisions_and_assumptions.md` | Every design decision and the reason for it | When someone asks "why did you do it this way?" |
| `FatigueAM_Project_Dossier.docx` | The project plan, written **before** the experiments: fatigue background, the dataset, the planned method, and a bank of 30 viva questions | Background reading. Where it differs from the results, the results documents are correct. |
| `dataset_provenance.md` | Where the data came from, the licence, and how to cite it | When you cite the data |

`.md` files are plain text with formatting marks. In VS Code, open one and press **Ctrl+Shift+V** to see it
formatted. The figures in `results_report.md` only appear after notebook 04 has run, because they are read from
`results/figures/`.

---

## 3. Software you need

- **Windows 10 or 11, 64-bit.**
- **Python 3.12 (64-bit)** from <https://www.python.org/downloads/windows/>. Choose the latest "Python 3.12.x"
  "Windows installer (64-bit)". During installation, tick **"Add python.exe to PATH"**. The project was tested
  with Python 3.12; other versions were not tested. Python 3.12 can be installed alongside other Python versions.
- **VS Code** (recommended) from <https://code.visualstudio.com/>, with the **Python** and **Jupyter** extensions
  from Microsoft (Extensions panel, Ctrl+Shift+X). If you prefer not to use VS Code, Jupyter in the web browser is
  installed by `requirements.txt` (option B in section 5).
- **Internet** for the one-time installation (about 150 MB of packages). Running the notebooks needs no internet.
- **About 1.5 GB of free disk space**: 0.6 GB for the packages in `.venv`, 0.3 GB for the outputs of the notebooks,
  and 0.1 GB for the folder itself.

---

## 4. Set up (one time)

**Step 1. Put the folder at a short path.** Create the folder `C:\Projects`. Right-click `Sister_Handover.zip`,
choose **Extract All…**, and type `C:\Projects` as the destination. Windows suggests
`…\Sister_Handover`; remove that last part, otherwise you get a folder inside a folder. This file should end up at
`C:\Projects\Sister_Handover\README.md`.

Avoid very deep folders and folders synced by OneDrive (Desktop and Documents often are). Windows limits file
paths to 260 characters, and some package files lie 161 characters deep inside `Sister_Handover`. So the path of the
`Sister_Handover` folder itself must be shorter than about 95 characters. `C:\Projects\Sister_Handover` is 27.

**Step 2. Open PowerShell in the folder.** In File Explorer, open `Sister_Handover`, click in the address bar,
type `powershell` and press Enter. Or open PowerShell and type:

```powershell
cd C:\Projects\Sister_Handover
```

**Step 3. Allow PowerShell to activate virtual environments (once per computer).** By default, Windows blocks this
with the message "running scripts is disabled on this system". Run this once, and answer `Y` if asked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

This lets scripts created on your own computer run (such as the activation script below). Scripts downloaded from
the internet still need a digital signature.

**Step 4. Create the virtual environment and install the packages.** Copy and paste these commands one at a time:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

- `py -3.12 -m venv .venv` creates a private Python environment in the folder `.venv`, so this project's packages
  do not mix with anything else on the computer.
- After `.venv\Scripts\activate`, the prompt starts with `(.venv)`. In every new PowerShell window you must
  activate again before running `python` commands. VS Code does this for you.
- If `py` is not recognised, use `python -m venv .venv` instead (first check that `python --version` shows 3.12).
- The last command downloads about 150 MB and takes a few minutes.

**Step 5. Check the installation:**

```powershell
python -c "import pandas, sklearn, lightgbm, matplotlib, openpyxl, pytest; print('All packages found')"
```

---

## 5. Open and run the notebooks

The four notebooks directly inside `notebooks/` are the official project notebooks. They are cleaned versions with
short explanations; their code and results are the same as in the originals. The original notebooks are kept only
as backups in `notebooks/archive_original/` and are not part of the workflow.

### Option A: VS Code (recommended)

1. Start VS Code, choose **File → Open Folder…** and select the `Sister_Handover` folder. If VS Code asks whether
   you trust the authors, choose **Yes**.
2. Open `notebooks/01_Dataset_Audit_and_EDA.ipynb`.
3. Click **Select Kernel** (top right), then **Python Environments…**, then **.venv (Python 3.12…)**. This is how you
   select the virtual environment as the Jupyter kernel. If `.venv` is not in the list, choose **Enter interpreter
   path…** and select `Sister_Handover\.venv\Scripts\python.exe`. VS Code remembers the choice.
4. Click **Run All**. Wait until every cell has finished: a running cell shows a spinning icon, and a finished cell
   shows a green tick and its time. Then save with **Ctrl+S**.
5. Repeat for notebooks 02, 03 and 04, in that order, with the same `.venv` kernel.

### Option B: Jupyter in the web browser

```powershell
cd C:\Projects\Sister_Handover
.venv\Scripts\activate
jupyter notebook
```

The browser opens. Go into `notebooks`, open notebook 01 and choose **Run → Run All Cells**. Notebooks started this
way automatically use the `.venv` environment (the kernel called "Python 3 (ipykernel)"). Then run 02, 03 and 04.
To stop Jupyter, close the browser tab and press **Ctrl+C** in PowerShell.

### Option C: all four at once, from PowerShell

```powershell
cd C:\Projects\Sister_Handover
.venv\Scripts\activate
python run_all.py
```

This runs 01 to 04 in order and saves the outputs into the notebooks. At the end it prints
`All requested notebooks ran without error.` To run only some notebooks, name them, for example
`python run_all.py 03 04`.

---

## 6. Execution order: always 01 → 02 → 03 → 04

Each notebook reads files written by the ones before it. The first time, run all four in order. Afterwards you
can re-run any single notebook, because the files it needs already exist.

| Notebook | What it does | Reads | Writes | Time on our test PC |
|---|---|---|---|---|
| **01** Dataset Audit and EDA | Checks the raw files against the download record, then explores the four sheets of the workbook: sizes, missing values, duplicates, first S-N plots | `data/raw/`, `results/audit/file_provenance.json` | `results/audit/`, `results/figures/01_*` | 17 s |
| **02** Data Preparation and Feature Engineering | Selects the room-temperature S-N cohort step by step, builds the input features, checks how temperatures were recorded | `data/raw/`, `results/audit/confirmed_columns.json` (from 01) | `data/processed/` (modelling dataset), `results/audit/`, `results/figures/02_*` | 10 s |
| **03** Model Development and Training | Builds the folds for schemes A, B and C, trains the five models with 5-fold cross-validation (seed 42, plus extra seeds and sensitivity runs), saves every prediction, trains the final models, runs the 42 tests | `data/processed/` (from 02) | `data/processed/folds.csv`, `results/predictions/`, `results/metrics/`, `models/`, `results/figures/03_*` | 6.5 min (the longest) |
| **04** Model Evaluation and Research Findings | Computes the errors and their confidence intervals, the sensitivity analyses and investigations A and B, then writes the results report, the paper figures and the guide's figures | the outputs of 01–03 | `results/metrics/`, `results/figures/04_*`, `results/figures/beginner/`, `results/paper/`, `reports/results_report.md` | 27 s |

Nothing is downloaded while the notebooks run. Notebook 01 would only try to download from Figshare if a raw file
were missing.

---

## 7. Where the data are

- `data/raw/FatigueData-AM2022.xlsx` (7.0 MB): the database as an Excel workbook. This is what the notebooks
  analyse.
- `data/raw/FatigueData-AM2022.json` (69.6 MB): the same database in JSON form. Notebook 02 uses it for one check
  on how test temperatures were recorded.
- **Do not open these two files in Excel and save them.** Notebook 01 checks that they are byte-for-byte the
  downloaded files (SHA-256 checksum) and stops if they changed. To look at the workbook, open a copy.
- `data/processed/` is filled by notebooks 02 and 03. `modelling_dataset.csv` is the cleaned table the models
  learn from.

**Licence and citation.** The dataset is published under CC BY 4.0. Please cite it (see the end of this file).

---

## 8. Where the outputs are saved

All outputs are written inside this folder. The folders are created automatically.

| Folder | What appears there |
|---|---|
| `results/figures/` | All figures as PNG files, named by notebook (`01_…`, `02_…`, `03_…`, `04_…`) |
| `results/figures/beginner/` | The five figures of the beginner's guide (PNG at 300 dpi, and PDF) |
| `results/paper/` | Figures and tables numbered for a paper (`Figure_1…`, `Table_1…`), with a README |
| `results/metrics/` | Tables of errors, intervals and sensitivity checks (CSV), and `test_report.txt` from notebook 03 |
| `results/predictions/` | Every cross-validation prediction (CSV) |
| `results/audit/` | The data checks of notebooks 01 and 02 |
| `models/` | The final trained models (`.joblib`) |
| `data/processed/` | The cleaned data and the fold assignment |
| `reports/results_report.md` | A fresh copy of the results report (the same as `documents/results_report.md`) |

Together they take about 310 MB.

---

## 9. How to check that it worked

- **Notebook 01** prints: `Both raw files match the provenance record (SHA-256) and the MD5 published by Figshare.`
- **Notebook 03**: the test report near the end finishes with `42 passed`.
- **Notebook 04**, last section, prints:
  `LightGBM RMSE of log10(Nf): random points 0.673, curves held out 0.787, articles held out 0.924 (ratio 1.37)`.
  These are the numbers in the guide.
- `run_all.py` ends with `All requested notebooks ran without error.`

Running again gives the same reported numbers. Some saved tables can differ in the 15th or 16th decimal place,
because the Random Forest adds up its trees' predictions in parallel, in a slightly different order each run. For
the same reason, in `results/metrics/model_effect_directions.csv` a few values that are practically zero can change
sign. No reported number changes.

---

## 10. If something goes wrong

First, read the **last line** of the red error message. It usually names the missing package or file.

| Message | What it means | What to do |
|---|---|---|
| `ModuleNotFoundError: No module named 'lightgbm'` (or `pandas`, `sklearn`, …) | The notebook is running with a Python that does not have the packages, usually because the wrong kernel is selected | Select the `.venv` kernel (top right in VS Code) and run again. If it still fails, activate `.venv` in PowerShell and run `python -m pip install -r requirements.txt` again, then restart the kernel. |
| `ModuleNotFoundError: No module named 'data_utils'` (or `feature_utils`, `model_utils`, `evaluation_utils`, `visualization`) | The notebook cannot find the `src/` folder | Keep the folder structure as it is: the notebooks must stay in `notebooks/`, next to `src/`. |
| `FileNotFoundError: … data\processed\modelling_dataset.csv` (or `results\predictions\…`, `models\…`) | An earlier notebook has not been run yet | Run the notebooks in order, 01 → 04. |
| `AssertionError: … differs from the provenance record` (notebook 01) | A raw data file has changed, for example because it was saved in Excel | Copy the two files in `data/raw/` again from the ZIP. |
| Notebook 01 tries to download, or shows `URLError` | A raw data file is missing from `data/raw/` | Copy it back from the ZIP. (Otherwise it is downloaded from Figshare, which needs internet.) |
| `AssertionError: Leakage tests failed` (notebook 03) | One of the 42 checks failed; the output above names it | This normally only happens if data files were changed. Extract the ZIP again and run from notebook 01. |
| `running scripts is disabled on this system` | PowerShell blocks the activation script | Do step 3 of section 4. If the computer does not allow it (some college or office computers), skip activation and type `.venv\Scripts\python.exe` wherever the instructions say `python`, for example `.venv\Scripts\python.exe run_all.py`. VS Code does not need activation. |
| `'py' is not recognized`, or `python` opens the Microsoft Store | Python is not installed, or not on the PATH | Install Python 3.12 from python.org with "Add python.exe to PATH" ticked, then open a new PowerShell window. |
| `pip` fails with "No such file or directory" or a very long path | The folder is too deep for Windows | Move the folder to `C:\Projects\Sister_Handover`, delete its `.venv` folder, and repeat step 4 of section 4. |
| `.venv` is not offered as a kernel in VS Code | VS Code has not noticed the new environment yet | Press Ctrl+Shift+P, run **Developer: Reload Window**, or use **Enter interpreter path…** (section 5). |
| A notebook seems stuck | Notebook 03 trains many models | Wait. A running cell shows a spinning icon in VS Code (`[*]` in the browser). |

If nothing helps, extract the ZIP again into a new folder and repeat sections 4 and 5.

---

## 11. Good to know

- `notebooks/archive_original/` holds the original, uncleaned notebooks, unchanged, as a backup. `run_all.py` runs
  only the four official notebooks in `notebooks/` and never the archived copies.
- Some notebook text mentions files from the development version of the project (the folders `archive/`, `audit/`
  and `docs/`, and reports such as `reports/CHANGELOG.md`). They are not included because you do not need them.
  `reports/Understanding_Our_Project.docx`, mentioned in notebook 04, is `documents/Understanding_Our_Project.docx`
  here.
- The random seed (42) and all model settings are fixed in the code. Leave them unchanged if you want the reported
  numbers.
- If you want to experiment, work in a copy of the whole folder, so this one keeps the reference results.

---

## Data source and licence

Zhang, Z. & Xu, Z. Fatigue database of additively manufactured alloys. *Scientific Data* 10, 249 (2023).
<https://doi.org/10.1038/s41597-023-02150-x>

Dataset: FatigueData-AM2022, Figshare, version 2, <https://doi.org/10.6084/m9.figshare.22337629>, licence CC BY 4.0.
The download record (date, file sizes, checksums) is in `documents/dataset_provenance.md` and
`results/audit/file_provenance.json`.
