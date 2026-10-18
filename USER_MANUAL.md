# User Manual

This manual explains how to operate the software for *Comparison of Deep Learning Features and Radiomic Features for Lung Cancer Classification*. It details what each part does, what it needs, what it produces and how to resolve common problems. For a short overview and installation summary, see `README.md`.

---

## 1. Overview

The software has two parts that run the same pipeline.

| Part | Runs on | What it does |
|---|---|---|
| `RadiomicsVsDL.ipynb` | Google Colab, T4 GPU | The complete pipeline: nodule selection, CT preprocessing, ResNet-50 feature extraction and fine-tuning, PyRadiomics extraction, classification and statistics. |
| `run_pipeline.py` (and `RadiomicsVsDL.exe`, built from it) | Any computer, CPU only | The notebook's pipeline stages, run from the outputs the notebook saved: sample and folds, ROI check, classification, statistics and the additional experiments. Feature extraction and fine-tuning are loaded, not repeated. |

Both print the results reported in the paper (Tables 1 to 4 and the values in Sections 4.1 to 4.5) and produce Figs. 2 to 4.

Notebook cells are numbered from 0 (the installs cell) in this manual and in the code comments. The differences between the notebook and `run_pipeline.py` are listed in `SOFTWARE_REPORT.md`.

---

## 2. Requirements

| Option | Requirements |
|---|---|
| Windows executable | Windows 10 or 11, 64-bit. |
| From source | Python 3.12 or 3.13 (64-bit), and internet access on the first run to install the packages in `requirements.txt`. |
| Notebook | A Google account, Google Colab with a T4 GPU runtime, and about 200 MB of Google Drive space for the repository. |

All options need the repository's data files in place: `final_dataset_manifest.csv`, `fold_splits.pkl`, `preprocessed/` and `results/`.

---

## 3. Running the pipeline stages on a computer

### 3.1 Windows executable

1. Download `RadiomicsVsDL-windows.zip` from the repository's Releases page.
2. Unzip it. Keep the folder structure: `RadiomicsVsDL.exe` must sit next to `results/`, `preprocessed/`, `final_dataset_manifest.csv` and `fold_splits.pkl`.
3. Double-click `RadiomicsVsDL.exe`.
4. A console window shows the stages as they run. When it finishes, press **Enter** to close it.

No installation or setup is needed. Because the executable is not code-signed, Windows may warn about it:

- Your browser may flag the download as uncommon or dangerous; choose **Keep**.
- If SmartScreen shows "Windows protected your PC", click **More info → Run anyway**.

### 3.2 From source

From the repository folder:

| System | Command |
|---|---|
| Windows | `run.bat` (or double-click it) |
| macOS / Linux | `./run.sh` |

The launcher:

1. creates a virtual environment in `.venv` if it does not exist,
2. installs or checks the pinned packages from `requirements.txt`,
3. runs `python run_pipeline.py`.

The first run may take a few minutes to install the packages. 

To run without the launcher, install the packages into your own environment and run the script directly:

```
python -m pip install -r requirements.txt
python run_pipeline.py
```
This installs the pinned versions into the current Python environment; use the launcher, or create a virtual environment first, to keep them separate from other projects.

### 3.3 Options

| Option | Effect |
|---|---|
| *(none)* | Prints all results and saves each figure to `figures/`. |
| `--show` | Also opens each figure in a window as it is produced. Close the window to continue. |

Example: `run.bat --show` or `python run_pipeline.py --show`.

### 3.4 Run time

With the saved predictions in `results/`, a full run takes about a minute. Most of that time is spent on the paired bootstrap (Stage 5) and the radiomic ablation (Stage 6). If saved predictions or the 10-seed results have been deleted (Section 6.2), recomputing them adds a few minutes.

---

## 4. The stages

The program starts by checking that the four required paths exist and prints `OK` or `MISSING` for each. It then runs six stages in order. Each stage prints a header naming the notebook cells its code comes from and the section of the paper it supports.

### Stage 1: Data (Section 3.2; `pipeline/data.py`; cells 9, 10, 12, 13)

**Reads:** `results/balanced_ceiling.json`, `final_dataset_manifest.csv`, `fold_splits.pkl`.

**Does:**
- loads the eligible pool counts and the balanced ceiling (the largest balanced sample possible with one nodule per patient),
- loads the 400 sampled nodules and checks there are 400 unique patients,
- loads the 10 stratified cross-validation folds and prints the class balance of each.

**Prints:** the pool (1,533 nodules: 1,068 benign, 465 malignant), the ceiling (662), 200 benign and 200 malignant, and for every fold 360 training nodules (180 per class) and 40 test nodules (20 per class).

### Stage 2: Preprocessing (Section 3.3; `pipeline/preprocessing.py`; cells 15, 17)

**Reads:** `preprocessed/`.

**Does:**
- checks that the three files every nodule needs are present (native volume, resampled and clamped volume, resampled mask),
- draws eight nodules (four benign, four malignant) with a fixed seed and plots the slice with the largest ROI area, with the consensus mask outlined in red.

**Prints:** `400/400 patients present`. **Figure:** `figures/roi_test.png`.

Resampling and clamping themselves run in the notebook (cell 15), because they need the original DICOM images.

### Stage 3: Feature extraction (Section 3.5; `pipeline/features.py`; cells 19, 20, 21, 28)

**Reads:** the feature files in `results/`.

**Does:** loads the features the notebook extracted on a T4 GPU:

| Condition | File | Shape |
|---|---|---|
| C1 frozen ResNet-50 | `hu_clamp_resampling_comparison/raw_no_clamp_features.npy` | 400 × 2048 |
| C2 fine-tuned ResNet-50 | `c2_seeded_fold_features.pkl` | 10 folds; 360 training and 40 test × 2048 per fold |
| C3 PyRadiomics | `radiomics_features.pkl` | 400 × 72 |

C2 has separate features for each fold, because a separate network was fine-tuned in each fold.

### Stage 4: Classification (Sections 3.5 and 4.1; `pipeline/classification.py`; cells 18 to 24)

**Does:** for each condition, in every fold, the reduction and classifier are fitted on the training nodules only and applied to the test nodules:

| Condition | Per-fold pipeline |
|---|---|
| C1, C2 | z-score → PCA (20 components) → random forest |
| C3 | z-score → LASSO (L1 logistic regression) → random forest |
| C4 | radiomics: z-score → LASSO; deep features: z-score; concatenated → random forest |

The random forest has 100 trees and seed 42 in all conditions. As in the notebook, saved predictions are loaded if present. Missing C1, C3 or C4 predictions are recomputed from the saved features and saved (Section 6.2). C2's predictions are always loaded: recomputing them would mean fine-tuning the ten networks again, which only the notebook can do.

**Prints:** accuracy, sensitivity, specificity, precision and AUC for C1 to C4 (Table 1), and the four confusion matrices. **Figures:** `figures/confusion_matrices.png` (Fig. 2) and `figures/bar_chart.png`.

### Stage 5: Statistics (Section 4.1; `pipeline/statistics.py`; cells 25, 26, 35)

**Does:**
- plots the ROC curve of each condition,
- runs DeLong's test for correlated ROC curves on all six pairs of conditions,
- computes 95% confidence intervals for each difference in AUC by paired stratified bootstrap (2000 resamples, seed 42).

**Prints:** the DeLong table (ΔAUC, z, p) and the bootstrap table (ΔAUC, 95% CI, half-width, whether it contains 0); together these form Table 2. **Figure:** `figures/roc_curves.png` (Fig. 3).

### Stage 6: Additional experiments (Sections 4.2 to 4.5; `pipeline/experiments.py`; cells 28 to 34, 36)

| Experiment | Paper | Prints |
|---|---|---|
| Preprocessing sensitivity of the frozen extractor: native or resampled, with or without HU clamping | Table 3 | AUC of each configuration, mean ± SD over 10 classifier seeds, DeLong p-values against the native unclamped configuration, and clamping alongside resampling |
| Matched preprocessing: frozen versus fine-tuned on the same resampled, clamped volumes | Section 4.3 | ΔAUC and p |
| Matched-dimensionality fusion (deep features reduced to 20 components before fusion) | Table 4 | metrics and p-values against C2, C3 and C4 |
| Size and shape ablation of the radiomic features | Section 4.4 | AUC and p without the 10 size features and without the 14 shape features |
| LASSO feature counts | Section 4.5 | features selected in each fold (20 to 26) |
| Error overlap between conditions | Section 4.5 | misclassified by both, by either, and the overlap |
| PCA projection of each feature space | Fig. 4 | variance explained by the first two components |

As in the notebook, the predictions of the preprocessing variants and of the matched fusion, and the 10-seed results (`ten_seed_results.pkl`), are loaded if present and recomputed from the saved features if missing. The ablation, LASSO counts, error overlap and PCA are always computed.

**Figure:** `figures/fig4_feature_space.png` (Fig. 4).

---

## 5. Outputs

| Output | Location |
|---|---|
| All tables and values | printed in the console |
| Figures | `figures/roi_test.png`, `confusion_matrices.png`, `bar_chart.png`, `roc_curves.png`, `fig4_feature_space.png` |
| Recomputed predictions (only if deleted) | written back to `results/` |

To keep a copy of the console output from source, redirect it to a file:

```
python run_pipeline.py > output.txt
```

---

## 6. Using the notebook

### 6.1 Running it

The notebook needs the repository's data files, so the whole repository folder is placed on Google Drive, not only the notebook.

1. Download the repository and upload the whole folder to Google Drive (drive.google.com → **New → Folder upload**).
2. In Drive, open `RadiomicsVsDL.ipynb` with **Google Colaboratory** and select a **T4 GPU** runtime (**Runtime → Change runtime type**).
3. In cell 3, set `BASE` to the folder's path on Drive. A folder downloaded from GitHub as a ZIP is named `RadiomicsVsDL-main` by default, so the path is usually `"/content/drive/MyDrive/RadiomicsVsDL-main"`.
4. Click **Runtime → Run all** and allow access to Google Drive when asked.

Cell 0 installs PyTorch 2.11.0+cu126 (the build the reported results were produced with), pylidc, PyRadiomics, SimpleITK and MLstatkit. Cell 1 checks the PyTorch build and stops with “Wrong PyTorch build: restart the session” if a different build is still loaded; in that case click **Runtime → Restart session** and **Run all** again. 

### 6.2 What runs and what loads

Each step that produces a file checks whether the file already exists:

| If the file exists | If it is deleted |
|---|---|
| the step loads it | the step runs and saves it |

This applies to `final_dataset_manifest.csv` (sample), `fold_splits.pkl` (folds), `preprocessed/` (preprocessing), each `*_features` file (feature extraction and fine-tuning), each `*_probs.npy` file (classification) and `ten_seed_results.pkl` (10-seed check). For C2, deleting `c2_seeded_probs.npy` or `c2_seeded_fold_features.pkl` makes the notebook fine-tune the ten networks again.

`run_pipeline.py` follows the same rule for the C1, C3 and C4 predictions, the preprocessing variants, the matched fusion and the 10-seed results. Since it cannot extract features or fine-tune, it stops with an error if a feature file or C2's predictions are missing.

Approximate times on a T4 GPU:

| Step | Time |
|---|---|
| Loading all saved outputs | several minutes, mostly grouping the LIDC-IDRI annotations with pylidc (cells 5 and 7) |
| C1 frozen feature extraction | a few minutes |
| C2 fine-tuning (ten networks) | about 45 minutes |
| C3 PyRadiomics extraction | a few minutes |

To rebuild `preprocessed/` from the original images, download LIDC-IDRI from The Cancer Imaging Archive, set `DATASET_DIR` in cell 3, and delete `preprocessed/` and `results/`. Keep `final_dataset_manifest.csv` and `fold_splits.pkl` so that the same 400 nodules and folds are used.

### 6.3 Main cell groups

| Cells | Purpose | Paper |
|---|---|---|
| 0 to 4 | installs, imports, Drive, paths, preprocessing check | |
| 5 to 9 | labels, exclusions, size and connectivity filters, balanced ceiling | Section 3.2 |
| 10 to 14 | sample, DICOM copy, folds, consensus masks | Section 3.2 |
| 15 to 17 | preprocessing, fragmentation guard, ROI figure | Section 3.3 |
| 18 to 24 | shared components, C1 to C4, confusion matrices, bar chart | Sections 3.5, 4.1 |
| 25, 26, 35 | ROC curves, DeLong tests, bootstrap CIs | Section 4.1 |
| 27 to 30 | preprocessing variants, 10-seed check, DeLong, matched preprocessing | Sections 4.2, 4.3 |
| 31 to 34 | matched fusion, ablation, LASSO counts, error overlap | Sections 4.4, 4.5 |
| 36 | PCA projection | Fig. 4 |

---

## 7. Troubleshooting

| Problem | Cause and solution |
|---|---|
| `MISSING` next to a path at the start of the output, followed by a `FileNotFoundError` | A data file or folder is not where the program expects it. `results/`, `preprocessed/`, `final_dataset_manifest.csv` and `fold_splits.pkl` must be in the same folder as `run_pipeline.py` (or `RadiomicsVsDL.exe`). |
| `Missing ... This file is produced by the notebook`, or `FileNotFoundError` for `c2_seeded_probs.npy` | A feature file or C2's predictions are missing from `results/`. Restore them from the repository or regenerate them with the notebook. |
| `ModuleNotFoundError` (for example `No module named 'matplotlib'`) | The script was run with a Python that does not have the packages. Use `run.bat` / `run.sh`, or run `python -m pip install -r requirements.txt` first. If a `.venv` folder was left by an interrupted install, delete it and run the launcher again. |
| The launcher reports that Python is required, or package installation fails | Install 64-bit Python 3.12 or 3.13 from python.org and tick "Add Python to PATH". Check with `python --version`. Typing `python` alone opens the Python prompt (`>>>`); type `exit()` to leave it. |
| The browser flags the download, or "Windows protected your PC" appears | The executable is not code-signed. In the browser choose **Keep**; if SmartScreen appears, click **More info → Run anyway**. |
| ROI figure skipped | The files for the eight sampled patients are not in `preprocessed/`. The other stages are unaffected. |
| Figure windows do not open | Figures open only with `--show`; without it they are saved to `figures/`. |
| After deleting `*_probs.npy`, some p-values differ slightly from the paper | When predictions are recomputed on a different computer, some p-values may differ slightly from those reported. In testing, recomputed p-values differed by up to about 0.02, though no comparison changed from significant to non-significant or the reverse. To print the reported values, restore the repository's files. |
| Notebook: "Wrong PyTorch build: restart the session" | The pinned PyTorch build was installed but a different one is still loaded. Restart the session and run all again. |
| Notebook: files not found | `BASE` in cell 3 does not point to the repository's folder on Drive. |
| Notebook: deep learning results differ from the paper | The runtime is not a T4 GPU, or the PyTorch build is not 2.11.0+cu126. A different GPU or CUDA build may give different results for C1 and C2. |
