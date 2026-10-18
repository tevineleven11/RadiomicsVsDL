# Software Report

This report details the choice of tools and the problems encountered in developing the software for *Comparison of Deep Learning Features and Radiomic Features for Lung Cancer Classification*, and describes how the system functions. Instructions for running the software are in `README.md` and `USER_MANUAL.md`.

---

## 1. How the system functions

### 1.1 Components

| Component | Role |
|---|---|
| `RadiomicsVsDL.ipynb` | The research pipeline. It selects the nodules, preprocesses the CT volumes, extracts frozen ResNet-50 features, fine-tunes ResNet-50 in each fold, extracts PyRadiomics features, classifies all four conditions and runs the statistical tests and additional experiments reported in the paper. It runs in Google Colab on a T4 GPU. |
| `results/`, `preprocessed/`, `final_dataset_manifest.csv`, `fold_splits.pkl` | The outputs the notebook saved: the 400-nodule sample, the 10 folds, the preprocessed volumes and masks, the extracted features and the predictions. |
| `pipeline/` and `run_pipeline.py` | The notebook's pipeline stages, taken cell by cell into one module per stage, run from the saved outputs on an ordinary computer. |
| `RadiomicsVsDL.exe` | `run_pipeline.py` packaged with Python and all required libraries, so that it runs on Windows without installation. |
| `run.bat`, `run.sh`, `requirements.txt` | Launchers that create a virtual environment with the pinned packages and run `run_pipeline.py` from source. |
| `.github/workflows/build-exe.yml` | Builds the executable on GitHub, checks it against the source and attaches it to each release. |

### 1.2 Pipeline

The pipeline runs in the same order in the notebook and in `run_pipeline.py`:

| Stage | Notebook cells | Module | Paper |
|---|---|---|---|
| 1. Data: labels, exclusions, filters, balanced ceiling, sample, folds | 5 to 13 | `data.py` | Section 3.2 |
| 2. Preprocessing: consensus masks, resampling, clamping, fragmentation guard, ROI figure | 14 to 17 | `preprocessing.py` | Section 3.3 |
| 3. Feature extraction: frozen ResNet-50, fine-tuned ResNet-50, PyRadiomics | 19 to 21, 28 | `features.py` | Section 3.5 |
| 4. Classification: per-fold z-score, PCA or LASSO, random forest; C1 to C4 | 18 to 24 | `classification.py` | Sections 3.5, 4.1 |
| 5. Statistics: ROC curves, DeLong tests, paired bootstrap CIs | 25, 26, 35 | `statistics.py` | Section 4.1 |
| 6. Additional experiments: preprocessing sensitivity, matched fusion, ablation, LASSO counts, error overlap, PCA | 27 to 34, 36 | `experiments.py` | Sections 4.2 to 4.5 |

### 1.3 Saved outputs

Every step that produces a file first checks whether the file exists. If it does, the step loads it; if not, the step runs and saves its output. This applies to the sample, the folds, the preprocessed volumes, the features, the predictions and the 10-seed results. It means a full run from the original images and a run that loads every saved output use the same code, and any single step can be repeated by deleting its file.

### 1.4 Relationship between the notebook and the executable

The analysis code in pipeline/ is taken from the notebook cell by cell; each block names the notebook cell it was taken from, and each change is marked # changed:. The differences, listed below, allow it to run on a standard computer without a GPU, Google Drive or the LIDC-IDRI images.

| | Notebook | `run_pipeline.py` / executable |
|---|---|---|
| Data location | `BASE` is a folder on Google Drive | `BASE` is the folder containing `run_pipeline.py` (or the executable) |
| Sample | each nodule is rebuilt from its LIDC annotation IDs with pylidc | the 400 nodules are read directly from `final_dataset_manifest.csv` |
| Labels, exclusions and filters (cells 5 to 8) | run with pylidc | not run; their results are loaded (`balanced_ceiling.json` and the manifest) |
| Consensus masks, preprocessing and fragmentation guard (cells 11, 14 to 16, 27) | run, or skipped when `preprocessed/` exists | not run; `preprocessed/` is read directly |
| Feature extraction and fine-tuning (cells 19 to 21, 28) | run when the saved features are missing | not run; the saved features are loaded |
| Figures | shown in the notebook | saved to `figures/`, and opened only with `--show` |
| ROI figure | always drawn | skipped if the eight sampled patients' files are missing |
| Code layout | one notebook | one module per stage in `pipeline/`; `fold_splits` and `y_master` are set once for the classification functions |

Every other line is unchanged, so the printed results are the same as the notebook's.

---

## 2. Choice of tools

| Tool | Used for | Reason |
|---|---|---|
| Python | All code | The language for the imaging, deep learning and machine learning libraries the study needed. |
| Google Colab (Tesla T4 GPU) | Running the notebook | Provided a GPU for ResNet-50 feature extraction and fine-tuning without local hardware, and Google Drive storage for the dataset and outputs. The Drive folder, containing the LIDC-IDRI images, the preprocessed volumes and the results, was shared within the group, so the group could run the notebook on the same data. |
| pylidc | Reading the LIDC-IDRI annotations, grouping them into nodules, consensus masks | Purpose-built for LIDC-IDRI; it includes the annotation database, so labels and masks can be derived without parsing the XML files. |
| SimpleITK, pydicom, SciPy | Image handling, resampling, connected-component analysis | Standard tools for medical images; `scipy.ndimage` provided B-spline and nearest-neighbour resampling and 26-connectivity labelling. |
| PyTorch and torchvision | ResNet-50 with ImageNet weights, frozen extraction and fine-tuning | Provides the pretrained ResNet-50 and control over which layers are trained and over deterministic cuDNN. |
| PyRadiomics | Radiomic feature extraction | An open-source, widely used radiomics extractor, used in the studies this work builds on. |
| scikit-learn | z-score, PCA, LASSO (L1 logistic regression), random forest, cross-validation, metrics | Provides every reduction step, the classifier and the folds with fixed seeds, so all conditions share one implementation. |
| MLstatkit | DeLong's test | Implements the fast DeLong algorithm for correlated ROC curves. The code checks that it used DeLong's method rather than falling back to a bootstrap. |
| NumPy, pandas, Matplotlib | Arrays, tables, figures | Standard scientific Python libraries. |
| Git, GitHub, GitHub Desktop | Version control and distribution | Holds the code, data and documentation in one public repository. |
| PyInstaller and GitHub Actions | Building the Windows executable | PyInstaller bundles Python and the libraries into a single file; building on GitHub's Windows machines means the executable is built and tested the same way for every release. |

---

## 3. Problems encountered

### 3.1 Reproducing the fine-tuned results

Re-running the notebook later in a new Colab session gave a different AUC for the fine-tuned condition (C2): 0.915 instead of 0.904 on a T4, and 0.895 when the session was assigned a different GPU. The code and seeds were unchanged. Colab's default PyTorch build had changed from the CUDA 12.6 build used for the paper to a CUDA 13.0 build.

- **Resolution:** the notebook now installs PyTorch 2.11.0+cu126 and torchvision 0.26.0+cu126 (cell 0) and stops if a different build is loaded (cell 1). With this build on a T4 GPU, C2 reproduced the reported AUC of 0.904.

### 3.2 Recomputed predictions on other computers

The saved features in results/ are loaded unchanged on every computer, but recomputing predictions from them did not always give identical probabilities. When the preprocessing-variant predictions were recomputed in a new Colab session, some probabilities differed by 0.01 to 0.03; on a Windows laptop, the p-values of comparisons involving C2 differed slightly. The inputs and seeds were identical, so the differences may be due to differences between the computers used; their exact source was not investigated. In testing, recomputed p-values differed by up to about 0.02, and no comparison changed from significant to non-significant or the reverse.

- **Resolution:** the predictions behind the paper are kept in `results/`, and both the notebook and `run_pipeline.py` load them when present (Section 1.3). An earlier version of `run_pipeline.py` recomputed every prediction; it was changed to use the notebook's loading step, which also brought its code closer to the notebook.

### 3.3 Masks split by resampling

Two consensus masks that were single regions at native resolution split into separate regions after resampling to 1 mm. This was found by visually inspecting the resampled ROI masks after preprocessing.

- **Resolution:** all resampled masks are rechecked with the same 26-connectivity rule (cell 16), and any that split are replaced from the eligible pool. This is described in Section 3.3 of the paper.

### 3.4 Storage

The LIDC-IDRI DICOM images (about 133 GB) were held on Google Drive during preprocessing but were too large to keep there long-term alongside the outputs, or to include in the repository.

- **Resolution:** the notebook's preprocessing output (`preprocessed/`, about 136 MB) is kept instead. With it and the manifest, the notebook runs every step after preprocessing without the original images, and the repository can include it.

### 3.5 Installing the libraries

- PyRadiomics has no pre-built package for Python 3.13 on PyPI, so the notebook installs it from its source repository (cell 0). This is one reason the executable does not include feature extraction.
- pylidc uses names that newer versions of NumPy and Python's `configparser` no longer provide; cell 1 restores them before importing pylidc.
- The packages in `requirements.txt` are pinned to the versions tested. These versions are available for Python 3.12 and 3.13 only, so the launchers require one of these. The executable bundles its own Python and is not affected.

### 3.6 Building and running the executable

- The executable is not code-signed, so browsers and Windows SmartScreen warn about it. The warnings and how to proceed are documented in the README and user manual.

---

## 4. Verification

- The notebook, `run_pipeline.py` (through `run.bat` and directly) and the executable were each run and their output compared with every value reported in the paper: Tables 1 to 4, the confusion matrices, the bootstrap confidence intervals, the 10-seed results, the ablation, the LASSO counts, the error overlap and the PCA variance. All matched.
- The notebook was also run in Google Colab from a fresh download of the repository, following the README, and matched the paper.
- For every release, the build workflow runs `run_pipeline.py` from source and then the built executable on a clean Windows machine, and stops if their outputs differ.

---

## 5. Limitations of the software

- `run_pipeline.py` and the executable do not extract features or fine-tune; they load the outputs of those steps. Repeating them requires the notebook, a GPU, PyTorch and PyRadiomics.
- Rebuilding `preprocessed/` requires the LIDC-IDRI DICOM images from The Cancer Imaging Archive, which are not included in the repository.
- Exact reproduction of the deep learning results was achieved on a T4 GPU with the pinned PyTorch build; a different GPU or build may give different results.
- The executable is built for Windows only; macOS and Linux users run from source.
