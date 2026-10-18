# Comparison of Deep Learning Features and Radiomic Features for Lung Cancer Classification

The study compares four feature representations for benign versus malignant lung nodule classification on the LIDC-IDRI dataset, under a shared evaluation framework.

| Condition | Features |
|---|---|
| C1 | Frozen ImageNet-pretrained ResNet-50 |
| C2 | ResNet-50 with layer 4 fine-tuned in each fold |
| C3 | PyRadiomics shape, first-order, GLCM and GLRLM features |
| C4 | Fusion of C2 and C3 |

Additional experiments test the frozen extractor's sensitivity to resampling and HU clamping, a size and shape ablation of the radiomic features, and a fusion with matched feature dimensionality.

---

## Contents of this repository

```
RadiomicsVsDL/
├── RadiomicsVsDL.ipynb          Full pipeline (Google Colab, T4 GPU): data, preprocessing,
│                                feature extraction, fine-tuning, classification and statistics
├── run_pipeline.py              Runs the pipeline stages on CPU from the saved outputs
├── pipeline/                    Code for each stage, taken from the notebook cell by cell
│   ├── data.py                  Stage 1  Section 3.2  sample, labels, folds, balanced ceiling
│   ├── preprocessing.py         Stage 2  Section 3.3  preprocessed volumes, ROI figure
│   ├── features.py              Stage 3  Section 3.5  loads the extracted features
│   ├── classification.py        Stage 4  Section 3.5  z-score, PCA, LASSO, random forest, C1-C4
│   ├── statistics.py            Stage 5  Section 4.1  ROC curves, DeLong tests, bootstrap CIs
│   ├── experiments.py           Stage 6  Sections 4.2-4.5  additional experiments
│   └── common.py                folder paths and figure saving
├── run.bat / run.sh             Launchers for Windows / macOS and Linux (run from source)
├── requirements.txt             Pinned packages for run_pipeline.py
├── final_dataset_manifest.csv   The 400 sampled nodules (patient, label, LIDC annotation IDs)
├── fold_splits.pkl              The 10 cross-validation splits used by every condition
├── preprocessed/                Cropped CT volumes and consensus masks for the 400 nodules
├── results/                     Extracted features and predictions produced by the notebook
├── USER_MANUAL.md               How to operate the software
└── SOFTWARE_REPORT.md           Tool choices, problems encountered and how the system works
```

---

## How to run

There are three ways to run the software. The first two need no GPU and no dataset download.

### 1. Windows executable

The executable runs with minimal configuration, including all required libraries.

1. Download `RadiomicsVsDL-windows.zip` from the [Releases](https://github.com/tevineleven11/RadiomicsVsDL/releases) page and unzip it.
2. Double-click `RadiomicsVsDL.exe`.

A console window shows each stage as it runs; figures are saved to the `figures` folder.

Although no setup is needed, Windows may warn about the executable because it is not code-signed. These warnings concern the file's unknown origin, not its behaviour:

* Your browser may flag the download as uncommon or dangerous; choose **Keep**.
* If SmartScreen shows "Windows protected your PC", click **More info → Run anyway**.

### 2. From source (Windows, macOS, Linux)

Requires Python 3.12 or 3.13. Clone or download this repository, then run:

```
cd <path to the repository folder>
run.bat            (Windows)
./run.sh           (macOS / Linux)
```

The launcher creates a virtual environment (`.venv`), installs the pinned packages from `requirements.txt` and runs `run_pipeline.py`. The first run needs internet access to install the packages. Add `--show` to open each figure in a window as it is produced.

### 3. Full pipeline in Google Colab (GPU)

The notebook contains the complete pipeline, including CT preprocessing, ResNet-50 feature extraction, fine-tuning and PyRadiomics extraction. It needs the repository's data files, so the whole repository folder is placed on Google Drive, not only the notebook.

1. Download this repository and upload the whole folder to Google Drive (drive.google.com → **New → Folder upload**).
2. In Drive, open `RadiomicsVsDL.ipynb` with **Google Colaboratory**, and select a **T4 GPU** runtime (**Runtime → Change runtime type**).
3. In cell 3, set `BASE` to the folder's path on Drive. A folder downloaded from GitHub as a ZIP is named `RadiomicsVsDL-main` by default, so the path is usually `"/content/drive/MyDrive/RadiomicsVsDL-main"`.
4. **Run all**, and allow access to Google Drive when asked. The first cell installs the pinned PyTorch build (2.11.0+cu126). If the import cell reports the wrong PyTorch build, restart the session and run all again.

---

## What the executable and `run_pipeline.py` do

`run_pipeline.py` runs the notebook's pipeline stages in order on a standard computer, using the outputs the notebook saved; feature extraction and fine-tuning, which need PyTorch, PyRadiomics and a GPU, run only in the notebook. The stages are described in `USER_MANUAL.md`, and the differences from the notebook in `SOFTWARE_REPORT.md`.

---

## Reproducibility

- **Environment.** The reported results were produced in Google Colab on a Tesla T4 GPU with PyTorch 2.11.0+cu126, torchvision 0.26.0+cu126 and scikit-learn 1.6.1. A different CUDA build or GPU may give different results for the deep learning conditions, so the notebook installs and checks this PyTorch build, and a T4 runtime should be selected.
- **Determinism.** A fixed seed (42) is used for sampling, fold generation, fine-tuning, dimensionality reduction and classification. Deterministic cuDNN is enabled and cuDNN benchmarking is disabled before any network is run.
- **Saved outputs.** `results/` holds the features and predictions behind every value reported in the paper, so the notebook, `run_pipeline.py` and the executable all print the reported values.

---

## Data

This study uses the **Lung Image Database Consortium and Image Database Resource Initiative (LIDC-IDRI)** collection, publicly available from The Cancer Imaging Archive (TCIA) under the Creative Commons Attribution 3.0 (CC BY 3.0) licence:

**Dataset**

- S. G. Armato III et al., "Data From LIDC-IDRI" [Data set], The Cancer Imaging Archive, 2015. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX

**Dataset paper**

- S. G. Armato III et al., "The Lung Image Database Consortium (LIDC) and Image Database Resource Initiative (IDRI): A Completed Reference Database of Lung Nodules on CT Scans," *Med. Phys.*, vol. 38, no. 2, pp. 915–931, 2011. https://doi.org/10.1118/1.3528204

The authors acknowledge the National Cancer Institute and the Foundation for the National Institutes of Health, and their critical role in the creation of the free publicly available LIDC/IDRI Database used in this study.

This repository does not include the original DICOM images, due to storage constraints; `preprocessed/` and `results/` are derived from them. To rebuild them from the original images, download LIDC-IDRI from TCIA, set `DATASET_DIR` in cell 3 of the notebook, and delete `preprocessed/` and `results/` before running the notebook. Keep `final_dataset_manifest.csv` and `fold_splits.pkl` so that the same 400 nodules and the same folds are used.

The files in this repository derived from LIDC-IDRI (`preprocessed/`, `results/`, `final_dataset_manifest.csv`) are shared under the same CC BY 3.0 licence.

---
