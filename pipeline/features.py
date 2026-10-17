"""Stage 3, Section 3.5: loads the extracted features.

Feature extraction runs in the notebook on a T4 GPU and is not repeated here:
  C1  frozen ResNet-50, native unclamped volumes     extract_vector()        cells 18, 19, 28
  C2  ResNet-50 with layer 4 fine-tuned per fold      train_fold_model()      cell 20
  C3  PyRadiomics, resampled and clamped volumes      extract_radiomics_df()  cell 21
The saved outputs of those cells are loaded below.
"""
import os
import pickle

import numpy as np
import pandas as pd

from .common import RESULTS_DIR, COMPARISON_DIR


def _require(path):
    if not os.path.exists(path):
        raise SystemExit(f"Missing {path}. This file is produced by the notebook (RadiomicsVsDL.ipynb).")
    return path


def run(master_pids):
    # C1: frozen ResNet-50 features, variant raw_no_clamp (notebook cell 28)
    X_c1 = np.load(_require(f"{COMPARISON_DIR}/raw_no_clamp_features.npy"))

    # notebook cell 20: load saved per-fold deep features
    C2_FEATS_PATH = f"{RESULTS_DIR}/c2_seeded_fold_features.pkl"
    fold_features = pickle.load(open(_require(C2_FEATS_PATH), "rb"))

    # notebook cell 21
    RAD_FEATS_PATH = f"{RESULTS_DIR}/radiomics_features.pkl"
    radiomics_df = pd.read_pickle(_require(RAD_FEATS_PATH))
    assert list(radiomics_df["patient_id"]) == list(master_pids)
    feature_cols = [c for c in radiomics_df.columns if c not in ("patient_id", "label")]
    X_rad = radiomics_df[feature_cols].values.astype(float)

    print(f"C1 frozen deep features:     {X_c1.shape[0]} nodules x {X_c1.shape[1]}")
    print(f"C2 fine-tuned deep features: {len(fold_features)} folds x {fold_features[0][0].shape[1]} "
          f"(train {fold_features[0][0].shape[0]}, test {fold_features[0][1].shape[0]} per fold)")
    print(f"C3 radiomic features:        {X_rad.shape[0]} nodules x {X_rad.shape[1]}")

    return dict(X_c1=X_c1, fold_features=fold_features, radiomics_df=radiomics_df,
                feature_cols=feature_cols, X_rad=X_rad)
