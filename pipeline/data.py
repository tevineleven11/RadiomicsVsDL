"""Stage 1, Section 3.2: balanced ceiling, the 400-nodule sample and the 10-fold splits.

Notebook cells 9, 10, 12 and 13. The notebook derives labels and applies the size and
connectivity filters with pylidc (cells 5-8) and saves the results used here.
"""
import json
import pickle

import numpy as np
import pandas as pd

from .common import RESULTS_DIR, MANIFEST_PATH, FOLD_SPLITS_PATH


def run():
    # notebook cell 9 (balanced ceiling, loaded from disk)
    CEILING_PATH = f"{RESULTS_DIR}/balanced_ceiling.json"
    with open(CEILING_PATH) as f:
        ceiling_info = json.load(f)
    print("Balanced ceiling loaded from disk.")
    for k, v in ceiling_info.items():
        print(f"  {k:24} {v}")

    # notebook cell 10 (manifest exists, so the saved sample is reloaded)
    TARGET_PER_CLASS = 200   # N = 400
    df_manifest = pd.read_csv(MANIFEST_PATH)
    # changed: the notebook rebuilds each nodule from its pylidc annotation IDs; here the manifest rows are used directly
    sampled_nodules = [{"patient_id": r["patient_id"], "label": int(r["label"]), "diameter": r["diameter"]}
                       for _, r in df_manifest.iterrows()]
    print(f"Loaded {len(sampled_nodules)} nodules from the manifest.")
    pids = [n["patient_id"] for n in sampled_nodules]
    assert len(pids) == len(set(pids)) == 2 * TARGET_PER_CLASS, "cohort must be 400 unique patients"
    print(f"  Benign: {sum(n['label'] == 0 for n in sampled_nodules)}, "
          f"Malignant: {sum(n['label'] == 1 for n in sampled_nodules)}")

    # notebook cell 12 (existing split loaded)
    with open(FOLD_SPLITS_PATH, "rb") as f:
        fold_data = pickle.load(f)
    master_pids = fold_data["pids_order"]
    fold_splits = fold_data["fold_splits"]
    print(f"Loaded existing fold splits: {len(fold_splits)} folds, {len(master_pids)} patients")

    # notebook cell 13 (Check Splits)
    pid_to_label_check = {n["patient_id"]: n["label"] for n in sampled_nodules}
    master_labels = np.array([pid_to_label_check[pid] for pid in master_pids])

    for i, (train_idx, test_idx) in enumerate(fold_splits):
        train_labels = master_labels[train_idx]
        test_labels = master_labels[test_idx]
        print(f"  Fold {i+1}: train={len(train_idx)} (benign={sum(train_labels==0)}, malignant={sum(train_labels==1)}), "
              f"test={len(test_idx)} (benign={sum(test_labels==0)}, malignant={sum(test_labels==1)})")

    # notebook cell 18 (labels in fold order)
    y_master = np.array([{n["patient_id"]: n["label"] for n in sampled_nodules}[p] for p in master_pids])

    return dict(ceiling_info=ceiling_info, sampled_nodules=sampled_nodules, master_pids=master_pids,
                fold_splits=fold_splits, y_master=y_master)
