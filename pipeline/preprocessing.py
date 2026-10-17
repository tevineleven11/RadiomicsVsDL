"""Stage 2, Section 3.3: checks the preprocessed volumes and shows the ROI figure.

Notebook cells 15 and 17. Resampling to 1 mm (B-spline image, nearest-neighbour mask) and
clamping to -1000..400 HU are done in notebook cell 15, which needs the LIDC-IDRI DICOMs;
its outputs are the files in preprocessed/.
"""
import os
import random

import matplotlib.pyplot as plt
import numpy as np

from .common import PREP_DIR, show


def run(sampled_nodules):
    # notebook cell 15 (completeness check)
    def patient_done(pid):
        return all(os.path.exists(f"{PREP_DIR}/{pid}{s}")
                   for s in ("_raw_c1.npy", "_resampled_clamped_c23.npy", "_mask_c23.npy"))

    n_done = sum(patient_done(n["patient_id"]) for n in sampled_nodules)
    print(f"Preprocessing completeness check: {n_done}/{len(sampled_nodules)} patients present.")

    # notebook cell 17 (ROI Test)
    random.seed(7)  # different seed just for visualization sample selection

    N_SAMPLES_TO_VIEW = 8

    valid_samples = [n for n in sampled_nodules]
    benign_samples = [n for n in valid_samples if n["label"] == 0]
    malignant_samples = [n for n in valid_samples if n["label"] == 1]

    viz_samples = (
        random.sample(benign_samples, N_SAMPLES_TO_VIEW // 2) +
        random.sample(malignant_samples, N_SAMPLES_TO_VIEW // 2)
    )

    # changed: skipped if the files for the sampled patients are missing
    if not all(os.path.exists(f"{PREP_DIR}/{n['patient_id']}_mask_c23.npy") for n in viz_samples):
        print("  preprocessed/ files for the ROI sample not found: ROI figure skipped.")
        return

    fig, axes = plt.subplots(2, N_SAMPLES_TO_VIEW // 2, figsize=(4 * (N_SAMPLES_TO_VIEW // 2), 8))
    axes = axes.flatten()

    for ax, nodule_info in zip(axes, viz_samples):
        pid = nodule_info["patient_id"]

        # read directly from the saved preprocessed files no DICOM access needed
        resampled_vol = np.load(f"{PREP_DIR}/{pid}_resampled_clamped_c23.npy")
        mask = np.load(f"{PREP_DIR}/{pid}_mask_c23.npy")

        slice_areas = mask.sum(axis=(0, 1))
        display_idx = int(np.argmax(slice_areas))

        ct_slice = resampled_vol[:, :, display_idx]
        mask_slice = mask[:, :, display_idx]

        ax.imshow(ct_slice, cmap="gray")
        ax.contour(mask_slice, colors="red", linewidths=1.5)
        label_str = "Malignant" if nodule_info["label"] == 1 else "Benign"
        ax.set_title(f"{pid}\n{label_str}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    show("roi_test")
