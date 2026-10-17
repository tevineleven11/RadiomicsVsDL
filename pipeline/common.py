"""Paths and figure handling shared by every stage (notebook cell 3)."""
import os
import sys

import matplotlib

# changed: BASE is the repository folder (or the folder holding the .exe) instead of Google Drive
BASE = os.path.dirname(os.path.abspath(sys.executable)) if getattr(sys, "frozen", False) \
    else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PREP_DIR         = f"{BASE}/preprocessed"
RESULTS_DIR      = f"{BASE}/results"
COMPARISON_DIR   = f"{RESULTS_DIR}/hu_clamp_resampling_comparison"
MANIFEST_PATH    = f"{BASE}/final_dataset_manifest.csv"
FOLD_SPLITS_PATH = f"{BASE}/fold_splits.pkl"
FIGURES_DIR      = f"{BASE}/figures"   # changed: figures are saved here

SHOW_FIGURES = False   # set by run_pipeline.py --show


def use_backend(show):
    global SHOW_FIGURES
    SHOW_FIGURES = show
    if not show:
        matplotlib.use("Agg")


def show(name):
    """changed: replaces plt.show(); saves the figure and opens it only with --show."""
    import matplotlib.pyplot as plt
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plt.savefig(f"{FIGURES_DIR}/{name}.png", dpi=200, bbox_inches="tight")
    print(f"  figure saved: figures/{name}.png")
    plt.show() if SHOW_FIGURES else plt.close()


def check_folders():
    print("Folders:")
    for p in [PREP_DIR, RESULTS_DIR, MANIFEST_PATH, FOLD_SPLITS_PATH]:
        print(("OK       " if os.path.exists(p) else "MISSING  ") + p)
