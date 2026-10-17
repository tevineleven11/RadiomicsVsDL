"""Stage 5, Section 4.1: ROC curves, pairwise DeLong tests and paired bootstrap CIs.

Notebook cells 25, 26 and 35.
"""
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MLstatkit import Delong_test
from sklearn.metrics import roc_auc_score, roc_curve, auc as sk_auc

from .common import show


# notebook cell 26
# DeLong's test for correlated ROC curves (DeLong et al., 1988; fast algorithm of Sun & Xu, 2014), via MLstatkit
def delong_roc_test(true_labels, probs_a, probs_b):
    z, p, _, _, auc_a, auc_b, info = Delong_test(true_labels, probs_a, probs_b)
    assert info["method"] == "delong", "MLstatkit fell back to bootstrap"
    return auc_a, auc_b, abs(z), p


# notebook cell 35
# 95% bootstrap CIs: paired stratified resampling of the 400 pooled out-of-fold predictions, 2000 resamples, seed 42
def paired_bootstrap_aucs(y, probs, B=2000, seed=42):
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    idx = [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(B)]  # same nodules for every condition
    return {name: np.array([roc_auc_score(y[i], p[i]) for i in idx]) for name, p in probs.items()}


def run(y_master, c1_probs, c2_probs, c3_probs, c4_probs):
    c1_true = c2_true = c3_true = c4_true = y_master

    # notebook cell 25 (ROC Curve)
    plt.figure(figsize=(8,6))

    conditions = [
        ("Condition 1", c1_true, c1_probs),
        ("Condition 2", c2_true, c2_probs),
        ("Condition 3", c3_true, c3_probs),
        ("Condition 4", c4_true, c4_probs)
    ]

    for name, y_true, y_prob in conditions:

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = sk_auc(fpr, tpr)

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{name} (AUC={roc_auc:.3f})"
        )

    plt.plot([0,1],[0,1],'k--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")

    plt.grid(True)

    show("roc_curves")

    # notebook cell 26
    # run pairwise DeLong tests across all four conditions
    cond_probs = {"C1": c1_probs, "C2": c2_probs, "C3": c3_probs, "C4": c4_probs}

    rows = []
    for a, b in combinations(cond_probs, 2):
        auc_a, auc_b, z, p = delong_roc_test(y_master, cond_probs[a], cond_probs[b])
        rows.append({"Comparison": f"{a} vs {b}", "AUC A": f"{auc_a:.3f}", "AUC B": f"{auc_b:.3f}",
                     "ΔAUC": f"{auc_a - auc_b:+.3f}", "z": f"{z:.3f}", "p-value": f"{p:.4f}"})
    print(pd.DataFrame(rows).to_string(index=False))

    # notebook cell 35
    probs = {"C1": c1_probs, "C2": c2_probs, "C3": c3_probs, "C4": c4_probs}
    boot  = paired_bootstrap_aucs(y_master, probs)

    rows = []
    for a, b in combinations(probs, 2):
        lo, hi = np.quantile(boot[a] - boot[b], [0.025, 0.975])
        rows.append({"Comparison": f"{a} vs {b}",
                     "ΔAUC": f"{roc_auc_score(y_master, probs[a]) - roc_auc_score(y_master, probs[b]):+.3f}",
                     "95% CI": f"[{lo:+.4f}, {hi:+.4f}]",
                     "Half-width": f"{(hi - lo) / 2:.3f}",
                     "Contains 0": lo <= 0 <= hi})
    print()
    print(pd.DataFrame(rows).to_string(index=False))
