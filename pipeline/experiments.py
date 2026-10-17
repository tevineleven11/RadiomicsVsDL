"""Stage 6, Sections 4.2-4.5: additional experiments.

Notebook cells 28-34 and 36: preprocessing sensitivity of the frozen extractor (with the
10-seed check), matched-dimensionality fusion, the radiomic size and shape ablation, LASSO
feature counts, error overlap and the PCA projection of each feature space.
"""
import itertools
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from .classification import cv_predict, split_folds, pca_reduce, lasso_reduce, report, cached
from .statistics import delong_roc_test
from .common import RESULTS_DIR, COMPARISON_DIR, show


# notebook cell 31
def matched_fusion_reduce(X_tr, X_te, y_tr, seed=42):   # deep branch reduced to 20 PCs (Mei et al., 2023)
    (r_tr, d_tr), (r_te, d_te) = X_tr, X_te
    rr, dd = lasso_reduce(r_tr, r_te, y_tr), pca_reduce(d_tr, d_te, y_tr, seed)
    return np.hstack([rr[0], dd[0]]), np.hstack([rr[1], dd[1]])


def run(y_master, fold_splits, master_pids, X_c1, fold_features, X_rad, radiomics_df, feature_cols,
        c1_probs, c2_probs, c3_probs, c4_probs, fusion_folds):
    # notebook cell 28 (preprocessing comparison; features loaded from the notebook's run)
    VARIANTS = {"raw_no_clamp":       ("_raw_c1.npy",                "native"),
                "raw_clamp":          ("_raw_clamped_c1.npy",        "native"),
                "resampled_no_clamp": ("_resampled_c1.npy",          "resampled"),
                "resampled_clamp":    ("_resampled_clamped_c23.npy", "resampled")}

    variant_features = {}
    for name, (suffix, kind) in VARIANTS.items():
        path = f"{COMPARISON_DIR}/{name}_features.npy"
        X = variant_features[name] = np.load(path)
        probs = cached(f"{COMPARISON_DIR}/{name}", lambda: cv_predict(split_folds(X), y_master, pca_reduce), y_master)
        print(f"{name}: AUC = {roc_auc_score(y_master, probs):.3f}")

    # notebook cell 29 (10 seed check)
    SEED_RESULTS_PATH = f"{COMPARISON_DIR}/ten_seed_results.pkl"
    if os.path.exists(SEED_RESULTS_PATH):
        seed_results = pickle.load(open(SEED_RESULTS_PATH, "rb"))
    else:
        seed_results = {name: [roc_auc_score(y_master, cv_predict(split_folds(X), y_master, pca_reduce, seed))
                               for seed in range(1, 11)]
                        for name, X in variant_features.items()}
        pickle.dump(seed_results, open(SEED_RESULTS_PATH, "wb"))

    for name, aucs in seed_results.items():
        print(f"{name}: mean={np.mean(aucs):.3f} ± {np.std(aucs):.3f} (range {min(aucs):.3f}-{max(aucs):.3f})")

    # notebook cell 30 (DeLong for preprocessing configs)
    var_probs = {name: np.load(f"{COMPARISON_DIR}/{name}_probs.npy") for name in VARIANTS}
    ref = var_probs["raw_no_clamp"]
    for name in ["raw_clamp", "resampled_no_clamp", "resampled_clamp"]:
        a, b, _, p = delong_roc_test(y_master.astype(float), ref, var_probs[name])
        print(f"raw_no_clamp ({a:.3f}) vs {name} ({b:.3f}): ΔAUC = {b - a:+.3f}, p = {p:.4f}")

    a, b, _, p = delong_roc_test(y_master.astype(float), var_probs["resampled_no_clamp"], var_probs["resampled_clamp"])
    print(f"resampled: no clamp ({a:.3f}) vs clamp ({b:.3f}): p = {p:.4f}")

    a, b, _, p = delong_roc_test(y_master.astype(float), var_probs["resampled_clamp"], c2_probs)
    print(f"matched preprocessing: frozen ({a:.3f}) vs fine-tuned ({b:.3f}): ΔAUC = {b - a:+.3f}, p = {p:.4f}")

    print(f"Radiomic features per nodule: {len(feature_cols)}")

    # notebook cell 31 (matched-dimensionality fusion)
    print()
    c4m_probs = cached(f"{RESULTS_DIR}/c4_matched_seeded",
                       lambda: cv_predict(fusion_folds, y_master, matched_fusion_reduce), y_master)
    report("Matched fusion", c4m_probs)
    for name, probs in [("Fine-tuned DL", c2_probs), ("Radiomics", c3_probs), ("Original fusion", c4_probs)]:
        a, b, _, p = delong_roc_test(y_master.astype(float), c4m_probs, probs)
        print(f"  vs {name}: {a:.3f} vs {b:.3f}, p = {p:.4f}")

    # notebook cell 32 (section 4.4 radiomic size ablation test)
    # explicit size measures; ratio descriptors (Elongation, Flatness, Sphericity) are scale-invariant and kept
    SIZE_TERMS = ["LeastAxisLength", "MajorAxisLength", "MinorAxisLength", "Maximum2DDiameterSlice",
                  "Maximum2DDiameterRow", "Maximum2DDiameterColumn", "Maximum3DDiameter",
                  "MeshVolume", "VoxelVolume", "SurfaceArea"]
    size_cols  = [c for c in feature_cols if any(t in c for t in SIZE_TERMS)]
    shape_cols = [c for c in feature_cols if c.startswith("original_shape_")]
    subsets = {"full":     feature_cols,
               "no_size":  [c for c in feature_cols if c not in size_cols],
               "no_shape": [c for c in feature_cols if c not in shape_cols]}
    print()
    print(f"Removed {len(size_cols)} size features and {len(shape_cols)} shape features")

    # run Condition 3's pipeline on each subset
    abl = {name: cv_predict(split_folds(radiomics_df[cols].values.astype(float)), y_master, lasso_reduce)
           for name, cols in subsets.items()}

    # DeLong: each reduced set against the full set
    for name in ["no_size", "no_shape"]:
        a, b, _, p = delong_roc_test(y_master.astype(float), abl["full"], abl[name])
        print(f"full ({a:.3f}) vs {name} ({b:.3f}): ΔAUC = {abs(a - b):.3f}, p = {p:.4f}")

    # notebook cell 33 (Selected LASSO features)
    n_sel = [SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", random_state=42))
             .fit(StandardScaler().fit_transform(X_rad[tr]), y_master[tr]).get_support().sum() for tr, _ in fold_splits]
    print(f"LASSO-selected radiomic features per fold: {[int(n) for n in n_sel]} \n(range {min(n_sel)}-{max(n_sel)})")

    # notebook cell 34 (error overlaps)
    print()
    res = {}
    for c in ["c1","c2_seeded","c3","c4_seeded"]:
        t = np.load(f"{RESULTS_DIR}/{c}_true.npy")
        p = np.load(f"{RESULTS_DIR}/{c}_probs.npy")
        res[c] = (p >= 0.5).astype(int) != t.astype(int)   # error mask

    for a, b in itertools.combinations(res, 2):
        both = (res[a] & res[b]).sum()
        either = (res[a] | res[b]).sum()
        print(f"{a} vs {b}: both wrong {both}, either wrong {either}, overlap {both/either:.1%}")

    print("wrong in all four:", np.logical_and.reduce(list(res.values())).sum())
    print("wrong in none:", (~np.logical_or.reduce(list(res.values()))).sum())

    # notebook cell 36 (PCA Feature Space Visualization)
    # Align C2 rows to the master patient order (since folds are shuffled)
    X_c2 = np.zeros((len(master_pids), fold_features[0][1].shape[1]))
    for (tr, te), (_, f_te) in zip(fold_splits, fold_features):
        X_c2[te] = f_te
    X_c3 = X_rad.astype(np.float64)
    y    = radiomics_df["label"].astype(int).values
    X_c4 = np.concatenate([X_c2, X_c3], axis=1)

    # PCA + Plot
    conditions = [
        ("C1 - Frozen DL",     X_c1),
        ("C2 - Fine-tuned DL", X_c2),
        ("C3 - Radiomics",     X_c3),
        ("C4 - Fusion",        X_c4),
    ]

    print()
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for ax, (name, X) in zip(axes.flatten(), conditions):
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA(n_components=2, random_state=42)
        X_pca = pca.fit_transform(X_scaled)
        ev = pca.explained_variance_ratio_
        print(f"{name}: PC1 {ev[0]:.1%}, PC2 {ev[1]:.1%}")

        ax.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1], c="tab:blue",
                   alpha=0.5, s=20, edgecolors="none", label="Benign")
        ax.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1], c="tab:red",
                   alpha=0.5, s=20, edgecolors="none", label="Malignant")
        ax.set_title(f"{name}  (PC1 {ev[0]:.1%}, PC2 {ev[1]:.1%})", fontsize=11)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("PCA Projection of Feature Spaces (n = 400)", fontsize=13, y=1.00)
    plt.tight_layout()
    show("fig4_feature_space")
