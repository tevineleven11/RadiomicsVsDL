"""Stage 4, Sections 3.5 and 4.1: per-fold reduction and random forest for C1-C4.

The shared components are notebook cell 18; the conditions are cells 19-22 and the figures
cells 23-24. As in the notebook, cached(...) loads the saved predictions; deleting a *_probs.npy file
makes it recompute that condition from its features.
"""
import matplotlib.pyplot as plt
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler

from .common import RESULTS_DIR, show

fold_splits = None   # changed: module-level copies of the notebook's globals, set by setup()
y_master = None


def setup(splits, labels):
    global fold_splits, y_master
    fold_splits, y_master = splits, labels


# notebook cell 18 (shared components)
def zscore_reduce(X_tr, X_te, y_tr, seed=42): # fitted on training fold only, applied unchanged to test fold
    sc = StandardScaler().fit(X_tr)
    return sc.transform(X_tr), sc.transform(X_te)

def pca_reduce(X_tr, X_te, y_tr, seed=42):            # deep features, C1 and C2 (Zheng et al., 2026)
    Z_tr, Z_te = zscore_reduce(X_tr, X_te, y_tr)
    pca = PCA(n_components=20, random_state=seed).fit(Z_tr)
    return pca.transform(Z_tr), pca.transform(Z_te)

def lasso_reduce(X_tr, X_te, y_tr, seed=42):          # radiomic features, C3 and C4 (Brocki & Chung, 2023); default C = 1 as in Demircioglu (2023)
    Z_tr, Z_te = zscore_reduce(X_tr, X_te, y_tr)
    sel = SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear", random_state=42)).fit(Z_tr, y_tr)
    return sel.transform(Z_tr), sel.transform(Z_te)

def split_folds(X):                                   # fold-independent features (C1, C3)
    return [(X[tr], X[te]) for tr, te in fold_splits]

def cv_predict(fold_data, y, reduce, seed=42): # random forest, 100 trees (scikit-learn default), fitted per training fold; pooled out-of-fold predictions
    probs = np.zeros(len(y))
    for (tr, te), (X_tr, X_te) in zip(fold_splits, fold_data):
        Z_tr, Z_te = reduce(X_tr, X_te, y[tr], seed)
        rf = RandomForestClassifier(n_estimators=100, random_state=seed).fit(Z_tr, y[tr])
        probs[te] = rf.predict_proba(Z_te)[:, 1]
    return probs

def cached(stem, compute, y): # loads saved predictions if present, otherwise computes and saves them
    t, p = f"{stem}_true.npy", f"{stem}_probs.npy"
    if os.path.exists(t) and os.path.exists(p) and len(np.unique(np.load(t))) == 2:
        return np.load(p)
    probs = compute()
    np.save(t, y.astype(float)); np.save(p, probs)
    return probs

def report(name, probs, y=None): # metrics at the default 0.5 threshold; AUC from probabilities
    y = y_master if y is None else y   # changed: default bound at call time
    pred = (probs >= 0.5).astype(int)
    print(f"{name} | Acc {accuracy_score(y, pred):.3f} | Sens {recall_score(y, pred):.3f} | "
          f"Spec {recall_score(y, pred, pos_label=0):.3f} | Prec {precision_score(y, pred):.3f} | "
          f"AUC {roc_auc_score(y, probs):.3f}")


# notebook cell 22
def fusion_reduce(X_tr, X_te, y_tr, seed=42):
    (r_tr, d_tr), (r_te, d_te) = X_tr, X_te
    rr = lasso_reduce(r_tr, r_te, y_tr)                # Z-score Normalisation + LASSO Feature Selection (radiomics)
    dd = zscore_reduce(d_tr, d_te, y_tr)               # Z-score Normalisation (deep features)
    return np.hstack([rr[0], dd[0]]), np.hstack([rr[1], dd[1]])   # Feature Fusion (radiomics first, then deep)


def run(X_c1, fold_features, X_rad):
    # notebook cell 19
    def compute_c1():   # changed: starts from the saved C1 features instead of extracting them
        return cv_predict(split_folds(X_c1), y_master, pca_reduce)   # z-score -> PCA (20) -> random forest

    c1_probs = cached(f"{RESULTS_DIR}/c1", compute_c1, y_master)
    c1_true, c1_preds = y_master, (c1_probs >= 0.5).astype(int)
    report("C1 Frozen DL", c1_probs)

    # notebook cell 20
    C2_STEM = f"{RESULTS_DIR}/c2_seeded"
    c2_probs = np.load(f"{C2_STEM}_probs.npy")
    c2_true, c2_preds = y_master, (c2_probs >= 0.5).astype(int)
    report("C2 Fine-tuned DL", c2_probs)

    # notebook cell 21
    c3_probs = cached(f"{RESULTS_DIR}/c3", lambda: cv_predict(split_folds(X_rad), y_master, lasso_reduce), y_master) # z-score -> LASSO -> RF
    c3_true, c3_preds = y_master, (c3_probs >= 0.5).astype(int)
    report("C3 Radiomics", c3_probs)

    # notebook cell 22
    # Radiomics features from Condition 3 (X_rad) and Deep Features from Condition 2 (fold_features), per fold
    fusion_folds = [((X_rad[tr], feat_train_dl), (X_rad[te], feat_test_dl))
                    for (tr, te), (feat_train_dl, feat_test_dl) in zip(fold_splits, fold_features)]

    # Cross Validation with Random Forest Classifier; Save Results, or Load Results if already computed
    c4_probs = cached(f"{RESULTS_DIR}/c4_seeded", lambda: cv_predict(fusion_folds, y_master, fusion_reduce), y_master)
    c4_true, c4_preds = y_master, (c4_probs >= 0.5).astype(int)

    # Final Performance
    report("C4 Fusion", c4_probs)

    # notebook cell 23 (Confusion Matrix)
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))

    conditions = [
        ("Condition 1", c1_true, c1_preds),
        ("Condition 2", c2_true, c2_preds),
        ("Condition 3", c3_true, c3_preds),
        ("Condition 4", c4_true, c4_preds)
    ]

    cms = []

    for ax, (title, y_true, y_pred) in zip(axes.ravel(), conditions):

        cm = confusion_matrix(y_true, y_pred)
        cms.append((title, cm))

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=["Benign", "Malignant"]
        )

        disp.plot(ax=ax, cmap="Blues", colorbar=False, text_kw={"fontsize": 32})
        ax.set_title(title, fontsize=22)
        ax.tick_params(labelsize=18)
        ax.set_xlabel('')
        ax.set_ylabel('')

    plt.tight_layout()
    show("confusion_matrices")

    for title, cm in cms:
        print(f"{title}:\n{cm}\n")

    # notebook cell 24 (Bar Chart)
    metrics = pd.DataFrame({

        "Condition":[
            "C1",
            "C2",
            "C3",
            "C4"
        ],

        "Accuracy":[
            accuracy_score(c1_true,c1_preds),
            accuracy_score(c2_true,c2_preds),
            accuracy_score(c3_true,c3_preds),
            accuracy_score(c4_true,c4_preds)
        ],

        "Sensitivity":[
            recall_score(c1_true,c1_preds),
            recall_score(c2_true,c2_preds),
            recall_score(c3_true,c3_preds),
            recall_score(c4_true,c4_preds)
        ],

        "Specificity":[
            recall_score(c1_true,c1_preds,pos_label=0),
            recall_score(c2_true,c2_preds,pos_label=0),
            recall_score(c3_true,c3_preds,pos_label=0),
            recall_score(c4_true,c4_preds,pos_label=0)
        ],

        "Precision":[
            precision_score(c1_true,c1_preds),
            precision_score(c2_true,c2_preds),
            precision_score(c3_true,c3_preds),
            precision_score(c4_true,c4_preds)
        ],

        "AUC":[
            roc_auc_score(c1_true,c1_probs),
            roc_auc_score(c2_true,c2_probs),
            roc_auc_score(c3_true,c3_probs),
            roc_auc_score(c4_true,c4_probs)
        ]

    })

    metrics.set_index("Condition").plot(
        kind="bar",
        figsize=(10,6)
    )

    plt.ylim(0,1)

    plt.ylabel("Score")

    plt.title("Performance Comparison")

    plt.grid(axis="y")

    show("bar_chart")

    return dict(c1_probs=c1_probs, c2_probs=c2_probs, c3_probs=c3_probs, c4_probs=c4_probs,
                fusion_folds=fusion_folds)
