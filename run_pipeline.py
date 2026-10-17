import argparse
import sys

from pipeline import common


def stage(number, title, cells, section):
    print(f"\n{'=' * 80}\nStage {number}: {title}\n  notebook cells {cells} | paper {section}\n{'=' * 80}")


def main(show_figures):
    common.use_backend(show_figures)   # must run before pyplot is imported by the stages
    from pipeline import data, preprocessing, features, classification, statistics, experiments

    common.check_folders()

    stage(1, "Data: sample, labels and cross-validation splits", "9, 10, 12, 13", "Section 3.2")
    d = data.run()
    classification.setup(d["fold_splits"], d["y_master"])

    stage(2, "Preprocessing: ROI check on the preprocessed volumes", "15, 17", "Section 3.3")
    preprocessing.run(d["sampled_nodules"])

    stage(3, "Feature extraction (loaded: extracted in the notebook on a T4 GPU)", "19, 20, 21, 28", "Section 3.5")
    f = features.run(d["master_pids"])

    stage(4, "Classification: per-fold reduction and random forest", "18-24", "Sections 3.5, 4.1")
    c = classification.run(f["X_c1"], f["fold_features"], f["X_rad"])

    stage(5, "Statistics: ROC curves, DeLong tests, bootstrap CIs", "25, 26, 35", "Section 4.1")
    statistics.run(d["y_master"], c["c1_probs"], c["c2_probs"], c["c3_probs"], c["c4_probs"])

    stage(6, "Additional experiments", "28-34, 36", "Sections 4.2-4.5")
    experiments.run(d["y_master"], d["fold_splits"], d["master_pids"], f["X_c1"], f["fold_features"],
                    f["X_rad"], f["radiomics_df"], f["feature_cols"], c["c1_probs"], c["c2_probs"],
                    c["c3_probs"], c["c4_probs"], c["fusion_folds"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true", help="open each figure in a window")
    main(parser.parse_args().show)
    if getattr(sys, "frozen", False):   # keeps the console window open when the .exe is double-clicked
        input("\nPress Enter to close.")