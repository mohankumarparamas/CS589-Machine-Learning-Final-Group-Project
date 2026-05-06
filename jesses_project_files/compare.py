import os

import matplotlib.pyplot as plt

from k_nn import K_VALUES, cross_validate_knn
from random_forest import (
    MAX_DEPTH,
    MIN_GAIN,
    MIN_SIZE_FOR_SPLIT,
    NTREE_VALUES,
    NUM_FOLDS,
    cross_validate_random_forest,
    load_dataset,
)

OUTPUT_PATH = "figures/comparison_summary.txt"
COMPARISON_PLOT = "figures/comparison_rf_vs_knn.pdf"

DATASET_PATH = "datasets/rice.csv"
DATASET_LABEL = "Dataset #1.3 (Rice)"


def fmt(mean, std):
    return f"{mean:.4f} +/- {std:.4f}"


def run_rf_sweep(attributes, classes, is_numerical):
    """Sweep ntree, return per-setting CV results."""
    rows = []
    print("\n--- Random Forest sweep ---")
    print(
        f"  Stopping criteria: max_depth={MAX_DEPTH}, "
        f"min_size={MIN_SIZE_FOR_SPLIT}, min_gain={MIN_GAIN}"
    )
    for ntree in NTREE_VALUES:
        print(f"  ntree={ntree:<3} ... ", end="", flush=True)
        cv = cross_validate_random_forest(attributes, classes, is_numerical, ntree)
        rows.append(
            {
                "setting": f"ntree={ntree}",
                "ntree": ntree,
                **cv,
            }
        )
        print(
            f"Acc={fmt(cv['accuracy_mean'], cv['accuracy_std'])}  "
            f"F1={fmt(cv['f1_mean'], cv['f1_std'])}"
        )
    return rows


def run_knn_sweep(attributes, classes, normalize):
    """Sweep k for k-NN, return per-setting CV results."""
    rows = []
    label = "normalized" if normalize else "raw"
    print(f"\n--- k-NN sweep ({label}) ---")
    for k in K_VALUES:
        print(f"  k={k:<3} ... ", end="", flush=True)
        cv = cross_validate_knn(attributes, classes, k, normalize=normalize)
        rows.append(
            {
                "setting": f"k={k} ({label})",
                "k": k,
                "normalize": normalize,
                **cv,
            }
        )
        print(
            f"Acc={fmt(cv['accuracy_mean'], cv['accuracy_std'])}  "
            f"F1={fmt(cv['f1_mean'], cv['f1_std'])}"
        )
    return rows


def best_by(rows, metric):
    return max(rows, key=lambda r: r[metric])


def write_summary(rf_rows, knn_norm_rows, knn_raw_rows):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write("=" * 90 + "\n")
        f.write(f"COMPARISON SUMMARY  -  {DATASET_LABEL}\n")
        f.write(f"Stratified {NUM_FOLDS}-fold CV; mean +/- std across folds\n")
        f.write(
            f"RF stopping criteria: max_depth={MAX_DEPTH}, "
            f"min_size={MIN_SIZE_FOR_SPLIT}, min_gain={MIN_GAIN}\n"
        )
        f.write("=" * 90 + "\n\n")

        # ── Per-setting tables ───────────────────────────────────────
        def write_table(title, rows):
            f.write(title + "\n")
            f.write("-" * 90 + "\n")
            f.write(
                f"{'Setting':<25} {'Accuracy':<22} {'Precision':<22} "
                f"{'Recall':<22} {'F1':<22}\n"
            )
            f.write("-" * 90 + "\n")
            for r in rows:
                f.write(
                    f"{r['setting']:<25} "
                    f"{fmt(r['accuracy_mean'], r['accuracy_std']):<22} "
                    f"{fmt(r['precision_mean'], r['precision_std']):<22} "
                    f"{fmt(r['recall_mean'], r['recall_std']):<22} "
                    f"{fmt(r['f1_mean'], r['f1_std']):<22}\n"
                )
            f.write("\n")

        write_table("Random Forest (varying ntree)", rf_rows)
        write_table("k-NN with min-max normalization (varying k)", knn_norm_rows)
        write_table("k-NN without normalization (varying k)", knn_raw_rows)

        # ── Headline comparison ──────────────────────────────────────
        rf_best_acc = best_by(rf_rows, "accuracy_mean")
        rf_best_f1 = best_by(rf_rows, "f1_mean")
        knn_best_acc = best_by(knn_norm_rows, "accuracy_mean")
        knn_best_f1 = best_by(knn_norm_rows, "f1_mean")
        knn_raw_best_acc = best_by(knn_raw_rows, "accuracy_mean")

        f.write("=" * 90 + "\n")
        f.write("HEADLINE COMPARISON (best hyperparameter for each algorithm)\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"{'Algorithm':<35} {'Best setting':<20} {'Accuracy':<22} {'F1':<22}\n")
        f.write("-" * 90 + "\n")
        f.write(
            f"{'Random Forest (best Acc)':<35} {rf_best_acc['setting']:<20} "
            f"{fmt(rf_best_acc['accuracy_mean'], rf_best_acc['accuracy_std']):<22} "
            f"{fmt(rf_best_acc['f1_mean'], rf_best_acc['f1_std']):<22}\n"
        )
        f.write(
            f"{'Random Forest (best F1)':<35} {rf_best_f1['setting']:<20} "
            f"{fmt(rf_best_f1['accuracy_mean'], rf_best_f1['accuracy_std']):<22} "
            f"{fmt(rf_best_f1['f1_mean'], rf_best_f1['f1_std']):<22}\n"
        )
        f.write(
            f"{'k-NN normalized (best Acc)':<35} {knn_best_acc['setting']:<20} "
            f"{fmt(knn_best_acc['accuracy_mean'], knn_best_acc['accuracy_std']):<22} "
            f"{fmt(knn_best_acc['f1_mean'], knn_best_acc['f1_std']):<22}\n"
        )
        f.write(
            f"{'k-NN normalized (best F1)':<35} {knn_best_f1['setting']:<20} "
            f"{fmt(knn_best_f1['accuracy_mean'], knn_best_f1['accuracy_std']):<22} "
            f"{fmt(knn_best_f1['f1_mean'], knn_best_f1['f1_std']):<22}\n"
        )
        f.write(
            f"{'k-NN raw (best Acc)':<35} {knn_raw_best_acc['setting']:<20} "
            f"{fmt(knn_raw_best_acc['accuracy_mean'], knn_raw_best_acc['accuracy_std']):<22} "
            f"{fmt(knn_raw_best_acc['f1_mean'], knn_raw_best_acc['f1_std']):<22}\n"
        )
        f.write("\n")

        # ── Normalization gap (for the writeup discussion) ───────────
        f.write("=" * 90 + "\n")
        f.write("NORMALIZATION EFFECT ON k-NN (accuracy, normalized minus raw)\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"{'k':<6} {'Normalized':<18} {'Raw':<18} {'Delta':<10}\n")
        f.write("-" * 60 + "\n")
        for norm, raw in zip(knn_norm_rows, knn_raw_rows):
            delta = norm["accuracy_mean"] - raw["accuracy_mean"]
            f.write(
                f"k={norm['k']:<4} "
                f"{norm['accuracy_mean']:.4f}             "
                f"{raw['accuracy_mean']:.4f}             "
                f"{delta:+.4f}\n"
            )
        f.write("\n")


def plot_overlay(rf_rows, knn_norm_rows, save_path):
    """Side-by-side accuracy plot with each algorithm on its own x-axis."""
    fig, (ax_rf, ax_knn) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    rf_x = [r["ntree"] for r in rf_rows]
    rf_mean = [r["accuracy_mean"] for r in rf_rows]
    rf_std = [r["accuracy_std"] for r in rf_rows]
    ax_rf.errorbar(
        rf_x,
        rf_mean,
        yerr=rf_std,
        fmt="-o",
        color="deepskyblue",
        ecolor="steelblue",
        capsize=3,
        markersize=7,
        label="Random Forest",
    )
    ax_rf.set_xlabel("Number of Trees (ntree)")
    ax_rf.set_ylabel("Accuracy")
    ax_rf.set_title("Random Forest")
    ax_rf.set_xticks(rf_x)
    ax_rf.grid(axis="y", alpha=0.3)

    knn_x = [r["k"] for r in knn_norm_rows]
    knn_mean = [r["accuracy_mean"] for r in knn_norm_rows]
    knn_std = [r["accuracy_std"] for r in knn_norm_rows]
    ax_knn.errorbar(
        knn_x,
        knn_mean,
        yerr=knn_std,
        fmt="-o",
        color="indianred",
        ecolor="firebrick",
        capsize=3,
        markersize=7,
        label="k-NN (normalized)",
    )
    ax_knn.set_xlabel("Number of Neighbors (k)")
    ax_knn.set_title("k-NN (min-max normalized)")
    ax_knn.set_xticks(knn_x)
    ax_knn.grid(axis="y", alpha=0.3)

    fig.suptitle(f"{DATASET_LABEL}: Random Forest vs. k-NN")
    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"Loading {DATASET_PATH}")
    attributes, classes, is_numerical = load_dataset(DATASET_PATH)
    print(
        f"  {attributes.shape[0]} samples, {attributes.shape[1]} features, "
        f"classes: {sorted(set(classes))}"
    )

    rf_rows = run_rf_sweep(attributes, classes, is_numerical)
    knn_norm_rows = run_knn_sweep(attributes, classes, normalize=True)
    knn_raw_rows = run_knn_sweep(attributes, classes, normalize=False)

    write_summary(rf_rows, knn_norm_rows, knn_raw_rows)
    plot_overlay(rf_rows, knn_norm_rows, COMPARISON_PLOT)

    print(f"\nWrote summary to {OUTPUT_PATH}")
    print(f"Wrote comparison plot to {COMPARISON_PLOT}")


if __name__ == "__main__":
    main()
