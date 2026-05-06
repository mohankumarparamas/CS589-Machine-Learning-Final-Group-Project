from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from random_forest import (
    NUM_FOLDS,
    compute_accuracy,
    compute_precision_recall_f1,
    load_dataset,
    stratified_k_fold_indices,
)

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
K_VALUES = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
SEED = 42


# ──────────────────────────────────────────────────────────────────────
# k-NN core
# ──────────────────────────────────────────────────────────────────────
def normalize_attributes(train_attrs, test_attrs):
    """Min-max normalization fit on training data, applied to both splits."""
    train_attrs = np.asarray(train_attrs, dtype=float)
    test_attrs = np.asarray(test_attrs, dtype=float)
    training_minimums = train_attrs.min(axis=0)
    training_maximums = train_attrs.max(axis=0)
    ranges = training_maximums - training_minimums
    ranges[ranges == 0] = 1.0
    return (
        (train_attrs - training_minimums) / ranges,
        (test_attrs - training_minimums) / ranges,
    )


def knn_predictions(train_attrs, train_classes, query_points, k):
    train_attrs = np.asarray(train_attrs, dtype=float)
    query_points = np.asarray(query_points, dtype=float)

    predictions = np.empty(query_points.shape[0], dtype=object)
    for i in range(query_points.shape[0]):
        euclidean_dists = np.linalg.norm(train_attrs - query_points[i], axis=1)
        neighbor_indices = np.argsort(euclidean_dists)[:k]
        neighbor_labels = train_classes[neighbor_indices]
        predictions[i] = Counter(neighbor_labels).most_common(1)[0][0]
    return predictions


# ──────────────────────────────────────────────────────────────────────
# Stratified k-fold CV for k-NN
# ──────────────────────────────────────────────────────────────────────
def cross_validate_knn(
    attributes, classes, k, num_folds=NUM_FOLDS, normalize=True, seed=SEED
):
    rng = np.random.default_rng(seed)
    folds = stratified_k_fold_indices(classes, num_folds, rng)

    accuracies, precisions, recalls, f1_scores = [], [], [], []

    # k-NN only operates on numerical features. Coerce up-front so we don't
    # repeat the cast each fold; rice.csv is fully numerical.
    attributes = np.asarray(attributes, dtype=float)

    for fold_i in range(num_folds):
        test_idx = folds[fold_i]
        train_idx = np.concatenate([folds[j] for j in range(num_folds) if j != fold_i])

        train_attrs = attributes[train_idx]
        train_classes = classes[train_idx]
        test_attrs = attributes[test_idx]
        test_classes = classes[test_idx]

        if normalize:
            train_attrs, test_attrs = normalize_attributes(train_attrs, test_attrs)

        predictions = knn_predictions(train_attrs, train_classes, test_attrs, k)

        accuracies.append(compute_accuracy(test_classes, predictions))
        p, r, f = compute_precision_recall_f1(test_classes, predictions)
        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f)

    return {
        "accuracy_mean": float(np.mean(accuracies)),
        "accuracy_std": float(np.std(accuracies)),
        "precision_mean": float(np.mean(precisions)),
        "precision_std": float(np.std(precisions)),
        "recall_mean": float(np.mean(recalls)),
        "recall_std": float(np.std(recalls)),
        "f1_mean": float(np.mean(f1_scores)),
        "f1_std": float(np.std(f1_scores)),
        "accuracies": accuracies,
        "precisions": precisions,
        "recalls": recalls,
        "f1_scores": f1_scores,
    }


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────
def plot_metric_vs_k(
    k_values,
    metric_values,
    metric_name,
    dataset_name,
    save_path,
    metric_stds=None,
    color="deepskyblue",
):
    plt.figure(figsize=(8, 5))
    if metric_stds is not None:
        plt.errorbar(
            k_values,
            metric_values,
            yerr=metric_stds,
            fmt="-o",
            linewidth=2,
            color=color,
            ecolor="steelblue",
            capsize=3,
            markersize=7,
        )
    else:
        plt.plot(
            k_values,
            metric_values,
            marker="o",
            linewidth=2,
            color=color,
            markersize=7,
        )
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel(metric_name)
    plt.title(f"{dataset_name}: {metric_name} vs. k")
    plt.xticks(k_values)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()


def plot_normalized_vs_raw(
    k_values, norm_means, norm_stds, raw_means, raw_stds, dataset_name, save_path
):
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        k_values,
        norm_means,
        yerr=norm_stds,
        fmt="-o",
        label="Min-max normalized",
        color="deepskyblue",
        ecolor="steelblue",
        capsize=3,
        markersize=6,
    )
    plt.errorbar(
        k_values,
        raw_means,
        yerr=raw_stds,
        fmt="-s",
        label="Unnormalized",
        color="indianred",
        ecolor="firebrick",
        capsize=3,
        markersize=6,
    )
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Accuracy")
    plt.title(f"{dataset_name}: k-NN Accuracy with vs. without Normalization")
    plt.xticks(k_values)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Experiment driver
# ──────────────────────────────────────────────────────────────────────
def run_knn_experiment(dataset_filepath, dataset_label, k_values=K_VALUES):
    attributes, classes, _ = load_dataset(dataset_filepath)

    norm_results = {
        key: []
        for key in (
            "accuracy",
            "accuracy_std",
            "precision",
            "precision_std",
            "recall",
            "recall_std",
            "f1",
            "f1_std",
        )
    }
    raw_results = {key: [] for key in ("accuracy", "accuracy_std")}

    print(f"\nDataset: {dataset_label}  ({dataset_filepath})")
    print(f"Stratified {NUM_FOLDS}-fold CV, sweeping k = {k_values}")
    print("\n--- With Min-Max Normalization ---")
    for k in k_values:
        cv = cross_validate_knn(attributes, classes, k, normalize=True)
        norm_results["accuracy"].append(cv["accuracy_mean"])
        norm_results["accuracy_std"].append(cv["accuracy_std"])
        norm_results["precision"].append(cv["precision_mean"])
        norm_results["precision_std"].append(cv["precision_std"])
        norm_results["recall"].append(cv["recall_mean"])
        norm_results["recall_std"].append(cv["recall_std"])
        norm_results["f1"].append(cv["f1_mean"])
        norm_results["f1_std"].append(cv["f1_std"])
        print(
            f"  k={k:3d}  Acc={cv['accuracy_mean']:.4f}+/-{cv['accuracy_std']:.4f}  "
            f"Prec={cv['precision_mean']:.4f}+/-{cv['precision_std']:.4f}  "
            f"Rec={cv['recall_mean']:.4f}+/-{cv['recall_std']:.4f}  "
            f"F1={cv['f1_mean']:.4f}+/-{cv['f1_std']:.4f}"
        )

    print("\n--- Without Normalization (raw features) ---")
    for k in k_values:
        cv = cross_validate_knn(attributes, classes, k, normalize=False)
        raw_results["accuracy"].append(cv["accuracy_mean"])
        raw_results["accuracy_std"].append(cv["accuracy_std"])
        print(
            f"  k={k:3d}  Acc={cv['accuracy_mean']:.4f}+/-{cv['accuracy_std']:.4f}  "
            f"F1={cv['f1_mean']:.4f}+/-{cv['f1_std']:.4f}"
        )

    tag = dataset_label.lower().replace(" ", "_").replace("#", "")
    metric_names = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1 Score",
    }
    for key, display in metric_names.items():
        plot_metric_vs_k(
            k_values,
            norm_results[key],
            display,
            dataset_label,
            f"figures/knn_{tag}_{key}.pdf",
            metric_stds=norm_results[f"{key}_std"],
        )

    plot_normalized_vs_raw(
        k_values,
        norm_results["accuracy"],
        norm_results["accuracy_std"],
        raw_results["accuracy"],
        raw_results["accuracy_std"],
        dataset_label,
        f"figures/knn_{tag}_normalization_comparison.pdf",
    )

    return norm_results, raw_results


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("\n--- Starting k-NN Experiment ---")
    run_knn_experiment("datasets/rice.csv", "Dataset #1 (Rice)")
    print("\n--- k-NN Experiment Complete ---")


if __name__ == "__main__":
    main()
