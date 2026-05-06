from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# ──────────────────────────────────────────────────────────────────────
# Configuration Vars
# ──────────────────────────────────────────────────────────────────────
# Single Run
NUM_RUNS = 50
TEST_SIZE = 0.2

# Multi Run
NUM_FOLDS = 10
NTREE_VALUES = [1, 5, 10, 20, 30, 40, 50]

# Stopping criteria
MAX_DEPTH = 5
MIN_SIZE_FOR_SPLIT = 2
MIN_GAIN = 0


# ──────────────────────────────────────────────────────────────────────
# Dataset loading
# ──────────────────────────────────────────────────────────────────────
def load_dataset(filepath):
    with open(filepath) as f:
        lines = [line.strip().split(",") for line in f if line.strip()]

    header = lines[0]
    data = np.array(lines[1:], dtype=object)

    label_col = None
    for i, col_name in enumerate(header):
        if col_name.lower() == "label":
            label_col = i
            break
    if label_col is None:
        label_col = len(header) - 1

    feature_indices = [i for i in range(len(header)) if i != label_col]
    feature_headers = [header[i] for i in feature_indices]

    attributes = data[:, feature_indices]
    classes = data[:, label_col]

    is_numerical = []
    for h in feature_headers:
        h_lower = h.lower()
        if h_lower.endswith("_num"):
            is_numerical.append(True)
        elif h_lower.endswith("_cat"):
            is_numerical.append(False)
        else:
            is_numerical.append(True)

    for j in range(attributes.shape[1]):
        if is_numerical[j]:
            attributes[:, j] = attributes[:, j].astype(float)

    return attributes, classes, is_numerical


def partition_dataset(attributes, classes):
    attributes_shuf, classes_shuf = shuffle(attributes, classes)
    attr_train, attr_test, cls_train, cls_test = train_test_split(
        attributes_shuf, classes_shuf, test_size=TEST_SIZE
    )
    training_set = {"attributes": attr_train, "classes": cls_train}
    testing_set = {"attributes": attr_test, "classes": cls_test}

    return training_set, testing_set


# ──────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────
def compute_accuracy(actual, predicted):
    return np.mean(actual == predicted)


def compute_precision_recall_f1(actual, predicted):
    labels = np.unique(actual)

    precisions, recalls, f1s = [], [], []
    for lbl in labels:
        tp = np.sum((predicted == lbl) & (actual == lbl))
        fp = np.sum((predicted == lbl) & (actual != lbl))
        fn = np.sum((predicted != lbl) & (actual == lbl))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    mean_f1 = np.mean(f1s)

    return mean_precision, mean_recall, mean_f1


# ──────────────────────────────────────────────────────────────────────
# Entropy & Information Gain helpers
# ──────────────────────────────────────────────────────────────────────
def entropy(classes):
    if len(classes) == 0:
        return 0.0

    _, counts = np.unique(classes, return_counts=True)
    probs = counts / len(classes)

    return -np.sum(probs * np.log2(probs))


def information_gain(classes, attribute_values):
    parent_entropy = entropy(classes)
    unique_vals = np.unique(attribute_values)
    weighted_entropy = 0.0

    for val in unique_vals:
        mask = attribute_values == val
        weighted_entropy += (mask.sum() / len(classes)) * entropy(classes[mask])

    return parent_entropy - weighted_entropy


def information_gain_numerical(classes, values):
    float_vals = values.astype(float)
    parent_ent = entropy(classes)

    order = np.argsort(float_vals)
    sorted_vals = float_vals[order]
    sorted_cls = classes[order]

    # Candidate thresholds:
    candidates = []
    for i in range(len(sorted_cls) - 1):
        if sorted_cls[i] != sorted_cls[i + 1]:
            candidates.append((sorted_vals[i] + sorted_vals[i + 1]) / 2.0)

    if len(candidates) == 0:
        candidates = [np.mean(float_vals)]

    best_gain = -1.0
    best_thresh = candidates[0]
    n = len(classes)
    for t in candidates:
        left_mask = float_vals <= t
        right_mask = ~left_mask
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            continue
        w_ent = (left_mask.sum() / n) * entropy(classes[left_mask]) + (
            right_mask.sum() / n
        ) * entropy(classes[right_mask])
        gain = parent_ent - w_ent
        if gain > best_gain:
            best_gain = gain
            best_thresh = t

    return best_gain, best_thresh


# ──────────────────────────────────────────────────────────────────────
# Decision Tree
# ──────────────────────────────────────────────────────────────────────
def best_split(attributes, classes, candidates, is_numerical):
    best_gain = -1
    best_index = candidates[0]
    best_threshold = None

    for i in candidates:
        if is_numerical[i]:
            gain, threshold = information_gain_numerical(classes, attributes[:, i])
        else:
            gain = information_gain(classes, attributes[:, i])
            threshold = None
        if gain > best_gain:
            best_gain = gain
            best_index = i
            best_threshold = threshold

    return best_index, best_threshold, best_gain


def build_tree(
    attributes, classes, available_indices, is_numerical, depth=0, m=None, rng=None
):
    max_depth = MAX_DEPTH
    min_size = MIN_SIZE_FOR_SPLIT
    min_gain = MIN_GAIN

    # Consolidate Setup
    unique_classes, counts = np.unique(classes, return_counts=True)
    majority_class = unique_classes[np.argmax(counts)]

    # Pure node
    if len(unique_classes) == 1:
        return {"leaf": True, "prediction": unique_classes[0]}

    # No attributes left — majority vote
    if len(available_indices) == 0:
        return {"leaf": True, "prediction": majority_class}

    # Max depth reached
    if depth >= max_depth:
        return {"leaf": True, "prediction": majority_class}

    # Too few samples to split
    if len(classes) < min_size:
        return {"leaf": True, "prediction": majority_class}

    # Random Forest: pick m random candidate features; otherwise use all
    if m is not None and rng is not None:
        size = min(m, len(available_indices))
        candidates = sorted(
            rng.choice(available_indices, size=size, replace=False).tolist()
        )
    else:
        candidates = available_indices

    # Split on best attribute
    split_idx, threshold, gain = best_split(
        attributes, classes, candidates, is_numerical
    )
    # Information gain too small
    if gain < min_gain:
        return {"leaf": True, "prediction": majority_class}

    # Build children
    if is_numerical[split_idx]:
        float_vals = attributes[:, split_idx].astype(float)
        left_mask = float_vals <= threshold
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return {"leaf": True, "prediction": majority_class}

        # For numerical re-split on the same attr with a different threshold
        left_child = build_tree(
            attributes[left_mask],
            classes[left_mask],
            available_indices,
            is_numerical,
            depth + 1,
            m,
            rng,
        )
        right_child = build_tree(
            attributes[right_mask],
            classes[right_mask],
            available_indices,
            is_numerical,
            depth + 1,
            m,
            rng,
        )

        return {
            "leaf": False,
            "attribute_index": split_idx,
            "numerical": True,
            "threshold": threshold,
            "left": left_child,
            "right": right_child,
            "default": majority_class,
        }
    else:
        remaining = [i for i in available_indices if i != split_idx]
        children = {}
        for val in np.unique(attributes[:, split_idx]):
            mask = attributes[:, split_idx] == val
            children[val] = build_tree(
                attributes[mask],
                classes[mask],
                remaining,
                is_numerical,
                depth + 1,
                m,
                rng,
            )

        return {
            "leaf": False,
            "attribute_index": split_idx,
            "numerical": False,
            "children": children,
            "default": majority_class,
        }


def predict_one(tree, instance):
    if tree["leaf"]:
        return tree["prediction"]

    idx = tree["attribute_index"]

    if tree["numerical"]:
        val = float(instance[idx])
        if val <= tree["threshold"]:
            return predict_one(tree["left"], instance)
        else:
            return predict_one(tree["right"], instance)
    else:
        val = instance[idx]
        if val in tree["children"]:
            return predict_one(tree["children"][val], instance)
        return tree["default"]


def predict(tree, attributes):
    pred = np.array(
        [predict_one(tree, attributes[i]) for i in range(len(attributes))],
        dtype=object,
    )

    return pred


def predict_forest(forest, attributes):
    all_preds = np.array([predict(tree, attributes) for tree in forest])
    n_samples = attributes.shape[0]
    final = np.empty(n_samples, dtype=object)

    for i in range(n_samples):
        votes = all_preds[:, i]
        counter = Counter(votes)
        final[i] = counter.most_common(1)[0][0]

    return final


# ──────────────────────────────────────────────────────────────────────
# Random Forest helpers
# ──────────────────────────────────────────────────────────────────────
def bootstrap_sample(attributes, classes, rng):
    num = len(classes)
    indices = rng.choice(num, size=num, replace=True)

    return attributes[indices], classes[indices]


def build_rf_tree(attributes, classes, is_numerical, rng):
    n_features = attributes.shape[1]
    m = max(1, int(np.floor(np.sqrt(n_features))))
    available_indices = list(range(n_features))

    return build_tree(attributes, classes, available_indices, is_numerical, 0, m, rng)


def build_random_forest(attributes, classes, is_numerical, rng, ntree):
    forest = []

    for _ in range(ntree):
        bootstrap_attributes, bootstrap_classes = bootstrap_sample(
            attributes, classes, rng
        )
        tree = build_rf_tree(bootstrap_attributes, bootstrap_classes, is_numerical, rng)
        forest.append(tree)

    return forest


# ──────────────────────────────────────────────────────────────────────
# Stratified k-fold cross-validation
# ──────────────────────────────────────────────────────────────────────
def stratified_k_fold_indices(classes, k, rng):
    unique_labels = np.unique(classes)

    # Group indices by class
    class_indices = {}
    for lbl in unique_labels:
        idx = np.where(classes == lbl)[0]
        rng.shuffle(idx)
        class_indices[lbl] = idx

    # Initialise empty folds
    folds = [[] for _ in range(k)]

    # Distribute each class's indices round-robin across folds
    for lbl in unique_labels:
        idxs = class_indices[lbl]
        splits = np.array_split(idxs, k)
        for fold_i in range(k):
            folds[fold_i].extend(splits[fold_i].tolist())

    # Shuffle within each fold
    for fold_i in range(k):
        arr = np.array(folds[fold_i])
        rng.shuffle(arr)
        folds[fold_i] = arr

    return folds


def cross_validate_random_forest(
    attributes, classes, is_numerical, ntree, k=NUM_FOLDS, seed=42
):
    rng = np.random.default_rng(seed)
    folds = stratified_k_fold_indices(classes, k, rng)
    accuracies, precisions, recalls, f1_scores = [], [], [], []

    for fold_i in range(k):
        test_idx = folds[fold_i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != fold_i])

        train_attribute = attributes[train_idx]
        train_class = classes[train_idx]
        test_attribute = attributes[test_idx]
        test_class = classes[test_idx]

        forest = build_random_forest(
            train_attribute, train_class, is_numerical, rng, ntree
        )
        predictions = predict_forest(forest, test_attribute)

        accuracies.append(compute_accuracy(test_class, predictions))
        p, r, f = compute_precision_recall_f1(test_class, predictions)
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
def plot_histogram(accuracies, title, ylabel, save_path):
    plt.figure(figsize=(8, 5))
    plt.hist(accuracies, bins=30, color="deepskyblue", edgecolor="deepskyblue")
    plt.xlabel("(Accuracy)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()


def plot_metric_vs_ntree(
    ntree_values,
    metric_values,
    metric_name,
    dataset_name,
    save_path,
    metric_stds=None,
):
    plt.figure(figsize=(8, 5))
    if metric_stds is not None:
        plt.errorbar(
            ntree_values,
            metric_values,
            yerr=metric_stds,
            fmt="-o",
            linewidth=2,
            color="deepskyblue",
            ecolor="steelblue",
            markersize=7,
            capsize=3,
        )
    else:
        plt.plot(
            ntree_values,
            metric_values,
            marker="o",
            linewidth=2,
            color="deepskyblue",
            markersize=7,
        )
    plt.xlabel("Number of Trees (ntree)")
    plt.ylabel(metric_name)
    plt.title(f"{dataset_name}: {metric_name} vs. Number of Trees")
    plt.xticks(ntree_values)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()


# ──────────────────────────────────────────────────────────────────────
# Original HW1 single decision tree experiment
# ──────────────────────────────────────────────────────────────────────
def run_single_tree_experiment(dataset_filepath):
    attributes, classes, is_numerical = load_dataset(dataset_filepath)
    train_accuracies = np.zeros(NUM_RUNS)
    test_accuracies = np.zeros(NUM_RUNS)

    for run in range(NUM_RUNS):
        training_set, testing_set = partition_dataset(attributes, classes)

        tree = build_tree(
            training_set["attributes"],
            training_set["classes"],
            list(range(training_set["attributes"].shape[1])),
            is_numerical,
        )

        train_predicted = predict(tree, training_set["attributes"])
        train_accuracies[run] = compute_accuracy(
            training_set["classes"], train_predicted
        )

        test_predicted = predict(tree, testing_set["attributes"])
        test_accuracies[run] = compute_accuracy(testing_set["classes"], test_predicted)

        print(f"Single tree run {run + 1}/{NUM_RUNS} complete")

    # Compute Statistics
    train_mean = train_accuracies.mean()
    train_std = train_accuracies.std()
    test_mean = test_accuracies.mean()
    test_std = test_accuracies.std()

    # Sanity Checks
    print(f"\n--- Decision Tree Results ({NUM_RUNS} runs) ---")
    print(f"Training accuracy: {train_mean:.4f} +/- {train_std:.4f}")
    print(f"Testing accuracy:  {test_mean:.4f} +/- {test_std:.4f}")

    # Plot Outputs
    plot_histogram(
        train_accuracies,
        "Decision Tree: Training Accuracy Distribution",
        "(Accuracy Frequency\non Training Data)",
        "figures/q2_1_training_accuracy.pdf",
    )

    plot_histogram(
        test_accuracies,
        "Decision Tree: Testing Accuracy Distribution",
        "(Accuracy Frequency\non Testing Data)",
        "figures/q2_2_testing_accuracy.pdf",
    )


# ──────────────────────────────────────────────────────────────────────
# Random Forest experiment
# ──────────────────────────────────────────────────────────────────────
def run_rf_experiment(dataset_filepath, dataset_label):
    ntree_values = NTREE_VALUES
    attributes, classes, is_numerical = load_dataset(dataset_filepath)
    results = {
        "accuracy": [],
        "accuracy_std": [],
        "precision": [],
        "precision_std": [],
        "recall": [],
        "recall_std": [],
        "f1": [],
        "f1_std": [],
    }

    print(f"\nDataset: {dataset_label}  ({dataset_filepath})")
    for ntree in ntree_values:
        print(f"  ntree = {ntree} ... ", end="", flush=True)
        cv = cross_validate_random_forest(attributes, classes, is_numerical, ntree)
        results["accuracy"].append(cv["accuracy_mean"])
        results["accuracy_std"].append(cv["accuracy_std"])
        results["precision"].append(cv["precision_mean"])
        results["precision_std"].append(cv["precision_std"])
        results["recall"].append(cv["recall_mean"])
        results["recall_std"].append(cv["recall_std"])
        results["f1"].append(cv["f1_mean"])
        results["f1_std"].append(cv["f1_std"])
        print(
            f"Acc={cv['accuracy_mean']:.4f}+/-{cv['accuracy_std']:.4f}  "
            f"Prec={cv['precision_mean']:.4f}+/-{cv['precision_std']:.4f}  "
            f"Rec={cv['recall_mean']:.4f}+/-{cv['recall_std']:.4f}  "
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
        save_path = f"figures/rf_{tag}_{key}.pdf"
        plot_metric_vs_ntree(
            ntree_values,
            results[key],
            display,
            dataset_label,
            save_path,
            metric_stds=results[f"{key}_std"],
        )

    return results


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    print("\n--- Start Experiments ---")

    # Original HW1 left in for reference, though some functions were
    # reused and expanded
    print("\n--- Decision Tree Test ---")

    # run_single_tree_experiment("datasets/wdbc.csv") / Skip for Random forest

    # Random Forest function is abstracted so it can be passed just a
    # a dataset link and label for the graphs
    print("\n--- Random Forest Tests ---")
    print(f"\nMax Depth: {MAX_DEPTH}")
    print(f"Min Split: {MIN_SIZE_FOR_SPLIT}")
    print(f"Min Gain:  {MIN_GAIN}")

    run_rf_experiment("datasets/rice.csv", "Dataset #1 (Rice)")

    print("\n--- Experiment Complete ---")


if __name__ == "__main__":
    main()
