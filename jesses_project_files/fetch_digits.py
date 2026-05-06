"""One-shot exporter for the Hand-Written Digits dataset (Project §1.1).

Loads the dataset via ``sklearn.datasets.load_digits`` (the PDF guide
explicitly recommends this loader) and writes it to ``datasets/digits.csv``
in the same format as the other project datasets:

  * ``_num`` suffix on every numerical column so ``random_forest.load_dataset``
    treats it as numerical.
  * A trailing ``label`` column with the digit (as a string) so the existing
    multi-class voting / metric code in the project works unchanged.

Run once; downstream scripts (``compare_digits.py`` etc.) just read the CSV.
"""

import csv
import os

from sklearn import datasets

OUTPUT_PATH = "datasets/digits.csv"


def main():
    X, y = datasets.load_digits(return_X_y=True)
    n_samples, n_features = X.shape
    assert n_features == 64, f"expected 64 pixel features, got {n_features}"

    # 8x8 image -> name columns by their (row, col) position so the ordering
    # is recoverable for any pixel-level analysis later on.
    feature_headers = [f"pixel_{r}_{c}_num" for r in range(8) for c in range(8)]
    header = feature_headers + ["label"]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n_samples):
            row = [int(v) for v in X[i].tolist()] + [int(y[i])]
            writer.writerow(row)

    print(f"Wrote {n_samples} rows x {len(header)} cols to {OUTPUT_PATH}")
    print(f"Classes present: {sorted(set(int(v) for v in y))}")


if __name__ == "__main__":
    main()
