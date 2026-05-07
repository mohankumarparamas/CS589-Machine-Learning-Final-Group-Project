"""Regenerate the per-metric digits plots from the most recent CV results.

The full ``compare_digits.py`` sweep takes ~1.75h. After it ran the first
time we realized only the side-by-side comparison plot was being written;
the per-metric RF/k-NN plots that the rice flow produces were missing.
Rather than re-run the full sweep, this script feeds the already-measured
mean/std values (verbatim from ``figures/digits_run.log``) into the same
plotting helpers ``compare_digits.plot_per_metric`` calls. The resulting
figures are byte-equivalent to what a fresh sweep would now emit.

Going forward, ``compare_digits.py`` itself produces these plots, so this
file is a one-shot \u2014 keep it for provenance, but you should not need to
run it again.
"""

import compare_digits

# RF sweep \u2014 from figures/digits_run.log (Acc/Prec/Rec/F1, mean +/- std)
RF_ROWS = [
    {
        "setting": "ntree=1",
        "ntree": 1,
        "accuracy_mean": 0.7963,
        "accuracy_std": 0.0323,
        "precision_mean": 0.8027,
        "precision_std": 0.0319,
        "recall_mean": 0.7962,
        "recall_std": 0.0324,
        "f1_mean": 0.7953,
        "f1_std": 0.0326,
    },
    {
        "setting": "ntree=5",
        "ntree": 5,
        "accuracy_mean": 0.9232,
        "accuracy_std": 0.0173,
        "precision_mean": 0.9266,
        "precision_std": 0.0162,
        "recall_mean": 0.9231,
        "recall_std": 0.0176,
        "f1_mean": 0.9232,
        "f1_std": 0.0174,
    },
    {
        "setting": "ntree=10",
        "ntree": 10,
        "accuracy_mean": 0.9616,
        "accuracy_std": 0.0099,
        "precision_mean": 0.9636,
        "precision_std": 0.0092,
        "recall_mean": 0.9615,
        "recall_std": 0.0101,
        "f1_mean": 0.9616,
        "f1_std": 0.0099,
    },
    {
        "setting": "ntree=20",
        "ntree": 20,
        "accuracy_mean": 0.9655,
        "accuracy_std": 0.0063,
        "precision_mean": 0.9668,
        "precision_std": 0.0063,
        "recall_mean": 0.9654,
        "recall_std": 0.0064,
        "f1_mean": 0.9654,
        "f1_std": 0.0064,
    },
    {
        "setting": "ntree=30",
        "ntree": 30,
        "accuracy_mean": 0.9672,
        "accuracy_std": 0.0105,
        "precision_mean": 0.9688,
        "precision_std": 0.0097,
        "recall_mean": 0.9671,
        "recall_std": 0.0104,
        "f1_mean": 0.9672,
        "f1_std": 0.0103,
    },
    {
        "setting": "ntree=40",
        "ntree": 40,
        "accuracy_mean": 0.9688,
        "accuracy_std": 0.0090,
        "precision_mean": 0.9703,
        "precision_std": 0.0086,
        "recall_mean": 0.9687,
        "recall_std": 0.0090,
        "f1_mean": 0.9687,
        "f1_std": 0.0090,
    },
    {
        "setting": "ntree=50",
        "ntree": 50,
        "accuracy_mean": 0.9739,
        "accuracy_std": 0.0083,
        "precision_mean": 0.9748,
        "precision_std": 0.0080,
        "recall_mean": 0.9738,
        "recall_std": 0.0082,
        "f1_mean": 0.9738,
        "f1_std": 0.0082,
    },
]

# k-NN normalized sweep
KNN_NORM_ROWS = [
    {
        "setting": "k=1 (normalized)",
        "k": 1,
        "accuracy_mean": 0.9872,
        "accuracy_std": 0.0080,
        "precision_mean": 0.9880,
        "precision_std": 0.0073,
        "recall_mean": 0.9871,
        "recall_std": 0.0081,
        "f1_mean": 0.9871,
        "f1_std": 0.0080,
    },
    {
        "setting": "k=3 (normalized)",
        "k": 3,
        "accuracy_mean": 0.9883,
        "accuracy_std": 0.0102,
        "precision_mean": 0.9891,
        "precision_std": 0.0093,
        "recall_mean": 0.9883,
        "recall_std": 0.0103,
        "f1_mean": 0.9883,
        "f1_std": 0.0102,
    },
    {
        "setting": "k=5 (normalized)",
        "k": 5,
        "accuracy_mean": 0.9883,
        "accuracy_std": 0.0089,
        "precision_mean": 0.9891,
        "precision_std": 0.0080,
        "recall_mean": 0.9883,
        "recall_std": 0.0089,
        "f1_mean": 0.9882,
        "f1_std": 0.0089,
    },
    {
        "setting": "k=7 (normalized)",
        "k": 7,
        "accuracy_mean": 0.9849,
        "accuracy_std": 0.0094,
        "precision_mean": 0.9858,
        "precision_std": 0.0087,
        "recall_mean": 0.9849,
        "recall_std": 0.0095,
        "f1_mean": 0.9848,
        "f1_std": 0.0096,
    },
    {
        "setting": "k=9 (normalized)",
        "k": 9,
        "accuracy_mean": 0.9816,
        "accuracy_std": 0.0115,
        "precision_mean": 0.9829,
        "precision_std": 0.0102,
        "recall_mean": 0.9814,
        "recall_std": 0.0117,
        "f1_mean": 0.9814,
        "f1_std": 0.0117,
    },
    {
        "setting": "k=11 (normalized)",
        "k": 11,
        "accuracy_mean": 0.9822,
        "accuracy_std": 0.0106,
        "precision_mean": 0.9836,
        "precision_std": 0.0091,
        "recall_mean": 0.9820,
        "recall_std": 0.0108,
        "f1_mean": 0.9820,
        "f1_std": 0.0107,
    },
    {
        "setting": "k=15 (normalized)",
        "k": 15,
        "accuracy_mean": 0.9799,
        "accuracy_std": 0.0088,
        "precision_mean": 0.9813,
        "precision_std": 0.0077,
        "recall_mean": 0.9797,
        "recall_std": 0.0090,
        "f1_mean": 0.9797,
        "f1_std": 0.0089,
    },
    {
        "setting": "k=21 (normalized)",
        "k": 21,
        "accuracy_mean": 0.9755,
        "accuracy_std": 0.0107,
        "precision_mean": 0.9769,
        "precision_std": 0.0100,
        "recall_mean": 0.9753,
        "recall_std": 0.0108,
        "f1_mean": 0.9752,
        "f1_std": 0.0109,
    },
    {
        "setting": "k=31 (normalized)",
        "k": 31,
        "accuracy_mean": 0.9676,
        "accuracy_std": 0.0105,
        "precision_mean": 0.9695,
        "precision_std": 0.0101,
        "recall_mean": 0.9674,
        "recall_std": 0.0106,
        "f1_mean": 0.9674,
        "f1_std": 0.0107,
    },
    {
        "setting": "k=51 (normalized)",
        "k": 51,
        "accuracy_mean": 0.9560,
        "accuracy_std": 0.0137,
        "precision_mean": 0.9580,
        "precision_std": 0.0137,
        "recall_mean": 0.9559,
        "recall_std": 0.0136,
        "f1_mean": 0.9557,
        "f1_std": 0.0139,
    },
]

# k-NN raw sweep (only accuracy mean/std are needed for the comparison plot,
# but we keep the full row shape for symmetry).
KNN_RAW_ROWS = [
    {
        "setting": "k=1 (raw)",
        "k": 1,
        "accuracy_mean": 0.9883,
        "accuracy_std": 0.0078,
        "precision_mean": 0.9890,
        "precision_std": 0.0072,
        "recall_mean": 0.9882,
        "recall_std": 0.0079,
        "f1_mean": 0.9882,
        "f1_std": 0.0078,
    },
    {
        "setting": "k=3 (raw)",
        "k": 3,
        "accuracy_mean": 0.9888,
        "accuracy_std": 0.0094,
        "precision_mean": 0.9896,
        "precision_std": 0.0085,
        "recall_mean": 0.9888,
        "recall_std": 0.0095,
        "f1_mean": 0.9888,
        "f1_std": 0.0094,
    },
    {
        "setting": "k=5 (raw)",
        "k": 5,
        "accuracy_mean": 0.9877,
        "accuracy_std": 0.0093,
        "precision_mean": 0.9886,
        "precision_std": 0.0084,
        "recall_mean": 0.9877,
        "recall_std": 0.0094,
        "f1_mean": 0.9877,
        "f1_std": 0.0094,
    },
    {
        "setting": "k=7 (raw)",
        "k": 7,
        "accuracy_mean": 0.9855,
        "accuracy_std": 0.0084,
        "precision_mean": 0.9863,
        "precision_std": 0.0079,
        "recall_mean": 0.9855,
        "recall_std": 0.0085,
        "f1_mean": 0.9854,
        "f1_std": 0.0086,
    },
    {
        "setting": "k=9 (raw)",
        "k": 9,
        "accuracy_mean": 0.9833,
        "accuracy_std": 0.0094,
        "precision_mean": 0.9844,
        "precision_std": 0.0086,
        "recall_mean": 0.9831,
        "recall_std": 0.0095,
        "f1_mean": 0.9830,
        "f1_std": 0.0095,
    },
    {
        "setting": "k=11 (raw)",
        "k": 11,
        "accuracy_mean": 0.9839,
        "accuracy_std": 0.0081,
        "precision_mean": 0.9850,
        "precision_std": 0.0072,
        "recall_mean": 0.9837,
        "recall_std": 0.0083,
        "f1_mean": 0.9837,
        "f1_std": 0.0082,
    },
    {
        "setting": "k=15 (raw)",
        "k": 15,
        "accuracy_mean": 0.9805,
        "accuracy_std": 0.0072,
        "precision_mean": 0.9819,
        "precision_std": 0.0060,
        "recall_mean": 0.9803,
        "recall_std": 0.0074,
        "f1_mean": 0.9803,
        "f1_std": 0.0073,
    },
    {
        "setting": "k=21 (raw)",
        "k": 21,
        "accuracy_mean": 0.9760,
        "accuracy_std": 0.0106,
        "precision_mean": 0.9775,
        "precision_std": 0.0096,
        "recall_mean": 0.9758,
        "recall_std": 0.0107,
        "f1_mean": 0.9758,
        "f1_std": 0.0108,
    },
    {
        "setting": "k=31 (raw)",
        "k": 31,
        "accuracy_mean": 0.9671,
        "accuracy_std": 0.0110,
        "precision_mean": 0.9690,
        "precision_std": 0.0106,
        "recall_mean": 0.9669,
        "recall_std": 0.0111,
        "f1_mean": 0.9668,
        "f1_std": 0.0112,
    },
    {
        "setting": "k=51 (raw)",
        "k": 51,
        "accuracy_mean": 0.9565,
        "accuracy_std": 0.0129,
        "precision_mean": 0.9586,
        "precision_std": 0.0127,
        "recall_mean": 0.9564,
        "recall_std": 0.0129,
        "f1_mean": 0.9562,
        "f1_std": 0.0131,
    },
]


def main():
    # 9 per-metric plots (rf/knn x acc/prec/rec/f1, plus the
    # normalized-vs-raw overlay).
    compare_digits.plot_per_metric(RF_ROWS, KNN_NORM_ROWS, KNN_RAW_ROWS)

    # Side-by-side overlay (titles changed when we relabeled the dataset).
    compare_digits.plot_overlay(RF_ROWS, KNN_NORM_ROWS, compare_digits.COMPARISON_PLOT)

    # Summary text file (also embeds the dataset label).
    compare_digits.write_summary(RF_ROWS, KNN_NORM_ROWS, KNN_RAW_ROWS)

    print(
        f"Wrote per-metric plots to figures/rf_{compare_digits.DATASET_TAG}_*.pdf "
        f"and figures/knn_{compare_digits.DATASET_TAG}_*.pdf"
    )
    print(f"Wrote comparison plot to {compare_digits.COMPARISON_PLOT}")
    print(f"Wrote summary to {compare_digits.OUTPUT_PATH}")


if __name__ == "__main__":
    main()
