Requirements: Python 3, numpy, matplotlib, scikit-learn

Setup:
  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

Run (Dataset 1.3 — Rice):
  python3 random_forest.py
  python3 k_nn.py
  python3 compare.py

Run (Dataset 1.1 — Hand-Written Digits):
  python3 fetch_digits.py        # one-shot: writes datasets/digits.csv
  python3 compare_digits.py      # ~1.75h: RF + k-NN sweep with stratified 10-fold CV

Output figures and summaries are saved to the figures/ directory.
