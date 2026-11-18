"""Evaluate a deployed checkpoint on a labeled dataset.

Usage:
  python check_deployed.py --data <dataset.csv> --ckpt <checkpoint_directory>

Reading the result:
  * accuracy on all comments close to 100% -> the checkpoint was trained on (almost) all of them
  * accuracy about 97-98%, with the errors concentrated -> it was trained on a subset and the errors are mostly held-out comments

The candidate-split search is only a heuristic. It assumes the same row order as the input dataset and should not be considered proof of the original train/test split.
"""
import argparse, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--tok", default="UBC-NLP/MARBERT", help="tokenizer of the ORIGINAL MARBERT (the checkpoint folder has no tokenizer files)")
ap.add_argument("--out", default="deployed_check.csv")
a = ap.parse_args()

df = pd.read_csv(a.data)
df = df.drop_duplicates("text").reset_index(drop=True)          # same preprocessing as cv_eval.py
df = df[df["label"].notna()].reset_index(drop=True)
y = df["label"].astype(float).astype(int).to_numpy(); X = df["text"].astype(str).tolist()
print("samples:", len(df), "label counts:", np.bincount(y))

tok = AutoTokenizer.from_pretrained(a.tok)
model = AutoModelForSequenceClassification.from_pretrained(a.ckpt).eval()
pred = []
with torch.no_grad():
    for i in range(0, len(X), 16):
        enc = tok(X[i:i + 16], truncation=True, max_length=128, padding=True, return_tensors="pt")
        pred += model(**enc).logits.argmax(1).tolist()
pred = np.array(pred); wrong = np.where(pred != y)[0]
print(f"accuracy on all {len(y)} comments: {np.mean(pred == y):.4f} | errors: {len(wrong)}")
pd.DataFrame({"idx": np.arange(len(y)), "y": y, "pred": pred, "text": X}).to_csv(a.out, index=False, encoding="utf-8-sig")
print("saved", a.out)

try:
    from sklearn.model_selection import train_test_split
    print("\nCandidate 80/20 splits (fraction of errors inside the test part; accuracy on the train part):")
    for rs in [42, 0, 1, 11, 22, 33, 2026]:
        for strat in (True, False):
            tr, te = train_test_split(np.arange(len(y)), test_size=0.2, random_state=rs, stratify=y if strat else None)
            inside = np.isin(wrong, te).mean() if len(wrong) else float("nan")
            print(f"  random_state={rs:5d} stratify={strat!s:5s}: train n={len(tr)} test n={len(te)} | errors in test: {inside:.2f} | train acc: {np.mean(pred[tr] == y[tr]):.4f}")
except Exception as e:
    print("\n(sklearn unavailable here, split search skipped:", type(e).__name__, ")")
