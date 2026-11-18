"""Experiment 1: 5-fold stratified CV x 3 seeds x 3 Arabic models on the Libyan dataset.
Usage: python cv_eval.py --data libyan.csv          (CSV columns: text,label with label in {0,1})
Output: cv_results/metrics.csv (one row per model/seed/fold) + cv_results/preds_<model>_s<seed>_f<fold>.csv
Resumable: finished runs are skipped. No test data is ever used for model selection (last epoch is used).
"""
import argparse, os, gc, numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, set_seed

MODELS = {  # replace with the exact checkpoints you used before
    "AraBERT":   "aubmindlab/bert-base-arabertv02",
    "MARBERT":   "UBC-NLP/MARBERTv2",
    "CAMeLBERT": "CAMeL-Lab/bert-base-arabic-camelbert-mix",
    "MARBERT_orig": "UBC-NLP/MARBERT",
}
SEEDS, FOLDS = [11, 22, 33], 5
LR, BS, EPOCHS, MAXLEN = 2e-5, 4, 8, 128   # = the earlier fine-tuning (checkpoint-680: 8 epochs x 85 steps, batch 4, lr 2e-5); keep IDENTICAL across models and folds

class DS(torch.utils.data.Dataset):
    def __init__(s, enc, y): s.enc, s.y = enc, y
    def __len__(s): return len(s.y)
    def __getitem__(s, i):
        d = {k: torch.tensor(v[i]) for k, v in s.enc.items()}; d["labels"] = torch.tensor(int(s.y[i])); return d

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default=None)
    ap.add_argument("--dialect", default=None, help="keep only rows where the dialect column equals this value")
    ap.add_argument("--smoke", action="store_true", help="quick test: 1 seed, 2 folds, 1 epoch")
    a = ap.parse_args()
    seeds, folds, epochs = ([11], 2, 1) if a.smoke else (SEEDS, FOLDS, EPOCHS)
    a.out = a.out or ("cv_smoke" if a.smoke else "cv_results"); os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.data)
    if a.dialect: df = df[df["dialect"] == a.dialect].reset_index(drop=True)
    df = df.drop_duplicates("text").reset_index(drop=True)
    n_all = len(df); df = df[df["label"].notna()].reset_index(drop=True)
    print(f"rows: {n_all}, with label: {len(df)}, dropped (empty label): {n_all - len(df)}")
    y = df["label"].astype(float).astype(int).to_numpy(); X = df["text"].astype(str).tolist()
    assert set(np.unique(y)) <= {0, 1}, f"labels must be 0/1, found {set(np.unique(y))}"
    print("samples:", len(df), "label counts:", np.bincount(y), "cuda:", torch.cuda.is_available())
    mpath = os.path.join(a.out, "metrics.csv")
    done = set(map(tuple, pd.read_csv(mpath)[["model", "seed", "fold"]].values)) if os.path.exists(mpath) else set()
    for seed in seeds:
        skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(skf.split(X, y)):
            for name, ckpt in MODELS.items():
                if (name, seed, fold) in done: continue
                set_seed(seed)
                tok = AutoTokenizer.from_pretrained(ckpt)
                enc = lambda idx: tok([X[i] for i in idx], truncation=True, max_length=MAXLEN, padding="max_length")
                model = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=2)
                args = TrainingArguments(output_dir="tmp_cv", per_device_train_batch_size=BS, per_device_eval_batch_size=64,
                                         num_train_epochs=epochs, learning_rate=LR, weight_decay=0.0, warmup_steps=0,
                                         save_strategy="no", report_to="none", seed=seed, fp16=torch.cuda.is_available(),
                                         logging_steps=1000)
                trn = Trainer(model=model, args=args, train_dataset=DS(enc(tr), y[tr]))
                trn.train()
                pred = trn.predict(DS(enc(te), y[te])).predictions.argmax(1)
                row = dict(model=name, seed=seed, fold=fold, n_test=len(te), acc=accuracy_score(y[te], pred),
                           f1_macro=f1_score(y[te], pred, average="macro"))
                pd.DataFrame({"idx": te, "y": y[te], "pred": pred}).to_csv(os.path.join(a.out, f"preds_{name}_s{seed}_f{fold}.csv"), index=False)
                pd.DataFrame([row]).to_csv(mpath, mode="a", header=not os.path.exists(mpath), index=False)
                print(row); del model, trn; gc.collect(); torch.cuda.empty_cache()
    m = pd.read_csv(mpath)
    print(m.groupby("model")[["acc", "f1_macro"]].agg(["mean", "std"]).round(4))

main()
