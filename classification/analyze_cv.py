"""Statistical analysis of cv_results/ (output of cv_eval.py).  Usage: python analyze_cv.py [cv_results_dir]
Reports: per-seed pooled accuracy/F1 (n = all labeled samples), mean+-sd over seeds, bootstrap 95% CI,
paired bootstrap differences between models, exact McNemar per seed (raw and Holm-adjusted), majority-class baseline.
"""
import sys, glob, os, re, math, numpy as np, pandas as pd

def f1_score(y_true, y_pred, average="macro"):
    """Macro-F1 for labels {0,1} without sklearn (avoids the blocked sklearn DLL)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    f1s = []
    for c in np.unique(np.concatenate([y_true, y_pred])):
        tp = np.sum((y_pred == c) & (y_true == c)); fp = np.sum((y_pred == c) & (y_true != c)); fn = np.sum((y_pred != c) & (y_true == c))
        f1s.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))
d = sys.argv[1] if len(sys.argv) > 1 else "cv_results"
files = glob.glob(os.path.join(d, "preds_*_s*_f*.csv"))
rows = []
for f in files:
    m = re.match(r"preds_(.+)_s(\d+)_f(\d+)\.csv", os.path.basename(f))
    p = pd.read_csv(f); p["model"], p["seed"], p["fold"] = m.group(1), int(m.group(2)), int(m.group(3)); rows.append(p)
P = pd.concat(rows, ignore_index=True)
models, seeds = sorted(P.model.unique()), sorted(P.seed.unique())
n = P.idx.nunique(); print(f"files: {len(files)} | samples: {n} | models: {models} | seeds: {seeds}")
cnt = P.groupby(["model", "seed"]).size(); assert (cnt == n).all(), f"each model/seed must cover all {n} samples once:\n{cnt}"
y = P.drop_duplicates("idx").set_index("idx").sort_index()["y"].to_numpy()
print(f"majority-class baseline: {max(np.mean(y == 0), np.mean(y == 1)):.4f}")
corr, per = {}, []
for mo in models:
    for s in seeds:
        q = P[(P.model == mo) & (P.seed == s)].set_index("idx").sort_index()
        corr[(mo, s)] = (q["pred"].to_numpy() == y).astype(float)
        per.append(dict(model=mo, seed=s, acc=corr[(mo, s)].mean(), f1_macro=f1_score(y, q["pred"].to_numpy(), average="macro")))
per = pd.DataFrame(per)
print("\nPer-seed pooled results (each sample tested once per seed):"); print(per.round(4).to_string(index=False))
print("\nMean +- sd over seeds:")
print(per.groupby("model")[["acc", "f1_macro"]].agg(["mean", "std"]).round(4))
S = {mo: np.mean([corr[(mo, s)] for s in seeds], axis=0) for mo in models}      # per-sample correctness averaged over seeds
rng = np.random.default_rng(2026); B = 10000; idx = rng.integers(0, n, (B, n))
print("\nBootstrap 95% CI of accuracy (resampling the labeled samples):")
for mo in models:
    b = S[mo][idx].mean(1); print(f"  {mo:14s} {S[mo].mean():.4f}  [{np.percentile(b, 2.5):.4f}, {np.percentile(b, 97.5):.4f}]")
print("\nPaired bootstrap difference in accuracy (A - B), 95% CI:")
pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1:]]
for a, b in pairs:
    dd = (S[a] - S[b])[idx].mean(1); lo, hi = np.percentile(dd, [2.5, 97.5])
    print(f"  {a} - {b}: {S[a].mean() - S[b].mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  {'(CI excludes 0)' if lo > 0 or hi < 0 else '(CI includes 0 -> not distinguishable)'}")

def mcnemar_exact(b, c):
    k, N = min(b, c), b + c
    return 1.0 if N == 0 else min(1.0, 2 * sum(math.comb(N, i) for i in range(k + 1)) / 2 ** N)

def holm(ps):
    ps = np.asarray(ps, dtype=float)
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj

print("\nExact McNemar per seed (b = A right/B wrong, c = A wrong/B right); p_holm = Holm-adjusted across the model pairs within each seed:")
res = {}
for s in seeds:
    raw = []
    for a, b_ in pairs:
        x, z = corr[(a, s)], corr[(b_, s)]
        bb, cc = int(((x == 1) & (z == 0)).sum()), int(((x == 0) & (z == 1)).sum())
        raw.append((bb, cc, mcnemar_exact(bb, cc)))
    adj = holm([r[2] for r in raw])
    for (a, b_), r, pa in zip(pairs, raw, adj):
        res.setdefault((a, b_), []).append(f"s{s}: b={r[0]} c={r[1]} p={r[2]:.3f} p_holm={pa:.3f}")
for (a, b_), items in res.items():
    print(f"  {a} vs {b_}: " + " | ".join(items))
print("\nNote: report differences as significant only if consistent across seeds and the bootstrap CI excludes 0 (use p_holm, not raw p).")
