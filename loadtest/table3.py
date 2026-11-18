"""Build Table 3 (throughput, median, p95, p99, failures) from Locust *_stats.csv files. Uses only the standard library.

Usage (from the experiments folder):
  python table3.py results                       # folder that holds the MAIN campaign (mono1x/mono3x/micro3, users 10 50 100 150)
  python table3.py results_tuned mono3x micro3   # optional: other systems; users are detected automatically
Each value is the mean over the repetitions found (rep1, rep2, rep3); failures = mean per run.
"""
import csv, glob, os, re, sys
from collections import defaultdict

folder = sys.argv[1] if len(sys.argv) > 1 else "results"
systems = sys.argv[2:] or ["mono1x", "mono3x", "micro3"]
rows = defaultdict(list)
for f in glob.glob(os.path.join(folder, "*_stats.csv")):
    m = re.match(r"(.+)_u(\d+)_rep(\d+)_stats\.csv$", os.path.basename(f))
    if not m or m.group(1) not in systems:
        continue
    with open(f, encoding="utf-8", newline="") as fh:
        agg = next((r for r in csv.DictReader(fh) if r["Name"] == "Aggregated"), None)
    if agg:
        rows[(int(m.group(2)), m.group(1))].append(agg)

def fmt(x):  # 2 decimals below 1 s, else 1 decimal
    return f"{x:.2f}" if x < 1 else f"{x:.1f}"

print("| # | Users | Deployment configuration | Throughput (requests/s) | Median (s) | p95 (s) | p99 (s) | Failures per run | reps |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
n = 0
for users in sorted({u for u, _ in rows}):
    for s in systems:
        g = rows.get((users, s))
        if not g:
            continue
        n += 1
        mean = lambda k: sum(float(r[k]) for r in g) / len(g)
        fails = mean("Failure Count")
        print(f"| {n} | {users} | {s} | {mean('Requests/s'):.1f} | {fmt(mean('Median Response Time')/1000)} | "
              f"{fmt(mean('95%')/1000)} | {fmt(mean('99%')/1000)} | {fails:.0f} | {len(g)} |" if fails >= 10 else
              f"| {n} | {users} | {s} | {mean('Requests/s'):.1f} | {fmt(mean('Median Response Time')/1000)} | "
              f"{fmt(mean('95%')/1000)} | {fmt(mean('99%')/1000)} | {fails:.1f} | {len(g)} |")
