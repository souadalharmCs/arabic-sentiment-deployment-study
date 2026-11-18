"""Run the full load-test campaign: systems x user levels x repetitions.

Usage (from the experiments folder):
    python run_experiments.py                              # full campaign
    python run_experiments.py --users 10 --reps 1 --duration 60   # quick dry run
Finished runs are skipped, so the script can be stopped and restarted.
"""
import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
HOST = "http://localhost:8080"

# name -> (folder, compose file, number of API replicas)
SYSTEMS = {
    "mono1x": ("mono1x", "docker-compose.mono1x.yml", 1),
    "mono3x": ("mono3x", "docker-compose.mono3x.yml", 1),
    "micro3": ("micro3", "docker-compose.micro3.yml", 3),
    # extra experiments: least_conn Nginx, and one container with 3 uvicorn worker processes
    "micro3lc": ("micro3lc", "docker-compose.micro3lc.yml", 3),
    "mono3w": ("mono3w", "docker-compose.mono3w.yml", 3),
    "mono6w": ("mono6w", "docker-compose.mono6w.yml", 6),
    "mono3w2t": ("mono3w2t", "docker-compose.mono3w2t.yml", 3),
}
MAIN_SYSTEMS = ["mono1x", "mono3x", "micro3"]


def compose(system, *args, capture=False):
    folder, fname, _ = SYSTEMS[system]
    return subprocess.run(
        ["docker", "compose", "-f", fname, *args],
        cwd=os.path.join(ROOT, folder),
        capture_output=capture, text=True, encoding="utf-8", errors="replace",
    )


def stop_all():
    for s in SYSTEMS:
        compose(s, "down", capture=True)


def wait_ready(system, timeout=900):
    replicas = SYSTEMS[system][2]
    t0 = time.time()
    while time.time() - t0 < timeout:
        out = compose(system, "logs", "--no-color", capture=True).stdout or ""
        if out.count("Application startup complete") >= replicas:
            return True
        time.sleep(5)
    return False


def warmup(sentences, n):
    """Varied real sentences: each new input length triggers one-off CPU kernel setup."""
    rng = random.Random(0)
    bad = 0
    for s in rng.sample(sentences, min(n, len(sentences))):
        try:
            r = requests.post(f"{HOST}/predict", json={"text": s}, timeout=60)
            bad += r.status_code != 200
        except requests.RequestException:
            bad += 1
    return bad


class StatsSampler(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = path
        self.stop_evt = threading.Event()

    def run(self):
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time", "container", "cpu_percent", "mem_usage"])
            while not self.stop_evt.is_set():
                out = subprocess.run(
                    ["docker", "stats", "--no-stream", "--format",
                     "{{.Name}},{{.CPUPerc}},{{.MemUsage}}"],
                    capture_output=True, text=True, encoding="utf-8", errors="replace",
                ).stdout or ""
                now = datetime.now().isoformat(timespec="seconds")
                for line in out.strip().splitlines():
                    parts = line.split(",", 2)
                    if len(parts) == 3:
                        w.writerow([now, *parts])
                f.flush()


def log_run(row):
    path = os.path.join(RESULTS, "run_log.csv")
    new = not os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["system", "users", "rep", "start", "end", "duration_s", "locust_exit", "warmup_failures"])
        w.writerow(row)


def run_one(system, users, rep, sentences, args):
    name = f"{system}_u{users}_rep{rep}"
    if os.path.exists(os.path.join(RESULTS, f"{name}_stats.csv")):
        print(f"[skip] {name} already done")
        return
    print(f"\n=== {name} ===  {datetime.now():%H:%M:%S}")
    stop_all()
    compose(system, "up", "-d", capture=True)
    if not wait_ready(system):
        print(f"!! {system} did not become ready, skipping {name}")
        return
    bad = warmup(sentences, args.warmup)
    time.sleep(5)

    nginx_log = os.path.join(ROOT, SYSTEMS[system][0], "logs", "access.log")
    if system.startswith("micro3") and os.path.exists(nginx_log):
        open(nginx_log, "w").close()  # truncate in place; nginx keeps appending

    sampler = StatsSampler(os.path.join(RESULTS, f"{name}_docker_stats.csv"))
    sampler.start()
    start = datetime.now()
    proc = subprocess.run(
        [sys.executable, "-m", "locust", "-f", "locustfile.py", "--headless",
         "-u", str(users), "-r", str(max(1, users // 10)), "-t", f"{args.duration}s",
         "--host", HOST, "--csv", os.path.join(args.results_dir, name), "--csv-full-history",
         "--exit-code-on-error", "0"],
        cwd=ROOT,
    )
    end = datetime.now()
    sampler.stop_evt.set()
    sampler.join(timeout=20)

    if system.startswith("micro3") and os.path.exists(nginx_log):
        shutil.copy(nginx_log, os.path.join(RESULTS, f"{name}_nginx.log"))

    dur = (end - start).total_seconds()
    if dur < args.duration - 5:
        print(f"!! WARNING: run lasted {dur:.0f}s instead of {args.duration}s")
    log_run([system, users, rep, start.isoformat(timespec="seconds"),
             end.isoformat(timespec="seconds"), round(dur, 1), proc.returncode, bad])
    time.sleep(20)  # cool-down


def main():
    global RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, nargs="+", default=[10, 50, 100, 150])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--duration", type=int, default=300, help="seconds per run")
    ap.add_argument("--warmup", type=int, default=200, help="warm-up requests per start")
    ap.add_argument("--systems", nargs="+", default=MAIN_SYSTEMS, choices=list(SYSTEMS))
    ap.add_argument("--results-dir", default="results", help="output folder (relative to this script)")
    args = ap.parse_args()
    RESULTS = os.path.join(ROOT, args.results_dir)

    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(ROOT, "sentences.txt"), encoding="utf-8") as f:
        sentences = [l.strip() for l in f if l.strip()]

    total = args.reps * len(args.users) * len(args.systems)
    print(f"{total} runs planned, {args.duration}s each")
    try:
        for rep in range(1, args.reps + 1):
            for users in args.users:
                for system in args.systems:  # systems alternate at the same load level
                    run_one(system, users, rep, sentences, args)
    finally:
        stop_all()
    print("\nAll done.")


if __name__ == "__main__":
    main()
