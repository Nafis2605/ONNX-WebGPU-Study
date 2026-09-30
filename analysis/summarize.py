"""
Group 2 statistics: end-to-end latency distribution and closed-loop throughput.

Only experiments with status == "ok" (backend verified by API counters) are included.
Quantiles use numpy's default linear interpolation (Hyndman & Fan type 7).

Writes, next to the input files:
    latency_by_trial.csv   one row per (model, backend, trial)
    latency_summary.csv    one row per (model, backend), pooled over trials, with bootstrap CIs

Usage:
    python analysis/summarize.py benchmark_results/<timestamp> [more dirs ...]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

QUANTILES = {"p50": 0.50, "p90": 0.90, "p95": 0.95, "p99": 0.99}
# With fewer samples than this, the sample p99 is set by one or two runs; it's still reported but flagged.
MIN_RUNS_FOR_P99 = 100


def load(dirs):
    runs, sessions = [], []
    for d in dirs:
        d = Path(d)
        r = pd.read_csv(d / "runs.csv")
        s = pd.read_csv(d / "sessions.csv")
        r["source"] = s["source"] = d.name
        runs.append(r)
        sessions.append(s)
    return pd.concat(runs, ignore_index=True), pd.concat(sessions, ignore_index=True)


def describe(durations):
    x = np.asarray(durations, dtype=float)
    out = {"n": len(x), "mean_ms": x.mean(), "std_ms": x.std(ddof=1) if len(x) > 1 else np.nan,
           "min_ms": x.min(), "max_ms": x.max()}
    for name, q in QUANTILES.items():
        out[f"{name}_ms"] = np.quantile(x, q)
    out["iqr_ms"] = np.quantile(x, 0.75) - np.quantile(x, 0.25)
    out["cov"] = out["std_ms"] / out["mean_ms"]
    out["p99_reliable"] = len(x) >= MIN_RUNS_FOR_P99
    return out


def throughput(group):
    """Completed requests per second over the measured window, including gaps between runs."""
    start = group["t_start_epoch_ms"].min()
    end = (group["t_start_epoch_ms"] + group["dur_ms"]).max()
    return len(group) / ((end - start) / 1000.0)


def bootstrap_ci(x, stat_q, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    stats = np.quantile(rng.choice(x, size=(n_boot, len(x)), replace=True), stat_q, axis=1)
    return np.quantile(stats, 0.025), np.quantile(stats, 0.975)


def summarize(runs, sessions):
    ok = sessions[sessions["status"] == "ok"][["source", "trial", "model", "backend"]]
    measured = runs[runs["phase"] == "measure"].merge(ok, on=["source", "trial", "model", "backend"])

    by_trial = []
    for (src, model, backend, trial), g in measured.groupby(["source", "model", "backend", "trial"]):
        by_trial.append({"source": src, "model": model, "backend": backend, "trial": trial,
                         **describe(g["dur_ms"]), "throughput_rps": throughput(g)})
    by_trial = pd.DataFrame(by_trial)

    summary = []
    for (model, backend), g in measured.groupby(["model", "backend"]):
        t = by_trial[(by_trial["model"] == model) & (by_trial["backend"] == backend)]
        row = {"model": model, "backend": backend, "trials": len(t), **describe(g["dur_ms"]),
               "throughput_rps_mean": t["throughput_rps"].mean(),
               "trial_median_min_ms": t["p50_ms"].min(), "trial_median_max_ms": t["p50_ms"].max()}
        row["p50_ci95_lo_ms"], row["p50_ci95_hi_ms"] = bootstrap_ci(g["dur_ms"], 0.50)
        row["p99_ci95_lo_ms"], row["p99_ci95_hi_ms"] = bootstrap_ci(g["dur_ms"], 0.99)
        summary.append(row)
    summary = pd.DataFrame(summary)

    not_ok = sessions[sessions["status"] != "ok"][
        ["source", "trial", "model", "backend", "status", "failure_category", "failure_message"]]
    return by_trial, summary, not_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None, help="output directory (default: the first input dir)")
    args = ap.parse_args()

    runs, sessions = load(args.dirs)
    by_trial, summary, not_ok = summarize(runs, sessions)
    out = Path(args.out or args.dirs[0])
    by_trial.to_csv(out / "latency_by_trial.csv", index=False)
    summary.to_csv(out / "latency_summary.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        cols = ["model", "backend", "trials", "n", "p50_ms", "p95_ms", "p99_ms", "cov", "throughput_rps_mean", "p99_reliable"]
        print(summary[cols].round(3).to_string(index=False) if len(summary) else "No verified experiments.")
        if len(not_ok):
            print("\nExcluded experiments:")
            print(not_ok.to_string(index=False, max_colwidth=90))
    print(f"\nWrote {out / 'latency_by_trial.csv'} and {out / 'latency_summary.csv'}")


if __name__ == "__main__":
    main()
