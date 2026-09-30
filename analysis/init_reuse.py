"""
Group 1: initialization and reuse.

From latency-mode experiments (status ok):
    model fetch time (localhost dev/preview server), session creation time,
    first and second inference latency, and steady-state median (measure phase).
    first_run_overhead_ms = first inference - steady median: a difference of two measured values,
    i.e. the extra cost of the session's first run (shader/pipeline creation, buffer allocation).

From shapes-mode experiments (status ok), per (model, backend, shape):
    first_seen   first call of a shape never run before in the session
    revisit      first call of a shape seen earlier, right after a different shape
    steady       later calls in the same shape block
Reported as medians over blocks and trials, plus first_seen / steady and revisit / steady ratios.

Usage:
    python analysis/init_reuse.py benchmark_results/<latency run> [benchmark_results/<shapes run> ...]
"""
import argparse
from pathlib import Path

import pandas as pd


def load(dirs):
    runs, sessions = [], []
    for d in map(Path, dirs):
        runs.append(pd.read_csv(d / "runs.csv").assign(source=d.name))
        sessions.append(pd.read_csv(d / "sessions.csv").assign(source=d.name))
    return pd.concat(runs, ignore_index=True), pd.concat(sessions, ignore_index=True)


def init_table(runs, sessions):
    ok = sessions[(sessions["status"] == "ok") & (sessions["mode"] == "latency")]
    if ok.empty:
        return pd.DataFrame()
    keys = ["source", "trial", "model", "backend"]
    steady = (runs[runs["phase"] == "measure"].groupby(keys)["dur_ms"].median()
              .rename("steady_median_ms").reset_index())
    t = ok[keys + ["model_bytes", "fetch_ms", "session_create_ms", "first_run_ms", "second_run_ms"]].merge(steady, on=keys)
    t["first_run_overhead_ms"] = t["first_run_ms"] - t["steady_median_ms"]
    cols = ["fetch_ms", "session_create_ms", "first_run_ms", "second_run_ms", "steady_median_ms", "first_run_overhead_ms"]
    agg = t.groupby(["model", "backend"]).agg(
        trials=("trial", "count"), model_mb=("model_bytes", lambda b: b.iloc[0] / 2**20),
        **{f"{c}_median": (c, "median") for c in cols},
        **{f"{c}_min": (c, "min") for c in ("session_create_ms", "first_run_ms")},
        **{f"{c}_max": (c, "max") for c in ("session_create_ms", "first_run_ms")},
    ).reset_index()
    return agg


def shapes_table(runs, sessions):
    ok = sessions[(sessions["status"] == "ok") & (sessions["mode"] == "shapes")][["source", "trial", "model", "backend"]]
    s = runs[runs["phase"] == "shape"].merge(ok, on=["source", "trial", "model", "backend"])
    if s.empty:
        return pd.DataFrame()
    s["first_seen"] = s["first_seen"].astype(str).str.lower() == "true"
    s["kind"] = "steady"
    s.loc[s["block_pos"] == 0, "kind"] = "revisit"
    s.loc[s["first_seen"], "kind"] = "first_seen"
    t = (s.groupby(["model", "backend", "shape", "shape_value", "kind"])["dur_ms"]
         .agg(["median", "count"]).unstack("kind"))
    t.columns = [f"{kind}_{stat}" for stat, kind in t.columns]
    t = t.reset_index()
    for kind in ("first_seen", "revisit"):
        if f"{kind}_median" in t:
            t[f"{kind}_over_steady"] = t[f"{kind}_median"] / t["steady_median"]
    return t.sort_values(["model", "backend", "shape_value"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    runs, sessions = load(args.dirs)
    out = Path(args.out or args.dirs[0])
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        init = init_table(runs, sessions)
        if not init.empty:
            init.to_csv(out / "init_summary.csv", index=False)
            print("Initialization (medians over trials, ms):")
            print(init.round(2).to_string(index=False))
        shapes = shapes_table(runs, sessions)
        if not shapes.empty:
            shapes.to_csv(out / "shape_reuse_summary.csv", index=False)
            print("\nShape reuse (median ms):")
            print(shapes.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
