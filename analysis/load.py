"""
Group 8: useful application performance under open-loop load (load-mode runs).

Per (model, backend, rate), pooled over trials:
    offered_rps              arrivals / window
    goodput_rps              requests finished within the SLO / window (deadline-compliant throughput)
    deadline_miss_rate       finished-late + rejected + unfinished, over all arrivals
    reject_rate              arrivals refused because the queue was full
    unfinished_rate          accepted but not served before the drain deadline
    latency_p50/p95/p99_ms   scheduled arrival to completion, finished requests only
    dispatch_lag_p50_ms      how late the page noticed arrivals (timer wake-up, busy main thread);
                             part of latency, reported so harness lag stays visible
    frame_missed_frac        (--with-animation runs) fraction of frame intervals > 1.5 x the
                             median interval at the lowest rate

Max sustainable arrival rate per (model, backend): the highest tested rate whose deadline-miss
rate is <= --max-miss (and, with --max-frame-miss, whose frame-miss fraction is also within
bound). Only tested rates are reported: nothing is interpolated.

Usage:
    python analysis/load.py benchmark_results/<load run> --slo-ms 100 [--max-miss 0.01] [--max-frame-miss 0.05]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["trial", "model", "backend"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--slo-ms", type=float, required=True, help="latency deadline per request")
    ap.add_argument("--max-miss", type=float, default=0.01)
    ap.add_argument("--max-frame-miss", type=float, default=None)
    args = ap.parse_args()
    d = Path(args.dir)
    sessions = pd.read_csv(d / "sessions.csv")
    ok = sessions[(sessions["status"] == "ok") & (sessions["mode"] == "load")][KEY]
    req = pd.read_csv(d / "load_requests.csv").merge(ok, on=KEY)
    env_args = __import__("json").loads((d / "env.json").read_text())["args"]
    window_s = env_args["load_s"]

    req["met"] = (~req["rejected"]) & (~req["unfinished"]) & (req["latency_ms"] <= args.slo_ms)
    rows = []
    for (model, backend, rate), g in req.groupby(["model", "backend", "rate"]):
        trials = g["trial"].nunique()
        done = g[g["latency_ms"].notna()]
        rows.append({
            "model": model, "backend": backend, "rate": rate, "trials": trials, "arrivals": len(g),
            "offered_rps": len(g) / (window_s * trials),
            "goodput_rps": g["met"].sum() / (window_s * trials),
            "deadline_miss_rate": 1 - g["met"].mean(),
            "reject_rate": g["rejected"].mean(),
            "unfinished_rate": g["unfinished"].mean(),
            "latency_p50_ms": done["latency_ms"].quantile(0.5) if len(done) else np.nan,
            "latency_p95_ms": done["latency_ms"].quantile(0.95) if len(done) else np.nan,
            "latency_p99_ms": done["latency_ms"].quantile(0.99) if len(done) else np.nan,
            "service_p50_ms": done["service_ms"].quantile(0.5) if len(done) else np.nan,
            "dispatch_lag_p50_ms": g["dispatch_lag_ms"].quantile(0.5) if "dispatch_lag_ms" in g else np.nan,
        })
    t = pd.DataFrame(rows)

    frames_path = d / "load_frames.csv"
    if frames_path.exists():
        f = pd.read_csv(frames_path).merge(ok, on=KEY)
        low = f[f["rate"] == f.groupby(KEY)["rate"].transform("min")].groupby(KEY)["interval_ms"].median().rename("ref_ms")
        f = f.merge(low.reset_index(), on=KEY)
        f["missed"] = f["interval_ms"] > 1.5 * f["ref_ms"]
        t = t.merge(f.groupby(["model", "backend", "rate"])["missed"].mean().rename("frame_missed_frac").reset_index(),
                    on=["model", "backend", "rate"], how="left")

    feasible = t["deadline_miss_rate"] <= args.max_miss
    if args.max_frame_miss is not None and "frame_missed_frac" in t:
        feasible &= t["frame_missed_frac"] <= args.max_frame_miss
    t["feasible"] = feasible
    msr = (t[t["feasible"]].groupby(["model", "backend"])["rate"].max().rename("max_sustainable_rps")
           .reindex(t.groupby(["model", "backend"]).size().index))
    t.to_csv(d / "load_summary.csv", index=False)
    msr.reset_index().to_csv(d / "load_max_sustainable.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(t.round(4).to_string(index=False))
        print(f"\nMax sustainable tested rate (SLO {args.slo_ms} ms, miss <= {args.max_miss}):")
        print(msr.reset_index().to_string(index=False))


if __name__ == "__main__":
    main()
