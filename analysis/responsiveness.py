"""
Group 4: browser responsiveness with and without inference (responsiveness-mode runs).

Frame metrics are only produced for runs recorded with --animate. The display refresh interval is
then measured, not assumed: the median rAF interval of the baseline phase in each experiment.

Per (model, backend, phase):
    frame_interval_p50/p95/p99_ms
    missed_deadline_frac   fraction of frame intervals > 1.5 x refresh interval (at least one
                           vsync missed)
    dropped_frames_per_s   sum(max(0, round(interval / refresh) - 1)) / phase seconds
    clicks                 every scripted click, from the page's own pointerdown listener:
    click_input_delay_p50/p95_ms     event timestamp -> handler start
    click_to_next_frame_p50/p95_ms   event timestamp -> start of the next animation frame
                                     (lower bound on time to the next paint)
    et_entries             Event Timing entries (only events lasting >= 16 ms are reported, so this
                           stream under-samples fast interactions)
    et_input_delay_p50/p95_ms   processingStart - startTime
    et_duration_p50/p95_ms      startTime to next paint (Chrome rounds this to 8 ms)
    loaf_per_s, loaf_blocking_ms_per_s   Long Animation Frames (> 50 ms) and their blocking time

Usage:
    python analysis/responsiveness.py benchmark_results/<responsiveness run>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["trial", "model", "backend"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    args = ap.parse_args()
    d = Path(args.dir)
    sessions = pd.read_csv(d / "sessions.csv")
    ok = sessions[(sessions["status"] == "ok") & (sessions["mode"] == "responsiveness")][KEY]
    empty = pd.DataFrame(columns=KEY + ["phase"])
    load = lambda name: pd.read_csv(d / name).merge(ok, on=KEY) if (d / name).exists() else empty
    phases = load("phases.csv")
    inter, clicks, loaf = load("interactions.csv"), load("clicks.csv"), load("loaf.csv")
    # Frame metrics exist only for runs with --animate (the study measures DL cost, not rendering).
    frames = load("frames.csv")
    if len(frames):
        refresh = frames[frames["phase"] == "baseline"].groupby(KEY)["interval_ms"].median().rename("refresh_ms")
        frames = frames.merge(refresh.reset_index(), on=KEY)
        frames["missed"] = frames["interval_ms"] > 1.5 * frames["refresh_ms"]
        frames["dropped"] = np.maximum(0, np.round(frames["interval_ms"] / frames["refresh_ms"]) - 1)
    phases["seconds"] = (phases["end_ms"] - phases["start_ms"]) / 1000.0

    def sel(df, trial, model, backend, phase):
        return df[(df["trial"] == trial) & (df["model"] == model) & (df["backend"] == backend) & (df["phase"] == phase)]

    rows = []
    for (trial, model, backend, phase), ph in phases.groupby(KEY + ["phase"]):
        secs = ph["seconds"].sum()
        it, ck, lf = (sel(x, trial, model, backend, phase) for x in (inter, clicks, loaf))
        f = sel(frames, trial, model, backend, phase) if len(frames) else frames
        row = {"trial": trial, "model": model, "backend": backend, "phase": phase, "seconds": secs}
        if len(f):
            row.update({
                "refresh_ms": f["refresh_ms"].iloc[0], "frames": len(f),
                "frame_interval_p50_ms": f["interval_ms"].quantile(0.5),
                "frame_interval_p95_ms": f["interval_ms"].quantile(0.95),
                "frame_interval_p99_ms": f["interval_ms"].quantile(0.99),
                "frame_interval_max_ms": f["interval_ms"].max(),
                "missed_deadline_frac": f["missed"].mean(),
                "dropped_frames_per_s": f["dropped"].sum() / secs if secs else np.nan,
            })
        row.update({
            "clicks": len(ck),
            "click_input_delay_p50_ms": ck["input_delay_ms"].quantile(0.5) if len(ck) else np.nan,
            "click_input_delay_p95_ms": ck["input_delay_ms"].quantile(0.95) if len(ck) else np.nan,
            "click_to_next_frame_p50_ms": ck["to_next_frame_ms"].quantile(0.5) if len(ck) else np.nan,
            "click_to_next_frame_p95_ms": ck["to_next_frame_ms"].quantile(0.95) if len(ck) else np.nan,
            "et_entries": len(it),
            "et_input_delay_p50_ms": it["input_delay_ms"].quantile(0.5) if len(it) else np.nan,
            "et_input_delay_p95_ms": it["input_delay_ms"].quantile(0.95) if len(it) else np.nan,
            "et_duration_p50_ms": it["duration"].quantile(0.5) if len(it) else np.nan,
            "et_duration_p95_ms": it["duration"].quantile(0.95) if len(it) else np.nan,
            "loaf_per_s": len(lf) / secs if secs else np.nan,
            "loaf_blocking_ms_per_s": lf["blocking_duration"].sum() / secs if secs and len(lf) else 0.0,
        })
        rows.append(row)
    t = pd.DataFrame(rows)
    summary = t.groupby(["model", "backend", "phase"]).median(numeric_only=True).drop(columns="trial").reset_index()
    summary.to_csv(d / "responsiveness_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
