"""
NVIDIA GPU telemetry per experiment, aligned to the measured inference window.

Windows (Unix epoch ms, one clock for browser and NVML):
    baseline  idle GPU before the browser launched (run_benchmark.py --baseline-s)
    measure   first measured inference start .. last measured inference end (runs.csv)

Per experiment (status ok, latency mode):
    energy_j                 cumulative NVML energy counter, linearly interpolated at the window edges
                             (the counter refreshes about every 96 ms, so windows under ~1 s are flagged)
    mean_power_w             energy_j / window duration
    idle_power_w             mean power over the baseline window
    energy_per_inference_mj  energy_j / measured runs
    dynamic_energy_per_inference_mj  (mean_power_w - idle_power_w) * duration / runs
    gpu_util_mean_pct, mem_util_mean_pct   driver utilization samples inside the window (~200 ms apart)
    sm_clock_mean_mhz, mem_clock_mean_mhz, temp_max_c, pstates, clock_reasons (union of throttle reasons)
    mem_used_peak_delta_mb   peak device memory in the window minus the baseline mean (device-wide)
    pcie_tx_mean_kbps, pcie_rx_mean_kbps   PCIe throughput samples starting inside the window

Per-inference energy for a single run isn't resolvable: runs are shorter than the counter's refresh period.

Usage:
    python analysis/gpu_telemetry.py benchmark_results/<timestamp>
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchkit.telemetry import decode_clock_reasons  # noqa: E402

KEY = ["trial", "model", "backend"]
MIN_WINDOW_S = 1.0


def window_value(t, v, at):
    """Linear interpolation of a cumulative counter at time `at`, or NaN outside the samples."""
    if at < t[0] or at > t[-1]:
        return np.nan
    return float(np.interp(at, t, v))


def analyse(out_dir):
    out_dir = Path(out_dir)
    sessions = pd.read_csv(out_dir / "sessions.csv")
    runs = pd.read_csv(out_dir / "runs.csv")
    samples = pd.read_csv(out_dir / "gpu_samples.csv")
    util = pd.read_csv(out_dir / "gpu_util_samples.csv") if (out_dir / "gpu_util_samples.csv").exists() else pd.DataFrame(columns=KEY + ["t_epoch_ms", "kind", "percent"])
    pcie = pd.read_csv(out_dir / "gpu_pcie_samples.csv") if (out_dir / "gpu_pcie_samples.csv").exists() else pd.DataFrame(columns=KEY + ["t_epoch_ms", "tx_kbps", "rx_kbps"])
    marks = pd.read_csv(out_dir / "gpu_marks.csv")

    rows = []
    ok = sessions[(sessions["status"] == "ok") & (sessions["mode"] == "latency")]
    for _, s in ok.iterrows():
        k = tuple(s[c] for c in KEY)
        sel = lambda df: df[(df["trial"] == k[0]) & (df["model"] == k[1]) & (df["backend"] == k[2])]
        r = sel(runs)
        r = r[r["phase"] == "measure"]
        smp = sel(samples).sort_values("t_epoch_ms")
        mk = sel(marks).set_index("mark")["t_epoch_ms"]
        if r.empty or smp.empty or "baseline_start" not in mk:
            continue
        w0 = r["t_start_epoch_ms"].min()
        w1 = (r["t_start_epoch_ms"] + r["dur_ms"]).max()
        dur_s = (w1 - w0) / 1000.0
        t = smp["t_epoch_ms"].to_numpy()
        energy = smp["energy_mj"].to_numpy(dtype=float)
        e_j = (window_value(t, energy, w1) - window_value(t, energy, w0)) / 1000.0

        base = smp[(smp["t_epoch_ms"] >= mk["baseline_start"]) & (smp["t_epoch_ms"] < mk["baseline_end"])]
        b_t = base["t_epoch_ms"].to_numpy()
        idle_power = ((base["energy_mj"].iloc[-1] - base["energy_mj"].iloc[0]) / (b_t[-1] - b_t[0])
                      if len(base) > 1 and b_t[-1] > b_t[0] else np.nan)  # mJ/ms = W
        inwin = smp[(smp["t_epoch_ms"] >= w0) & (smp["t_epoch_ms"] <= w1)]
        u = sel(util)
        u = u[(u["t_epoch_ms"] >= w0) & (u["t_epoch_ms"] <= w1)]
        p = sel(pcie)
        p = p[(p["t_epoch_ms"] >= w0) & (p["t_epoch_ms"] <= w1)]
        n = len(r)
        mean_power = e_j / dur_s if dur_s > 0 else np.nan
        reasons = 0
        for m in inwin["clock_reasons"].dropna().astype(int):
            reasons |= m
        rows.append({
            **dict(zip(KEY, k)),
            "measured_runs": n, "window_s": dur_s, "window_ok": dur_s >= MIN_WINDOW_S,
            "energy_j": e_j, "mean_power_w": mean_power, "idle_power_w": idle_power,
            "energy_per_inference_mj": e_j * 1000.0 / n,
            "dynamic_energy_per_inference_mj": (mean_power - idle_power) * dur_s * 1000.0 / n,
            "gpu_util_mean_pct": u.loc[u["kind"] == "gpu", "percent"].mean(),
            "mem_util_mean_pct": u.loc[u["kind"] == "mem", "percent"].mean(),
            "util_samples": int((u["kind"] == "gpu").sum()),
            "sm_clock_mean_mhz": inwin["sm_clock_mhz"].mean(),
            "mem_clock_mean_mhz": inwin["mem_clock_mhz"].mean(),
            "temp_max_c": inwin["temp_c"].max(),
            "pstates": ",".join(str(x) for x in sorted(inwin["pstate"].unique())),
            "clock_reasons": decode_clock_reasons(reasons),
            "mem_used_peak_delta_mb": inwin["mem_used_mb"].max() - base["mem_used_mb"].mean(),
            "pcie_tx_mean_kbps": p["tx_kbps"].mean(), "pcie_rx_mean_kbps": p["rx_kbps"].mean(),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    args = ap.parse_args()
    t = analyse(args.dir)
    out = Path(args.dir) / "gpu_telemetry_summary.csv"
    t.to_csv(out, index=False)
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        cols = ["model", "backend", "trial", "window_s", "mean_power_w", "idle_power_w", "energy_per_inference_mj",
                "dynamic_energy_per_inference_mj", "gpu_util_mean_pct", "sm_clock_mean_mhz", "temp_max_c", "clock_reasons",
                "mem_used_peak_delta_mb"]
        print(t[cols].round(2).to_string(index=False) if len(t) else "No telemetry for verified latency experiments.")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
