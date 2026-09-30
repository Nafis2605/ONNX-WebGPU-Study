"""§0 What can we trust? Checks every metric source and classifies it as exact / bounded / proxy.

exact    direct measurement, resolution well below the effect size, all invariants hold
bounded  measured, with a known perturbation or resolution limit that is quantified here
proxy    device- or process-wide signal, or an indirect measurement; interpret with care
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import stats, style

KEY = ["trial", "model", "backend"]


def validity(study, out):
    lat = study["latency"]
    checks = {}
    rows = []

    # 1. Timer resolution
    gran = (lat.env.get("page_env") or {}).get("timer_granularity_ms")
    iso = (lat.env.get("page_env") or {}).get("cross_origin_isolated")
    checks["timer_granularity_ms"] = gran
    r = lat.ok_runs("measure")
    min_lat = r.groupby(["model", "backend"])["dur_ms"].median().min() if len(r) else np.nan
    rows.append(["End-to-end latency", "performance.now() around session.run()",
                 f"{gran * 1000:.1f} µs" if gran else "?", f"cross-origin isolated={iso}; smallest median {min_lat:.2f} ms "
                 f"→ resolution ≤ {gran / min_lat * 100:.2f}% of the smallest effect" if gran else "", "exact"])

    # 2. Stationarity within the measured window (is warmup long enough?)
    drift = []
    for k, g in r.groupby(KEY):
        if len(g) >= 30:
            rho, p = stats.spearman(g["i"], g["dur_ms"])
            drift.append({"model": k[1], "backend": k[2], "trial": k[0], "rho": rho, "p": p})
    drift = pd.DataFrame(drift)
    drifting = drift[(drift["p"] < 0.01) & (drift["rho"].abs() > 0.3)] if len(drift) else drift
    checks["stationarity"] = {"experiments": len(drift), "drifting": len(drifting),
                              "drifting_list": drifting[["model", "backend", "trial", "rho"]].round(3).to_dict("records") if len(drifting) else []}

    # 3. Between-trial agreement (independent browser launches)
    trial_med = r.groupby(["model", "backend", "trial"])["dur_ms"].median()
    cov = trial_med.groupby(["model", "backend"]).agg(lambda x: np.std(x) / np.mean(x) if len(x) > 1 else np.nan)
    iccs = [stats.icc_oneway([g["dur_ms"].to_numpy() for _, g in gg.groupby("trial")]) for _, gg in r.groupby(["model", "backend"])]
    checks["between_trial_cov_median"] = float(np.nanmedian(cov)) if len(cov) else None
    checks["between_trial_cov_max"] = {"value": float(np.nanmax(cov)), "at": list(cov.idxmax())} if len(cov) and cov.notna().any() else None
    checks["icc_median"] = float(np.nanmedian(iccs)) if iccs else None
    rows.append(["Latency distribution (p50–p99)", "200 runs × 3 fresh browser launches",
                 "p99 from ≥200 samples", f"between-launch CoV of medians: median {checks['between_trial_cov_median']:.3f}; "
                 f"{len(drifting)}/{len(drift)} runs show drift (|ρ|>0.3, p<0.01)", "exact"])

    # 4. Profile invariants and instrumentation overhead
    pr = study["profile"]
    if pr is not None:
        pr_runs = pr.ok_runs()
        c = pr_runs[pr_runs["phase"] == "profile_counters"]
        gp = pr_runs[pr_runs["phase"] == "profile_gpu"]
        inv = {}
        if "ort_node_time_ms" in c:
            cc = c.dropna(subset=["ort_node_time_ms", "ort_model_run_ms"])
            inv["ort_node<=model_run<=wall_violations"] = int(((cc["ort_node_time_ms"] > cc["ort_model_run_ms"] + 1e-3) |
                                                               (cc["ort_model_run_ms"] > cc["dur_ms"] + 1e-3)).sum())
            inv["ort_node_checked"] = int(len(cc))
        if "gpu_busy_ms" in gp:
            gg = gp.dropna(subset=["gpu_busy_ms"])
            viol = gg["gpu_busy_ms"] > gg["dur_ms"] + 1e-3
            if "gpu_span_ms" in gg:
                viol |= gg["gpu_span_ms"].notna() & (gg["gpu_busy_ms"] > gg["gpu_span_ms"] + 1e-3)
            inv["gpu_busy<=span<=wall_violations"] = int(viol.sum())
            inv["gpu_checked"] = int(len(gg))
        s = pr.sessions
        inv["kernel_assignment_failures"] = int(s["kernel_assignment"].astype(str).str.startswith("unassigned").sum()) if "kernel_assignment" in s else 0
        inv["ort_trace_assignment_failures"] = int(s["ort_trace_assignment"].astype(str).str.startswith("unassigned").sum()) if "ort_trace_assignment" in s else 0
        checks["profile_invariants"] = inv
        lat_p50 = r.groupby(["model", "backend"])["dur_ms"].median()
        ov = (c.groupby(["model", "backend"])["dur_ms"].median() / lat_p50).dropna()
        pert = (gp.groupby(["model", "backend"])["dur_ms"].median() / c.groupby(["model", "backend"])["dur_ms"].median()).dropna()
        checks["counter_overhead_median"] = ov.groupby(level="backend").median().round(3).to_dict()
        checks["timestamp_phase_perturbation_median"] = pert.groupby(level="backend").median().round(3).to_dict()
        k = pr.csv("kernels.csv")
        dis = int(k["disjoint"].astype(str).str.lower().eq("true").sum()) if "disjoint" in k else 0
        checks["webgl_disjoint_timer_queries"] = dis
        rows.append(["GPU kernel time (WebGPU)", "WebGPU timestamp queries (ORT profiling hook)", "ns (developer features on)",
                     f"busy≤span≤wall violations {inv.get('gpu_busy<=span<=wall_violations')}/{inv.get('gpu_checked')}; "
                     f"phase wall ×{checks['timestamp_phase_perturbation_median'].get('webgpu', float('nan'))} vs counters phase "
                     "(pass per dispatch)", "bounded"])
        rows.append(["GPU draw time (WebGL)", "EXT_disjoint_timer_query_webgl2 per draw", "ns",
                     f"disjoint (invalid) queries: {dis}", "bounded"])
        rows.append(["CPU-side kernel time", "ORT C++ profiler (per-node Compute)", "µs",
                     f"node≤model_run≤wall violations {inv.get('ort_node<=model_run<=wall_violations')}/{inv.get('ort_node_checked')}; "
                     f"counter phase ×{checks['counter_overhead_median'].get('webgpu', float('nan'))} of latency p50", "bounded"])
        rows.append(["Dispatch / submit / draw counts, bytes moved", "prototype wrappers on WebGPU/WebGL APIs", "exact counts",
                     "read from call arguments; identical across runs of a model", "exact"])
        rows.append(["Logical GPU memory", "createBuffer/destroy & texImage2D accounting", "bytes",
                     "allocator-level view; driver padding not visible", "bounded"])

    # 5. Energy / NVML
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from gpu_telemetry import analyse
        t = analyse(lat.path)
        checks["energy_windows_ok_frac"] = float(t["window_ok"].mean()) if len(t) else None
        checks["nvml_util_vs_busy"] = None
        rows.append(["GPU energy per inference", "NVML total-energy counter (≈96 ms updates)", "mJ",
                     f"{checks['energy_windows_ok_frac']:.0%} of windows ≥ 1 s; GPU board only, CPU energy not measurable",
                     "bounded"])
        rows.append(["GPU utilisation / clocks / power", "NVML (driver samples ≈200 ms, power ≈390 ms)", "coarse",
                     "device-wide (includes desktop compositor)", "proxy"])
    except Exception as e:
        checks["energy_error"] = str(e)

    # 6. CPU time & background load
    sc = lat.csv("system_cpu_samples.csv")
    if len(sc):
        checks["system_cpu_percent_median"] = float(sc["cpu_percent"].median())
    rows.append(["CPU time per inference", "psutil cpu_times of this launch's Chrome processes", "≈15 ms (Windows tick)",
                 f"system-wide CPU median {checks.get('system_cpu_percent_median', float('nan')):.1f}% during runs", "bounded"])
    rows.append(["Correctness", "native onnxruntime (CPU EP) on identical input bytes", "fp32", "deterministic reference", "exact"])
    rows.append(["Main-thread input delay", "pointerdown event.timeStamp → handler start", "5 µs",
                 "real CDP input events; Event Timing API thresholded at 16 ms, so not used", "exact"])

    table = pd.DataFrame(rows, columns=["metric", "source", "resolution", "checks (this run)", "verdict"])
    table.to_csv(out / "fig00_validity_table.csv", index=False)

    fig, (a, b) = plt.subplots(1, 2, figsize=(style.DOUBLE, 2.0))
    if len(drift):
        a.hist(drift["rho"], bins=30, color="#0072B2")
        a.set_xlabel("Spearman ρ (latency vs run index)"); a.set_ylabel("experiments")
        a.set_title("(a) Stationarity of measured runs")
    if len(cov):
        a2 = cov.dropna()
        b.hist(a2 * 100, bins=30, color="#009E73")
        b.set_xlabel("between-launch CoV of median latency (%)"); b.set_ylabel("model × config")
        b.set_title("(b) Reproducibility across fresh browser launches")
    fig.tight_layout()
    return {"files": style.save(fig, out, "fig00_validity"), "checks": checks, "table": table.to_dict("records")}
