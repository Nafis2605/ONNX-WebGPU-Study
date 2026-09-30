"""§3 Why? Where WebGPU's time goes, and which costs it adds on top of GPU compute."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import stats, style

OP_ALIASES = {"FusedConv": "Conv", "FusedMatMul": "MatMul", "FusedGemm": "Gemm"}


def _profile(study):
    st = study["profile"]
    if st is None:
        return None, None, None
    runs = st.ok_runs()
    kernels = st.csv("kernels.csv").merge(st.ok_keys(), on=["trial", "model", "backend"]) if len(st.ok_keys()) else pd.DataFrame()
    return st, runs, kernels


def decomposition_table(study):
    st, runs, _ = _profile(study)
    if st is None:
        return pd.DataFrame()
    lat = study.steady("latency")
    feats = study.features.set_index("model")
    cnt = runs[runs["phase"] == "profile_counters"]
    gpu = runs[runs["phase"] == "profile_gpu"]
    rows = []
    for (m, c), g in cnt.groupby(["model", "backend"]):
        if not c.startswith("webgpu") and c != "webgl":
            continue
        gg = gpu[(gpu["model"] == m) & (gpu["backend"] == c)]
        rows.append({
            "model": m, "backend": c, "macs": feats.loc[m, "macs"],
            "wall_latency_p50_ms": float(np.median(lat[(m, c)])) if (m, c) in lat.index else np.nan,
            "wall_profile_ms": g["dur_ms"].median(),
            "cpu_node_ms": g["ort_node_time_ms"].median() if "ort_node_time_ms" in g else np.nan,
            "gpu_busy_ms": gg["gpu_busy_ms"].median() if "gpu_busy_ms" in gg and len(gg) else np.nan,
            "gpu_span_ms": gg["gpu_span_ms"].median() if "gpu_span_ms" in gg and len(gg) else np.nan,
            "dispatches": g["webgpu_dispatches"].median() if "webgpu_dispatches" in g else np.nan,
            "draws": g["webgl_draws"].median() if "webgl_draws" in g else np.nan,
            "submits": g["webgpu_submits"].median() if "webgpu_submits" in g else np.nan,
            "map_read_wait_ms": g["webgpu_map_read_wait_ms"].median() if "webgpu_map_read_wait_ms" in g else np.nan,
            "upload_bytes": (g.get("webgpu_write_buffer_bytes", 0) + g.get("webgl_tex_upload_bytes", 0)).median(),
            "readback_bytes": (g.get("webgpu_map_read_bytes", 0) + g.get("webgl_read_pixels_bytes", 0)).median(),
        })
    t = pd.DataFrame(rows)
    if len(t):
        t["gpu_busy_frac"] = t["gpu_busy_ms"] / t["wall_latency_p50_ms"]
        t["cpu_prep_frac"] = t["cpu_node_ms"] / t["wall_latency_p50_ms"]
        t["cpu_over_gpu"] = t["cpu_node_ms"] / t["gpu_busy_ms"]
    return t


def time_decomposition(study, out):
    t = decomposition_table(study)
    if t.empty:
        return {"skipped": "no profile stage"}
    t.to_csv(out / "fig06_decomposition_table.csv", index=False)
    j = t[t["backend"] == "webgpu"].sort_values("macs")
    fig, (a, b) = plt.subplots(1, 2, figsize=(style.DOUBLE, 2.4), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(j))
    w = 0.27
    a.bar(x - w, j["wall_latency_p50_ms"], w, color="#444444", label="end-to-end latency (p50)")
    a.bar(x, j["cpu_node_ms"], w, color="#D55E00", label="CPU-side kernel preparation (ORT nodes)")
    a.bar(x + w, j["gpu_busy_ms"], w, color="#0072B2", label="GPU kernel execution (timestamps)")
    a.set_yscale("log")
    a.set_xticks(x, [style.short(m) for m in j["model"]], rotation=45, ha="right")
    a.set_ylabel("ms per inference (log)")
    a.legend(fontsize=6, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, borderaxespad=0.2)
    a.set_title("(a) WebGPU (JSEP): where one inference's time goes", pad=18)
    # Graph capture replays recorded commands, so its CPU-side kernel time is 0 by construction.
    for c in ["webgpu"]:
        g = t[(t["backend"] == c) & t["cpu_over_gpu"].notna() & (t["cpu_over_gpu"] > 0)]
        b.scatter(g["macs"] / 1e9, g["cpu_over_gpu"], color=style.color(c), marker=style.marker(c), label=style.label(c), s=14)
        for _, row in g.iterrows():
            if row["cpu_over_gpu"] > 1:
                b.annotate(style.short(row["model"]), (row["macs"] / 1e9, row["cpu_over_gpu"]), fontsize=6,
                           xytext=(3, 2), textcoords="offset points")
    b.axhline(1, color="k", ls="--", lw=0.6)
    b.set_xscale("log"); b.set_yscale("log")
    b.set_xlabel("GMACs (log)"); b.set_ylabel("CPU prep time / GPU busy time")
    b.set_title("(b) >1: the CPU, not the GPU, is the bottleneck", pad=18)
    fig.tight_layout()
    cpu_bound = j[j["cpu_over_gpu"] > 1]["model"].tolist()
    rho = stats.spearman(np.log(j["macs"]), np.log(j["cpu_over_gpu"])) if j["cpu_over_gpu"].notna().sum() > 2 else None
    return {"files": style.save(fig, out, "fig06_time_decomposition"),
            "cpu_bound_models_webgpu": cpu_bound,
            "gpu_busy_frac_range": [float(j["gpu_busy_frac"].min()), float(j["gpu_busy_frac"].max())],
            "cpu_over_gpu_by_model": dict(zip(j["model"], j["cpu_over_gpu"].round(3))),
            "spearman_cpu_over_gpu_vs_macs": rho}


def dispatch_overhead(study, out):
    t = decomposition_table(study)
    if t.empty:
        return {"skipped": "no profile stage"}
    fig, ax = plt.subplots(figsize=(style.SINGLE, 2.3))
    findings = {}
    for c in ["webgpu", "webgpu-native", "webgpu-capture"]:
        g = t[(t["backend"] == c) & t["cpu_node_ms"].notna() & t["dispatches"].notna()]
        if len(g) < 3:
            continue
        ax.scatter(g["dispatches"], g["cpu_node_ms"], color=style.color(c), marker=style.marker(c), label=style.label(c), s=14)
        fit = stats.linfit(g["dispatches"], g["cpu_node_ms"])
        xs = np.linspace(0, g["dispatches"].max(), 50)
        ax.plot(xs, fit["intercept"] + fit["slope"] * xs, color=style.color(c), lw=0.7, ls=":")
        findings[c] = {"us_per_dispatch": fit["slope"] * 1000, "r2": fit["r2"], "intercept_ms": fit["intercept"],
                       "per_model_us_per_dispatch": dict(zip(g["model"], (g["cpu_node_ms"] / g["dispatches"] * 1000).round(2)))}
    ax.set_xlabel("GPU dispatches per inference")
    ax.set_ylabel("CPU-side kernel time (ms)")
    ax.legend(fontsize=6)
    return {"files": style.save(fig, out, "fig07_dispatch_overhead"), "dispatch_cost": findings}


def op_speedup(study, out):
    st, runs, k = _profile(study)
    if st is None or k.empty:
        return {"skipped": "no kernels"}
    k = k.assign(op=k["op"].replace(OP_ALIASES))
    per_run = lambda df: (df.groupby(["model", "trial", "run", "op"])["dur_us"].sum()
                          .groupby(["model", "op"]).median())
    wasm = per_run(k[(k["backend"] == "wasm-8t") & (k["source"] == "ort_trace") & (k["phase"] == "profile_counters")])
    gpu = per_run(k[(k["backend"] == "webgpu") & (k["source"] == "webgpu_timestamp")])
    both = pd.concat({"wasm_us": wasm, "gpu_us": gpu}, axis=1).dropna()
    both = both[(both["gpu_us"] > 0)]
    both["speedup"] = both["wasm_us"] / both["gpu_us"]
    agg = both.reset_index().groupby("op").agg(
        models=("model", "nunique"), wasm_total_us=("wasm_us", "sum"), gpu_total_us=("gpu_us", "sum"),
        geomean_speedup=("speedup", lambda s: float(np.exp(np.log(s).mean()))))
    agg["wasm_share"] = agg["wasm_total_us"] / agg["wasm_total_us"].sum()
    top = agg.sort_values("wasm_total_us", ascending=False).head(12).sort_values("geomean_speedup")

    # Kernel-level vs end-to-end speedup: how much of WebGPU's compute advantage survives.
    lat = study.steady("latency")
    rows = []
    for m in sorted(both.index.get_level_values(0).unique()):
        kb = both.loc[m]
        if (m, "wasm-8t") in lat.index and (m, "webgpu") in lat.index:
            e2e = np.median(lat[(m, "wasm-8t")]) / np.median(lat[(m, "webgpu")])
            kern = kb["wasm_us"].sum() / kb["gpu_us"].sum()
            rows.append({"model": m, "kernel_speedup": kern, "e2e_speedup": e2e, "retained": e2e / kern})
    ret = pd.DataFrame(rows)

    fig, (a, b) = plt.subplots(1, 2, figsize=(style.DOUBLE, 2.5), gridspec_kw={"width_ratios": [1, 1]})
    a.barh(range(len(top)), top["geomean_speedup"], color="#0072B2")
    a.set_yticks(range(len(top)), [f"{o} ({int(n)})" for o, n in zip(top.index, top["models"])])
    a.set_xscale("log"); a.axvline(1, color="k", lw=0.6, ls="--")
    a.set_xlabel("GPU kernel speedup vs WASM 8T op time (geo-mean, log)")
    a.set_title("(a) Per-operator speedup (models)")
    if len(ret):
        feats = study.features.set_index("model")
        ret = ret.assign(macs=ret["model"].map(feats["macs"])).sort_values("macs")
        x = np.arange(len(ret))
        b.bar(x - 0.2, ret["kernel_speedup"], 0.4, color="#56B4E9", label="kernels only")
        b.bar(x + 0.2, ret["e2e_speedup"], 0.4, color="#0072B2", label="end-to-end")
        b.set_yscale("log"); b.axhline(1, color="k", lw=0.6, ls="--")
        b.set_xticks(x, [style.short(m) for m in ret["model"]], rotation=45, ha="right")
        b.set_ylabel("speedup vs WASM 8T (log)")
        b.set_title("(b) Compute advantage vs what the user sees")
        b.legend(fontsize=6)
    fig.tight_layout()
    agg.to_csv(out / "fig08_op_speedup_table.csv")
    ret.to_csv(out / "fig08_retained_speedup.csv", index=False)
    return {"files": style.save(fig, out, "fig08_op_speedup"),
            "op_geomean_speedup": top["geomean_speedup"].round(3).to_dict(),
            "retained_speedup": ret.round(3).to_dict("records") if len(ret) else []}


def runtime_variants(study, out):
    lat = study.steady("latency")
    feats = study.features.set_index("model")
    rows = []
    for m in sorted({m for m, _ in lat.index}):
        if (m, "webgpu") not in lat.index:
            continue
        for c in ["webgpu-native", "webgpu-gpuio", "webgpu-capture"]:
            if (m, c) in lat.index:
                r, lo, hi = stats.ratio_of_medians_ci(lat[(m, c)], lat[(m, "webgpu")])
                rows.append({"model": m, "variant": c, "rel_latency": r, "lo": lo, "hi": hi, "macs": feats.loc[m, "macs"]})
    t = pd.DataFrame(rows)
    if t.empty:
        return {"skipped": "no variants"}
    # Dispatch counts per inference (latency-stage verification run), native vs JSEP.
    s = study["latency"].sessions
    disp = s[s["status"] == "ok"].groupby(["model", "backend"])["webgpu_dispatches"].median()
    models = sorted(t["model"].unique(), key=lambda m: feats.loc[m, "macs"])
    fig, ax = plt.subplots(figsize=(style.DOUBLE / 1.6, 2.4))
    width = 0.26
    for k, c in enumerate(["webgpu-native", "webgpu-gpuio", "webgpu-capture"]):
        g = t[t["variant"] == c].set_index("model").reindex(models)
        x = np.arange(len(models)) + (k - 1) * width
        ax.bar(x, g["rel_latency"], width, color=style.color(c), label=style.label(c),
               yerr=[g["rel_latency"] - g["lo"], g["hi"] - g["rel_latency"]], error_kw={"lw": 0.6, "capsize": 1})
    ax.axhline(1, color="k", lw=0.6, ls="--")
    ax.set_xticks(range(len(models)), [style.short(m) for m in models], rotation=45, ha="right")
    ax.set_ylabel("latency relative to WebGPU (JSEP)")
    ax.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    t.to_csv(out / "fig09_variants_table.csv", index=False)
    cap = t[t["variant"] == "webgpu-capture"]
    rho = stats.spearman(np.log(cap["macs"]), cap["rel_latency"]) if len(cap) > 2 else None
    native_disp = {m: float(disp.get((m, "webgpu-native"), np.nan) / disp.get((m, "webgpu"), np.nan))
                   for m in models if (m, "webgpu") in disp.index}
    return {"files": style.save(fig, out, "fig09_runtime_variants"),
            "variants_median_rel_latency": t.groupby("variant")["rel_latency"].median().to_dict(),
            "capture_gain_by_model": dict(zip(cap["model"], (1 - cap["rel_latency"]).round(3))),
            "spearman_capture_rel_vs_macs": rho,
            "native_over_jsep_dispatch_ratio": native_disp}


def readback(study, out):
    lat = study.steady("latency")
    rb = study.steady("readback") if study.has("readback") else None
    rows = []
    if rb is not None:
        for (m, c), x in rb.items():
            if (m, c) in lat.index:
                r, lo, hi = stats.ratio_of_medians_ci(lat[(m, c)], x)
                rows.append({"model": m, "backend": c, "all_p50": float(np.median(lat[(m, c)])),
                             "primary_p50": float(np.median(x)), "all_over_primary": r, "lo": lo, "hi": hi})
    t = pd.DataFrame(rows)
    # GPU-I/O runs split submit vs readback-of-primary (includes waiting for the GPU to finish).
    r = study["latency"].ok_runs("measure")
    split = r[r["backend"] == "webgpu-gpuio"].groupby("model")[["submit_ms", "readback_ms", "dur_ms"]].median() \
        if "submit_ms" in r else pd.DataFrame()
    fig, (a, b) = plt.subplots(1, 2, figsize=(style.DOUBLE, 2.3))
    if len(t):
        models = sorted(t["model"].unique())
        configs = [c for c in style.CONFIG_ORDER if c in set(t["backend"])]
        w = 0.8 / len(configs)
        for k, c in enumerate(configs):
            g = t[t["backend"] == c].set_index("model").reindex(models)
            a.bar(np.arange(len(models)) + (k - (len(configs) - 1) / 2) * w, g["all_over_primary"], w,
                  color=style.color(c), label=style.label(c))
        a.axhline(1, color="k", lw=0.6, ls="--")
        a.set_xticks(range(len(models)), [style.short(m) for m in models])
        a.set_ylabel("latency: all outputs / primary only")
        a.set_title("(a) Cost of fetching every output")
        a.legend(fontsize=6)
    if len(split):
        feats = study.features.set_index("model")
        split = split.assign(macs=split.index.map(feats["macs"])).sort_values("macs")
        x = np.arange(len(split))
        b.bar(x, split["submit_ms"], color="#0072B2", label="run() returns (CPU encode + submit)")
        b.bar(x, split["readback_ms"], bottom=split["submit_ms"], color="#E69F00", label="wait for GPU + read primary output")
        b.set_yscale("log")
        b.set_xticks(x, [style.short(m) for m in split.index], rotation=45, ha="right")
        b.set_ylabel("ms (log)")
        b.set_title("(b) GPU-resident I/O: submit vs completion")
        b.legend(fontsize=6)
    fig.tight_layout()
    t.to_csv(out / "fig10_readback_table.csv", index=False)
    split.to_csv(out / "fig10_gpuio_split.csv")
    return {"files": style.save(fig, out, "fig10_readback"),
            "all_over_primary": t[["model", "backend", "all_over_primary", "lo", "hi"]].round(3).to_dict("records") if len(t) else [],
            "gpuio_readback_share": (split["readback_ms"] / split["dur_ms"]).round(3).to_dict() if len(split) else {}}
