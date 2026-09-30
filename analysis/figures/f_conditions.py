"""§5 Under which conditions? Arrival rate (DVFS idle penalty), load, CPU cost, energy, main thread, LLM."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import stats, style

KEY = ["trial", "model", "backend"]


def idle_penalty(study, out, slo_ms=100.0):
    st = study["load"]
    if st is None:
        return {"skipped": "no load stage"}
    req = st.csv("load_requests.csv").merge(st.ok_keys(), on=KEY)
    samples = st.csv("gpu_samples.csv")
    lat = study.steady("latency")
    window_s = st.env["args"]["load_s"]
    rows = []
    for (trial, m, c, rate), g in req.groupby(KEY + ["rate"]):
        done = g[g["service_ms"].notna()]
        w0 = g["t_arrival_epoch_ms"].min()
        w1 = w0 + window_s * 1000
        smp = samples[(samples["trial"] == trial) & (samples["model"] == m) & (samples["backend"] == c)
                      & (samples["t_epoch_ms"] >= w0) & (samples["t_epoch_ms"] <= w1)] if len(samples) else pd.DataFrame()
        met = (~g["rejected"]) & (~g["unfinished"]) & (g["latency_ms"] <= slo_ms)
        rows.append({"trial": trial, "model": m, "backend": c, "rate": rate,
                     "service_p50_ms": done["service_ms"].median(), "latency_p95_ms": done["latency_ms"].quantile(0.95),
                     "goodput_rps": met.sum() / window_s, "offered_rps": len(g) / window_s,
                     "miss_rate": 1 - met.mean(),
                     "sm_clock_mhz": smp["sm_clock_mhz"].median() if len(smp) else np.nan,
                     "closed_loop_p50_ms": float(np.median(lat[(m, c)])) if (m, c) in lat.index else np.nan})
    t = pd.DataFrame(rows).groupby(["model", "backend", "rate"]).median(numeric_only=True).drop(columns="trial").reset_index()
    t["service_over_closed_loop"] = t["service_p50_ms"] / t["closed_loop_p50_ms"]
    models = sorted(t["model"].unique())
    fig, axes = plt.subplots(2, len(models), figsize=(style.DOUBLE, 3.6), squeeze=False, sharex=True)
    for j, m in enumerate(models):
        g = t[t["model"] == m]
        for c in [c for c in style.CONFIG_ORDER if c in set(g["backend"])]:
            gc = g[g["backend"] == c].sort_values("rate")
            axes[0][j].plot(gc["rate"], gc["service_over_closed_loop"], marker=style.marker(c), color=style.color(c), label=style.label(c))
            if gc["sm_clock_mhz"].notna().any() and c != "wasm-8t":
                axes[1][j].plot(gc["rate"], gc["sm_clock_mhz"], marker=style.marker(c), color=style.color(c), label=style.label(c))
        axes[0][j].axhline(1, color="k", lw=0.6, ls="--")
        axes[0][j].set_title(style.short(m))
        axes[0][j].set_xscale("log")
        axes[1][j].set_xlabel("arrival rate (req/s, log)")
    axes[0][0].set_ylabel("service time /\nclosed-loop p50")
    axes[1][0].set_ylabel("GPU SM clock (MHz)")
    handles = {}
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    fig.legend(handles.values(), handles.keys(), ncol=len(handles), loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    t.to_csv(out / "fig14_idle_penalty.csv", index=False)
    low = t[t["rate"] == t["rate"].min()].set_index(["model", "backend"])["service_over_closed_loop"]
    return {"files": style.save(fig, out, "fig14_idle_penalty"),
            "service_over_closed_loop_at_lowest_rate": {f"{m}|{c}": round(float(v), 3) for (m, c), v in low.items()},
            "sm_clock_by_rate_webgpu": t[t["backend"] == "webgpu"].groupby("rate")["sm_clock_mhz"].median().round(0).to_dict()}


def goodput(study, out, slo_ms=100.0):
    path = out / "fig14_idle_penalty.csv"
    if not path.exists():
        return {"skipped": "needs idle_penalty"}
    t = pd.read_csv(path)
    models = sorted(t["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(style.DOUBLE, 2.0), squeeze=False)
    msr = {}
    for ax, m in zip(axes[0], models):
        g = t[t["model"] == m]
        for c in [c for c in style.CONFIG_ORDER if c in set(g["backend"])]:
            gc = g[g["backend"] == c].sort_values("rate")
            ax.plot(gc["offered_rps"], gc["goodput_rps"], marker=style.marker(c), color=style.color(c), label=style.label(c))
            ok = gc[gc["miss_rate"] <= 0.01]
            msr[f"{m}|{c}"] = float(ok["rate"].max()) if len(ok) else None
        lim = g["offered_rps"].max() * 1.1
        ax.plot([0, lim], [0, lim], color="k", lw=0.5, ls=":")
        ax.set_title(style.short(m)); ax.set_xlabel("offered load (req/s)")
    axes[0][0].set_ylabel(f"goodput (req/s within {slo_ms:.0f} ms)")
    axes[0][0].legend(fontsize=6)
    return {"files": style.save(fig, out, "fig15_goodput"), "max_sustainable_rate_rps": msr}


def cpu_cost(study, out):
    st = study["latency"]
    cm = st.csv("chrome_memory_samples.csv")
    if cm.empty or "cpu_user_s" not in cm:
        return {"skipped": "no CPU-time telemetry"}
    r = st.ok_runs("measure")
    rows = []
    for (trial, m, c), g in r.groupby(KEY):
        w0, w1 = g["t_start_epoch_ms"].min(), (g["t_start_epoch_ms"] + g["dur_ms"]).max()
        c_ = cm[(cm["trial"] == trial) & (cm["model"] == m) & (cm["backend"] == c)]
        for ptype, pg in c_.groupby("process_type"):
            pg = pg.sort_values("t_epoch_ms")
            inw = pg[(pg["t_epoch_ms"] >= w0) & (pg["t_epoch_ms"] <= w1)]
            if len(inw) < 2:
                continue
            cpu = inw["cpu_user_s"] + inw["cpu_system_s"]
            dt = (inw["t_epoch_ms"].iloc[-1] - inw["t_epoch_ms"].iloc[0]) / 1000
            n = ((g["t_start_epoch_ms"] >= inw["t_epoch_ms"].iloc[0]) & (g["t_start_epoch_ms"] <= inw["t_epoch_ms"].iloc[-1])).sum()
            if n == 0:
                continue
            rows.append({"trial": trial, "model": m, "backend": c, "process": ptype,
                         "cpu_ms_per_inference": (cpu.iloc[-1] - cpu.iloc[0]) * 1000 / n,
                         "cores_busy": (cpu.iloc[-1] - cpu.iloc[0]) / dt if dt > 0 else np.nan})
    t = pd.DataFrame(rows)
    if t.empty:
        return {"skipped": "no windows"}
    agg = t.groupby(["model", "backend", "process"]).median(numeric_only=True).drop(columns="trial").reset_index()
    agg.to_csv(out / "fig16_cpu_cost.csv", index=False)
    feats = study.features.set_index("model")
    models = sorted(agg["model"].unique(), key=lambda m: feats.loc[m, "macs"])
    configs = [c for c in style.CONFIG_ORDER if c in set(agg["backend"])]
    fig, ax = plt.subplots(figsize=(style.DOUBLE, 2.4))
    w = 0.85 / len(configs)
    for k, c in enumerate(configs):
        x = np.arange(len(models)) + (k - (len(configs) - 1) / 2) * w
        bottom = np.zeros(len(models))
        for ptype, alpha in (("renderer", 1.0), ("gpu-process", 0.45)):
            v = agg[(agg["backend"] == c) & (agg["process"] == ptype)].set_index("model").reindex(models)["cpu_ms_per_inference"].fillna(0)
            ax.bar(x, v, w, bottom=bottom, color=style.color(c), alpha=alpha, label=style.label(c) if ptype == "renderer" else None)
            bottom += v.to_numpy()
    ax.set_yscale("log")
    ax.set_xticks(range(len(models)), [style.short(m) for m in models], rotation=30, ha="right")
    ax.set_ylabel("CPU ms per inference (log)\nrenderer (solid) + GPU process (light)")
    ax.legend(fontsize=6, ncol=4)
    tot = agg[agg["process"].isin(["renderer", "gpu-process"])].groupby(["model", "backend"])["cpu_ms_per_inference"].sum()
    cores = agg[agg["process"] == "renderer"].set_index(["model", "backend"])["cores_busy"]
    return {"files": style.save(fig, out, "fig16_cpu_cost"),
            "cpu_ms_per_inference": {f"{m}|{c}": round(float(v), 2) for (m, c), v in tot.items()},
            "renderer_cores_busy": {f"{m}|{c}": round(float(v), 2) for (m, c), v in cores.items()}}


def energy(study, out):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gpu_telemetry import analyse
    try:
        t = analyse(study["latency"].path)
    except FileNotFoundError:
        return {"skipped": "no telemetry"}
    if t.empty:
        return {"skipped": "no telemetry rows"}
    agg = t.groupby(["model", "backend"]).median(numeric_only=True).reset_index()
    lat = study.steady("latency")
    agg["p50_ms"] = [float(np.median(lat[(m, c)])) if (m, c) in lat.index else np.nan for m, c in zip(agg["model"], agg["backend"])]
    agg.to_csv(out / "fig17_energy.csv", index=False)
    fig, ax = plt.subplots(figsize=(style.SINGLE, 2.4))
    for c in [c for c in style.CONFIG_ORDER if c in set(agg["backend"])]:
        g = agg[agg["backend"] == c]
        ax.scatter(g["p50_ms"], g["energy_per_inference_mj"], color=style.color(c), marker=style.marker(c), label=style.label(c), s=14)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("latency p50 (ms, log)"); ax.set_ylabel("GPU energy per inference (mJ, log)")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    ax.set_title("GPU board energy only (CPU package energy not measurable)", fontsize=7)
    gpu = agg[agg["backend"].str.startswith("web")]
    return {"files": style.save(fig, out, "fig17_energy"),
            "energy_per_inference_mj": {f"{m}|{c}": round(float(v), 1) for m, c, v in zip(gpu["model"], gpu["backend"], gpu["energy_per_inference_mj"])},
            "mean_power_w": {f"{m}|{c}": round(float(v), 1) for m, c, v in zip(agg["model"], agg["backend"], agg["mean_power_w"])},
            "short_windows": int((~t["window_ok"]).sum())}


def main_thread(study, out):
    st = study["responsiveness"]
    if st is None:
        return {"skipped": "no responsiveness stage"}
    ck = st.csv("clicks.csv").merge(st.ok_keys(), on=KEY)
    if ck.empty:
        return {"skipped": "no clicks"}
    ck = ck[ck["phase"] == "inference"]
    t = ck.groupby(["model", "backend"])["input_delay_ms"].agg(p50="median", p95=lambda x: x.quantile(0.95), n="count").reset_index()
    lat = study.steady("latency")
    t["latency_p50_ms"] = [float(np.median(lat[(m, c)])) if (m, c) in lat.index else np.nan for m, c in zip(t["model"], t["backend"])]
    t.to_csv(out / "fig18_main_thread.csv", index=False)
    models = sorted(t["model"].unique())
    configs = [c for c in style.CONFIG_ORDER if c in set(t["backend"])]
    fig, ax = plt.subplots(figsize=(style.DOUBLE / 1.5, 2.3))
    w = 0.85 / len(configs)
    for k, c in enumerate(configs):
        g = t[t["backend"] == c].set_index("model").reindex(models)
        x = np.arange(len(models)) + (k - (len(configs) - 1) / 2) * w
        ax.bar(x, g["p95"], w, color=style.color(c), alpha=0.4)
        ax.bar(x, g["p50"], w, color=style.color(c), label=style.label(c))
    ax.set_yscale("log")
    ax.set_xticks(range(len(models)), [style.short(m) for m in models])
    ax.set_ylabel("input delay during inference (ms, log)\np50 solid, p95 light")
    ax.legend(fontsize=6, ncol=3)
    t["delay_over_latency"] = t["p50"] / t["latency_p50_ms"]
    return {"files": style.save(fig, out, "fig18_main_thread"),
            "input_delay": t[["model", "backend", "p50", "p95", "latency_p50_ms", "delay_over_latency"]].round(3).to_dict("records")}


def llm(study, out):
    st = study["llm"]
    if st is None:
        return {"skipped": "no llm stage"}
    r = st.ok_runs("gen")
    if r.empty:
        return {"skipped": "no generations"}
    t = r.groupby("backend").apply(lambda g: pd.Series({
        "ttft_ms": g.loc[g["step"] == 0, "dur_ms"].median(),
        "step_p50_ms": g.loc[g["step"] > 0, "dur_ms"].median(),
        "step_p95_ms": g.loc[g["step"] > 0, "dur_ms"].quantile(0.95),
        "tokens_per_s": 1000 / g.loc[g["step"] > 0, "dur_ms"].mean()}), include_groups=False).reset_index()
    lat = study.steady("latency")
    t["full_output_forward_p50_ms"] = [float(np.median(lat[("gpt2lmhead_Opset18", c)])) if ("gpt2lmhead_Opset18", c) in lat.index else np.nan
                                       for c in t["backend"]]
    t.to_csv(out / "fig19_llm.csv", index=False)
    configs = [c for c in style.CONFIG_ORDER if c in set(t["backend"])]
    t = t.set_index("backend").reindex(configs)
    fig, ax = plt.subplots(figsize=(style.SINGLE, 2.2))
    x = np.arange(len(configs))
    ax.bar(x - 0.2, t["full_output_forward_p50_ms"], 0.4, color="#BBBBBB", label="forward, all 25 outputs read back")
    ax.bar(x + 0.2, t["step_p50_ms"], 0.4, color=[style.color(c) for c in configs], label="decode step, logits only")
    ax.set_yscale("log")
    ax.set_xticks(x, [style.label(c) for c in configs], rotation=20, ha="right")
    ax.set_ylabel("ms per token / forward (log)")
    ax.legend(fontsize=6)
    ax.set_title("GPT-2 (no KV cache): per-token latency", fontsize=7)
    return {"files": style.save(fig, out, "fig19_llm"), "llm": t.round(2).reset_index().to_dict("records")}
