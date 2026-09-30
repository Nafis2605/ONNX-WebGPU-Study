"""§4 At what cost? Startup (session creation + first inference), break-even, shape recompilation."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import style

BASE = "wasm-8t"


def init_table(study):
    st = study["latency"]
    s = st.sessions[st.sessions["status"] == "ok"]
    r = st.ok_runs()
    steady = r[r["phase"] == "measure"].groupby(["trial", "model", "backend"])["dur_ms"].median().rename("steady_ms")
    t = s.set_index(["trial", "model", "backend"])[["fetch_ms", "session_create_ms", "first_run_ms"]].join(steady).reset_index()
    t["first_run_overhead_ms"] = t["first_run_ms"] - t["steady_ms"]
    t["startup_ms"] = t["session_create_ms"] + t["first_run_ms"]
    return t.groupby(["model", "backend"]).median(numeric_only=True).drop(columns="trial").reset_index()


def startup_costs(study, out):
    t = init_table(study)
    feats = study.features.set_index("model")
    models = sorted(t["model"].unique(), key=lambda m: feats.loc[m, "macs"])
    configs = [c for c in ["wasm-8t", "webgl", "webgpu", "webgpu-native", "webgpu-capture"] if c in set(t["backend"])]
    fig, ax = plt.subplots(figsize=(style.DOUBLE / 1.5, 2.4))
    w = 0.8 / len(configs)
    for k, c in enumerate(configs):
        g = t[t["backend"] == c].set_index("model").reindex(models)
        x = np.arange(len(models)) + (k - (len(configs) - 1) / 2) * w
        ax.bar(x, g["session_create_ms"], w, color=style.color(c), label=style.label(c))
        ax.bar(x, g["first_run_overhead_ms"].clip(lower=0), w, bottom=g["session_create_ms"], color=style.color(c),
               alpha=0.45, hatch="////", edgecolor="white", lw=0)
    ax.set_yscale("log")
    ax.set_xticks(range(len(models)), [style.short(m) for m in models], rotation=45, ha="right")
    ax.set_ylabel("ms (log): session create (solid) + first-run overhead (hatched)")
    ax.legend(fontsize=6, ncol=3)
    t.to_csv(out / "fig11_startup_table.csv", index=False)
    j = t[t["backend"] == "webgpu"].set_index("model")
    w8 = t[t["backend"] == BASE].set_index("model")
    return {"files": style.save(fig, out, "fig11_startup_costs"),
            "webgpu_first_run_overhead_ms": j["first_run_overhead_ms"].round(1).to_dict(),
            "webgpu_first_run_over_steady": (j["first_run_ms"] / j["steady_ms"]).round(1).to_dict(),
            "wasm8t_first_run_over_steady": (w8["first_run_ms"] / w8["steady_ms"]).round(2).to_dict()}


def break_even(study, out):
    """Inferences needed before WebGPU's total time (startup + n * steady) beats WASM 8T's."""
    t = init_table(study).set_index(["model", "backend"])
    feats = study.features.set_index("model")
    rows = []
    for m in sorted({m for m, _ in t.index}, key=lambda m: feats.loc[m, "macs"]):
        if (m, "webgpu") not in t.index or (m, BASE) not in t.index:
            continue
        g, c = t.loc[(m, "webgpu")], t.loc[(m, BASE)]
        # Startup excludes the first run's steady-state share so it isn't counted twice.
        dg = g["session_create_ms"] + g["first_run_overhead_ms"]
        dc = c["session_create_ms"] + c["first_run_overhead_ms"]
        gain = c["steady_ms"] - g["steady_ms"]
        extra = dg - dc
        if gain <= 0:
            n = np.inf if extra >= 0 else 0.0
        else:
            n = max(0.0, extra / gain)
        rows.append({"model": m, "startup_extra_ms": extra, "per_inference_gain_ms": gain, "break_even_n": n,
                     "macs": feats.loc[m, "macs"]})
    b = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(style.SINGLE, 2.2))
    finite = b["break_even_n"].replace(np.inf, np.nan)
    ax.bar(range(len(b)), finite.fillna(0).clip(lower=0.5), color="#0072B2")
    for i, n in enumerate(b["break_even_n"]):
        ax.text(i, max(0.6, (0 if np.isinf(n) else n)) * 1.15, "never" if np.isinf(n) else (f"{n:.1f}" if n < 10 else f"{n:.0f}"), ha="center", fontsize=6)
    ax.set_yscale("log")
    ax.set_xticks(range(len(b)), [style.short(m) for m in b["model"]], rotation=45, ha="right")
    ax.set_ylabel(f"inferences until WebGPU\nbeats {style.label(BASE)} (log)")
    b.to_csv(out / "fig12_break_even.csv", index=False)
    return {"files": style.save(fig, out, "fig12_break_even"),
            "break_even_n": {m: (None if np.isinf(n) else round(float(n), 1)) for m, n in zip(b["model"], b["break_even_n"])}}


def shape_penalty(study, out):
    st = study["shapes"]
    if st is None:
        return {"skipped": "no shapes stage"}
    r = st.ok_runs("shape").copy()
    r["first_seen"] = r["first_seen"].astype(str).str.lower() == "true"
    r["kind"] = np.where(r["first_seen"], "first_seen", np.where(r["block_pos"] == 0, "revisit", "steady"))
    t = r.groupby(["model", "backend", "shape_value", "kind"])["dur_ms"].median().unstack("kind").reset_index()
    t["first_seen_x"] = t["first_seen"] / t["steady"]
    t["revisit_x"] = t["revisit"] / t["steady"]
    models = sorted(t["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(style.SINGLE * len(models) / 1.2, 2.1), squeeze=False)
    for ax, m in zip(axes[0], models):
        g = t[t["model"] == m]
        for c in [c for c in style.CONFIG_ORDER if c in set(g["backend"])]:
            gc = g[(g["backend"] == c) & g["first_seen_x"].notna()].sort_values("shape_value")
            ax.plot(gc["shape_value"], gc["first_seen_x"], marker=style.marker(c), color=style.color(c), label=style.label(c))
        ax.axhline(1, color="k", lw=0.6, ls="--")
        ax.set_xscale("log", base=2)
        ax.set_title(style.short(m)); ax.set_xlabel("new batch size"); ax.set_ylabel("first call / steady")
    axes[0][0].legend(fontsize=6)
    t.to_csv(out / "fig13_shape_penalty.csv", index=False)
    return {"files": style.save(fig, out, "fig13_shape_penalty"),
            "first_seen_x": t.dropna(subset=["first_seen_x"])[["model", "backend", "shape_value", "first_seen_x", "revisit_x"]]
            .round(3).to_dict("records")}
