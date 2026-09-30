"""§2 When does WebGPU win? Speedup vs compute, distributions, tails, batch scaling."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import stats, style


def _lat(study):
    return study.steady("latency")


def speedup_table(study, target="webgpu"):
    lat = _lat(study)
    feats = study.features.set_index("model")
    rows = []
    for model in sorted({m for m, _ in lat.index}):
        if (model, target) not in lat.index:
            continue
        for base in ["wasm-1t", "wasm-4t", "wasm-8t", "webgl"]:
            if (model, base) not in lat.index:
                continue
            r, lo, hi = stats.ratio_of_medians_ci(lat[(model, base)], lat[(model, target)])
            rows.append({"model": model, "baseline": base, "speedup": r, "lo": lo, "hi": hi,
                         "macs": feats.loc[model, "macs"], "weight_bytes": feats.loc[model, "weight_bytes"],
                         "target_p50_ms": float(np.median(lat[(model, target)])),
                         "baseline_p50_ms": float(np.median(lat[(model, base)]))})
    return pd.DataFrame(rows)


def speedup_vs_compute(study, out):
    t = speedup_table(study)
    fig, ax = plt.subplots(figsize=(style.SINGLE, 2.5))
    findings = {}
    for base in ["wasm-1t", "wasm-4t", "wasm-8t", "webgl"]:
        g = t[t["baseline"] == base].sort_values("macs")
        if g.empty:
            continue
        x = g["macs"] / 1e9
        ax.errorbar(x, g["speedup"], yerr=[g["speedup"] - g["lo"], g["hi"] - g["speedup"]], fmt=style.marker(base),
                    color=style.color(base), label=f"vs {style.label(base)}", capsize=1.5, lw=0.8, ms=4)
        rho, p = stats.spearman(np.log(g["macs"]), np.log(g["speedup"]))
        fit = stats.linfit(np.log10(g["macs"]), np.log10(g["speedup"]))
        # Crossover (speedup = 1) of the log-log fit, reported only if it lies inside the tested range.
        cross = 10 ** (-fit["intercept"] / fit["slope"]) if fit["slope"] != 0 else None
        in_range = cross is not None and g["macs"].min() <= cross <= g["macs"].max()
        findings[base] = {
            "speedup_min": float(g["speedup"].min()), "speedup_max": float(g["speedup"].max()),
            "speedup_median": float(g["speedup"].median()),
            "models_where_webgpu_slower": g.loc[g["hi"] < 1, "model"].tolist(),
            "spearman_rho_vs_macs": rho, "spearman_p": p, "loglog_fit": fit,
            "crossover_gmacs": float(cross / 1e9) if in_range else None,
        }
    ax.axhline(1, color="k", lw=0.6, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Model compute (GMACs, log)")
    ax.set_ylabel("WebGPU speedup (×, log)")
    ax.legend(ncol=2, loc="upper left")
    t.to_csv(out / "fig02_speedup_table.csv", index=False)
    return {"files": style.save(fig, out, "fig02_speedup_vs_compute"), "speedup": findings}


def latency_ecdf(study, out):
    lat = _lat(study)
    feats = study.features.set_index("model")
    models = sorted({m for m, _ in lat.index}, key=lambda m: feats.loc[m, "macs"])
    configs = ["wasm-8t", "webgl", "webgpu", "webgpu-native", "webgpu-capture"]
    n = len(models)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(style.DOUBLE, 1.45 * rows), squeeze=False)
    for ax, m in zip(axes.flat, models):
        for c in configs:
            if (m, c) in lat.index:
                x = np.sort(lat[(m, c)])
                ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=style.color(c), label=style.label(c), lw=1)
        ax.set_xscale("log")
        ax.set_title(style.short(m))
        ax.set_ylim(0, 1)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("ECDF")
    for ax in axes[-1, :]:
        ax.set_xlabel("latency (ms, log)")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(configs), loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return {"files": style.save(fig, out, "fig03_latency_ecdf")}


def tail_ratio(study, out):
    lat = _lat(study)
    rows = []
    for (m, c), x in lat.items():
        rows.append({"model": m, "backend": c, "p50": np.percentile(x, 50), "p99": np.percentile(x, 99),
                     "tail_ratio": np.percentile(x, 99) / np.percentile(x, 50), "cov": np.std(x) / np.mean(x), "n": len(x)})
    t = pd.DataFrame(rows)
    feats = study.features.set_index("model")
    models = sorted(t["model"].unique(), key=lambda m: feats.loc[m, "macs"])
    fig, ax = plt.subplots(figsize=(style.DOUBLE / 2, 2.4))
    configs = [c for c in style.CONFIG_ORDER if c in set(t["backend"])]
    for k, c in enumerate(configs):
        g = t[t["backend"] == c].set_index("model")
        xs = [models.index(m) + (k - len(configs) / 2) * 0.08 for m in g.index]
        ax.scatter(xs, g["tail_ratio"], color=style.color(c), marker=style.marker(c), label=style.label(c), s=12)
    ax.set_xticks(range(len(models)), [style.short(m) for m in models], rotation=60, ha="right")
    ax.set_ylabel("p99 / p50 latency")
    ax.axhline(1, color="k", lw=0.5)
    ax.legend(ncol=2, fontsize=6)
    t.to_csv(out / "fig04_tail_table.csv", index=False)
    by_cfg = t.groupby("backend")["tail_ratio"].median().to_dict()
    return {"files": style.save(fig, out, "fig04_tail_ratio"), "median_tail_ratio_by_config": by_cfg,
            "worst": t.sort_values("tail_ratio", ascending=False).head(5)[["model", "backend", "tail_ratio"]].to_dict("records")}


def batch_scaling(study, out):
    st = study["batch"]
    if st is None:
        return {"skipped": "no batch stage"}
    r = st.ok_runs("batch")
    r = r.assign(per_sample=r["dur_ms"] / r["batch"])
    t = r.groupby(["model", "backend", "batch"])["per_sample"].median().reset_index()
    models = sorted(t["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(style.SINGLE * len(models) / 1.2, 2.2), squeeze=False)
    findings = {}
    for ax, m in zip(axes[0], models):
        g = t[t["model"] == m]
        for c in [c for c in style.CONFIG_ORDER if c in set(g["backend"])]:
            gc = g[g["backend"] == c].sort_values("batch")
            ax.plot(gc["batch"], gc["per_sample"], marker=style.marker(c), color=style.color(c), label=style.label(c))
            b1 = gc[gc["batch"] == gc["batch"].min()]["per_sample"].iloc[0]
            bmax = gc[gc["batch"] == gc["batch"].max()]["per_sample"].iloc[0]
            findings[f"{m}|{c}"] = {"per_sample_b1_ms": float(b1), "per_sample_bmax_ms": float(bmax),
                                    "bmax": int(gc["batch"].max()), "amortisation": float(b1 / bmax)}
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_title(style.short(m)); ax.set_xlabel("batch size"); ax.set_ylabel("latency per sample (ms)")
    axes[0][0].legend(fontsize=6)
    t.to_csv(out / "fig05_batch_table.csv", index=False)
    return {"files": style.save(fig, out, "fig05_batch_scaling"), "batch": findings}
