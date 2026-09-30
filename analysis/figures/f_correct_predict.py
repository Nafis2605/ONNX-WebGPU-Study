"""§6 Is it correct? §7 Can we predict when WebGPU is the right choice?"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import stats, style

FEATURES = ["log_macs", "log_weight_bytes", "log_nodes"]


def correctness(study, out):
    st = study["correctness"]
    if st is None:
        return {"skipped": "no correctness stage"}
    c = st.csv("correctness.csv")
    if c.empty:
        return {"skipped": "no rows"}
    # One row per (model, config): the primary output (declared logits, else the largest-error output).
    primary = {m: (e.get("logits_outputs") or [None])[0] for m, e in study.registry.items()}
    c["is_primary"] = [o == primary.get(m) for m, o in zip(c["model"], c["output"])]
    rows = []
    for (m, b), g in c.groupby(["model", "backend"]):
        p = g[g["is_primary"]]
        row = (p if len(p) else g.sort_values("norm_max_err", ascending=False)).iloc[0]
        rows.append({"model": m, "backend": b, "output": row["output"], "norm_max_err": row["norm_max_err"],
                     "cosine": row["cosine"], "top1_agreement": row.get("top1_agreement", np.nan),
                     "worst_norm_err_any_output": g["norm_max_err"].max()})
    t = pd.DataFrame(rows)
    t.to_csv(out / "fig20_correctness.csv", index=False)
    feats = study.features.set_index("model")
    models = sorted(t["model"].unique(), key=lambda m: feats.loc[m, "macs"])
    configs = [c for c in style.CONFIG_ORDER if c in set(t["backend"])]
    grid = t.pivot(index="model", columns="backend", values="worst_norm_err_any_output").reindex(index=models, columns=configs)
    fig, ax = plt.subplots(figsize=(style.SINGLE, 3.0))
    im = ax.imshow(np.log10(grid.to_numpy(dtype=float)), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(configs)), [style.label(c) for c in configs], rotation=45, ha="right")
    ax.set_yticks(range(len(models)), [style.short(m) for m in models])
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("log10 max|err| / max|ref| (worst output)")
    ax.set_title("Deviation from native ONNX Runtime (CPU)")
    top1 = t.dropna(subset=["top1_agreement"])
    return {"files": style.save(fig, out, "fig20_correctness"),
            "worst_norm_err_by_config": t.groupby("backend")["worst_norm_err_any_output"].max().to_dict(),
            "min_top1_agreement": float(top1["top1_agreement"].min()) if len(top1) else None,
            "min_cosine": float(t["cosine"].min())}


def _fit(train, test_x):
    X = np.column_stack([np.ones(len(train))] + [train[f] for f in FEATURES])
    y = np.log(train["p50"].to_numpy())
    A = X.T @ X + 1e-3 * np.eye(X.shape[1])
    A[0, 0] -= 1e-3
    w = np.linalg.solve(A, X.T @ y)
    Xt = np.column_stack([np.ones(len(test_x))] + [test_x[f] for f in FEATURES])
    return np.exp(Xt @ w)


def prediction(study, out, configs=("wasm-8t", "webgl", "webgpu")):
    lat = study.steady("latency")
    feats = study.features.set_index("model")
    d = pd.DataFrame([{"model": m, "backend": c, "p50": float(np.median(x))} for (m, c), x in lat.items() if c in configs])
    d = d.join(feats[FEATURES + ["webgl_predicted_feasible"]], on="model")
    s = study["latency"].sessions
    feasible = s.groupby(["model", "backend"])["status"].apply(lambda x: (x == "ok").any())
    preds, picks = [], []
    for held in sorted(feats.index):
        cand = []
        for c in configs:
            train = d[(d["backend"] == c) & (d["model"] != held)]
            if len(train) < len(FEATURES) + 1:
                continue
            p = _fit(train, feats.loc[[held]])[0]
            meas = d[(d["backend"] == c) & (d["model"] == held)]["p50"]
            pf = bool(feats.loc[held, "webgl_predicted_feasible"]) if c == "webgl" else True
            m_ok = bool(feasible.get((held, c), False))
            preds.append({"model": held, "backend": c, "predicted": p, "measured": meas.iloc[0] if len(meas) else np.nan,
                          "predicted_feasible": pf, "measured_feasible": m_ok})
            cand.append((c, p, pf, meas.iloc[0] if len(meas) else np.nan, m_ok))
        ok = [x for x in cand if x[2]]
        real = [x for x in cand if x[4] and not np.isnan(x[3])]
        if not ok or not real:
            continue
        pick = min(ok, key=lambda x: x[1])
        best = min(real, key=lambda x: x[3])
        picks.append({"model": held, "picked": pick[0], "best": best[0], "violation": not pick[4],
                      "regret_ms": (pick[3] - best[3]) if pick[4] else np.nan,
                      "regret_rel": (pick[3] / best[3] - 1) if pick[4] else np.nan})
    p, k = pd.DataFrame(preds), pd.DataFrame(picks)
    p["ape"] = (p["predicted"] - p["measured"]).abs() / p["measured"]
    p.to_csv(out / "fig21_prediction.csv", index=False)
    k.to_csv(out / "fig21_selection.csv", index=False)
    fig, (a, b) = plt.subplots(1, 2, figsize=(style.DOUBLE, 2.4))
    for c in configs:
        g = p[(p["backend"] == c) & p["measured"].notna()]
        a.scatter(g["measured"], g["predicted"], color=style.color(c), marker=style.marker(c), label=style.label(c), s=14)
    lim = [p[["measured", "predicted"]].min().min() * 0.7, p[["measured", "predicted"]].max().max() * 1.4]
    a.plot(lim, lim, color="k", lw=0.6, ls="--")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("measured p50 (ms)"); a.set_ylabel("predicted (held-out model)")
    a.legend(fontsize=6); a.set_title("(a) Leave-one-model-out prediction from static features")
    if len(k):
        x = np.arange(len(k))
        b.bar(x, k["regret_rel"].fillna(0) * 100, color=["#D55E00" if v else ("#009E73" if r == 0 else "#E69F00")
                                                          for v, r in zip(k["violation"], k["regret_rel"].fillna(0))])
        b.set_xticks(x, [style.short(m) for m in k["model"]], rotation=45, ha="right")
        b.set_ylabel("regret vs best backend (%)")
        b.set_title("(b) Backend selection: green = optimal, red = infeasible pick")
    fig.tight_layout()
    return {"files": style.save(fig, out, "fig21_prediction"),
            "median_ape_by_config": p.dropna(subset=["ape"]).groupby("backend")["ape"].median().round(3).to_dict(),
            "selection_optimal_frac": float((k["picked"] == k["best"]).mean()) if len(k) else None,
            "violations": int(k["violation"].sum()) if len(k) else None,
            "median_regret_rel": float(k["regret_rel"].median()) if len(k) else None,
            "webgl_rule_accuracy": float((p[p["backend"] == "webgl"]["predicted_feasible"] == p[p["backend"] == "webgl"]["measured_feasible"]).mean())}
