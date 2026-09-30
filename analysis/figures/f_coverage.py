"""§1 Where can each backend run at all? Model x config support matrix with failure causes."""
import matplotlib.pyplot as plt
import numpy as np

from . import style

ABBREV = {
    "op_unsupported_on_ep": "op",
    "dtype_unsupported_on_ep": "int64",
    "webgl_symbolic_dim_rejected": "dim",
    "cpu_nodes_blocked": "cpu",
    "timeout": "t/o",
    "out_of_memory": "oom",
    "browser_failure": "crash",
}

LEGEND = {"op": "unsupported operator", "int64": "int64 tensor", "dim": "symbolic input dim rejected",
          "cpu": "CPU node blocked", "t/o": "timeout", "oom": "out of memory", "crash": "browser crash",
          "fail": "other failure"}


def support_matrix(study, out):
    s = study["latency"].sessions
    configs = [c for c in style.CONFIG_ORDER if c in set(s["backend"])]
    models = sorted(set(s["model"]), key=lambda m: study.features.set_index("model").loc[m, "macs"])
    status = np.full((len(models), len(configs)), np.nan)
    text = [["" for _ in configs] for _ in models]
    summary = {}
    pinned = []
    for i, m in enumerate(models):
        for j, c in enumerate(configs):
            g = s[(s["model"] == m) & (s["backend"] == c)]
            if g.empty:
                text[i][j] = "—"
                continue
            ok = (g["status"] == "ok").mean()
            status[i, j] = ok
            if ok < 1:
                cat = g.loc[g["status"] != "ok", "failure_category"].mode()
                text[i][j] = ABBREV.get(cat.iloc[0], "fail") if len(cat) else "fail"
            if "model_path" in g and (g["model_path"].dropna() != study.registry[m]["path"]).any():
                # Ran a registry variant (e.g. WebGL's fixed-shape copy: same graph and weights).
                text[i][j] += "\u2020"
                pinned.append((m, c))
    for c in configs:
        g = s[s["backend"] == c].groupby("model")["status"].apply(lambda x: (x == "ok").all())
        summary[c] = {"models_ok": int(g.sum()), "models_total": int(len(g))}

    fig, ax = plt.subplots(figsize=(style.SINGLE, 3.0))
    ax.imshow(status, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(configs)), [style.label(c) for c in configs], rotation=45, ha="right")
    ax.set_yticks(range(len(models)), [style.short(m) for m in models])
    for i in range(len(models)):
        for j in range(len(configs)):
            ax.text(j, i, text[i][j], ha="center", va="center", fontsize=6)
    ax.grid(False)
    ax.set_title("Verified execution (green) and failure cause")
    # Legend lists only the failure causes that appear.
    used = {t.rstrip("\u2020") for row in text for t in row}
    note = " · ".join(f"{a}: {LEGEND[a]}" for a in LEGEND if a in used)
    if pinned:
        note += "\n\u2020 WebGL ran a copy with input dims pinned (same graph and weights)"
    fig.subplots_adjust(bottom=0.34)
    fig.text(0.02, 0.005, note, fontsize=6, va="bottom")
    return {"files": style.save(fig, out, "fig01_support_matrix"), "support": summary,
            "variant_file_cells": [list(x) for x in pinned]}
