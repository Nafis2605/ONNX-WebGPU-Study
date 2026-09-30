"""
Merge a supplementary run into a stage's main run.

Every (trial, model, config) experiment in the supplement replaces the main run's experiment with
the same key (e.g. WebGL re-runs on a fixed-shape model copy replace the failed originals); all
other experiments are carried over untouched. Per-experiment CSVs written by run_benchmark.py
(telemetry, correctness) are filtered the same way; sessions/runs/kernels/... are rebuilt from the
merged experiments.jsonl by export_csvs. Logs stay in the source folders (listed in env.json).

    python -m benchkit.merge <main dir> <supplement dir> <out dir>
"""
import json
import sys
from pathlib import Path

import pandas as pd

from .results import export_csvs, load_experiments

KEY = ["trial", "model", "backend"]   # "backend" = config label in every CSV
PER_EXPERIMENT_CSVS = ["correctness.csv", "gpu_samples.csv", "gpu_util_samples.csv", "gpu_pcie_samples.csv",
                       "gpu_marks.csv", "chrome_memory_samples.csv", "system_cpu_samples.csv"]


def exp_key(e):
    return (int(e["trial"]), e["model"], e.get("config_label", e["backend"]))


def merge(main, supp, out):
    main, supp, out = Path(main), Path(supp), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    new = load_experiments(supp)
    replaced = {exp_key(e) for e in new}
    kept = [e for e in load_experiments(main) if exp_key(e) not in replaced]
    with open(out / "experiments.jsonl", "w", encoding="utf-8") as f:
        for e in kept + new:
            f.write(json.dumps(e, default=str) + "\n")

    for name in PER_EXPERIMENT_CSVS:
        parts = []
        for src, drop in ((main, True), (supp, False)):
            if (src / name).exists():
                d = pd.read_csv(src / name, low_memory=False)
                if drop and len(d):
                    keys = pd.Series(list(zip(d["trial"].astype(int), d["model"], d["backend"])), index=d.index)
                    d = d[~keys.isin(replaced)]
                parts.append(d)
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(out / name, index=False)

    env = json.loads((main / "env.json").read_text(encoding="utf-8"))
    env["merged_from"] = {"main": str(main.resolve()), "supplement": str(supp.resolve()),
                          "replaced_experiments": sorted(map(list, replaced)),
                          "supplement_args": json.loads((supp / "env.json").read_text(encoding="utf-8")).get("args")}
    (out / "env.json").write_text(json.dumps(env, indent=1, default=str), encoding="utf-8")
    n_sessions, n_runs = export_csvs(out)
    print(f"merged {len(kept)} kept + {len(new)} supplementary experiments "
          f"({len(replaced)} keys replaced) -> {out} ({n_sessions} sessions, {n_runs} runs)")
    return out


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    merge(*sys.argv[1:])
