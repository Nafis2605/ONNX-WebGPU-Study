"""
Group 3: token generation timing (llm-mode runs).

The GPT-2 export has no KV-cache inputs, so every token is a full forward pass over the fixed
128-token window. "Prefill" and "decode" are the same computation here; results are labelled
no-KV-cache and aren't comparable with cached decoding.

Per (model, backend), pooled over generations and trials:
    ttft_ms                  first step of each generation (prompt forward + argmax), median
    step_ms_p50 / p95        steps after the first (visible inter-token gaps), median and p95
    decode_tokens_per_s      1000 / mean of steps after the first
    generation_ms            wall time per generation, median

Usage:
    python analysis/llm.py benchmark_results/<llm run>
"""
import argparse
from pathlib import Path

import pandas as pd

KEY = ["trial", "model", "backend"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    args = ap.parse_args()
    d = Path(args.dir)
    sessions = pd.read_csv(d / "sessions.csv")
    ok = sessions[(sessions["status"] == "ok") & (sessions["mode"] == "llm")][KEY]
    runs = pd.read_csv(d / "runs.csv").merge(ok, on=KEY)
    gen = runs[runs["phase"] == "gen"]
    first = gen[gen["step"] == 0]
    rest = gen[gen["step"] > 0]
    per_gen = gen.groupby(KEY + ["generation"])["dur_ms"].sum()
    out = pd.DataFrame({
        "generations": first.groupby(["model", "backend"]).size(),
        "ttft_ms": first.groupby(["model", "backend"])["dur_ms"].median(),
        "step_ms_p50": rest.groupby(["model", "backend"])["dur_ms"].median(),
        "step_ms_p95": rest.groupby(["model", "backend"])["dur_ms"].quantile(0.95),
        "decode_tokens_per_s": 1000.0 / rest.groupby(["model", "backend"])["dur_ms"].mean(),
        "generation_ms": per_gen.groupby(level=["model", "backend"]).median(),
    }).reset_index()
    out["kv_cache"] = False
    out.to_csv(d / "llm_summary.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
