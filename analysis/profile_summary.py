"""
Groups 5-7 from profile-mode runs: execution & coordination, data movement, memory.

All inputs are measured: graphics-API counters, GPU timestamp / timer queries, ORT's own trace,
buffer/texture accounting, measureUserAgentSpecificMemory, and Chrome process memory.

Per (model, backend), medians over the profiled runs and trials:

  Execution & coordination (group 5)
    wall_ms                  run wall time in the counters phase (no GPU timestamps active)
    ort_model_run_ms         ORT's own model_run duration (WASM / WebGPU)
    ort_node_cpu_ms          sum of ORT per-node Compute() time on the CPU thread. For WebGPU this
                             is CPU-side kernel preparation (shape logic, pipeline lookup, encoding)
    outside_node_ms          wall_ms - ort_node_cpu_ms: time in the run not spent in node Compute(),
                             i.e. waiting on the GPU, readback and runtime overhead (observed, not modelled)
    gpu_busy_ms              sum of per-kernel GPU durations (profile_gpu phase)
    gpu_span_ms              first kernel start to last kernel end on the GPU timeline (WebGPU)
    dispatches, submits, compute_passes, dispatches_per_submit (mean group size), draws (WebGL)
  Data movement (group 6)
    upload_bytes             WebGPU queue.writeBuffer bytes / WebGL texture upload bytes per run
    readback_bytes           WebGPU mapAsync(READ) bytes / WebGL readPixels bytes per run
    map_read_wait_ms         time from mapAsync() to its promise resolving (includes waiting for GPU work)
    memcpy nodes             CPU<->GPU boundaries inserted by ORT (sessions.csv, WebGPU)
  Memory (group 7)
    gpu_live_mb_after_create logical GPU memory right after session creation (weights, constants)
    gpu_peak_live_mb         peak logical GPU memory during the session
    page_memory_mb_*         measureUserAgentSpecificMemory() (JS heap + WASM memory)
    chrome_*_private_peak_mb peak private bytes of this launch's renderer / GPU process

Also writes top_kernels.csv: the 10 op types with the most time per run, by source.

Usage:
    python analysis/profile_summary.py benchmark_results/<profile run>
"""
import argparse
from pathlib import Path

import pandas as pd

KEY = ["trial", "model", "backend"]


def node_op_types(out_dir, models):
    """{(model, node name): op type} from the ONNX graphs of the given models (via env.json args)."""
    import json
    import onnx
    if len(models) == 0:
        return {}
    args = json.loads((Path(out_dir) / "env.json").read_text())["args"]
    registry = Path(args["models"])
    public = registry.resolve().parents[1]
    entries = {m["name"]: m for m in json.loads(registry.read_text())["models"]}
    out = {}
    for name in models:
        graph = onnx.load(public / entries[name]["path"].lstrip("/"), load_external_data=False).graph
        for n in graph.node:
            out[(name, n.name)] = n.op_type
    return out


def per_run(runs):
    r = runs.copy()
    for col in ("webgpu_dispatches", "webgpu_submits", "webgpu_compute_passes", "webgl_draws",
                "webgpu_write_buffer_bytes", "webgl_tex_upload_bytes", "webgpu_map_read_bytes",
                "webgl_read_pixels_bytes", "webgpu_map_read_wait_ms", "ort_node_time_ms", "ort_model_run_ms",
                "gpu_busy_ms", "gpu_span_ms"):
        if col not in r:
            r[col] = float("nan")
    r["upload_bytes"] = r["webgpu_write_buffer_bytes"].fillna(0) + r["webgl_tex_upload_bytes"].fillna(0)
    r["readback_bytes"] = r["webgpu_map_read_bytes"].fillna(0) + r["webgl_read_pixels_bytes"].fillna(0)
    r["dispatches_per_submit"] = r["webgpu_dispatches"] / r["webgpu_submits"].where(r["webgpu_submits"] > 0)
    r["outside_node_ms"] = r["dur_ms"] - r["ort_node_time_ms"]
    return r


def summarize(out_dir):
    out_dir = Path(out_dir)
    sessions = pd.read_csv(out_dir / "sessions.csv")
    runs = pd.read_csv(out_dir / "runs.csv")
    ok = sessions[(sessions["status"] == "ok") & (sessions["mode"] == "profile")]
    runs = per_run(runs.merge(ok[KEY], on=KEY))

    counters = runs[runs["phase"] == "profile_counters"]
    gpu = runs[runs["phase"] == "profile_gpu"]
    g = ["model", "backend"]
    summary = counters.groupby(g).agg(
        trials=("trial", "nunique"), runs=("dur_ms", "count"),
        wall_ms=("dur_ms", "median"),
        ort_model_run_ms=("ort_model_run_ms", "median"),
        ort_node_cpu_ms=("ort_node_time_ms", "median"),
        outside_node_ms=("outside_node_ms", "median"),
        dispatches=("webgpu_dispatches", "median"),
        submits=("webgpu_submits", "median"),
        compute_passes=("webgpu_compute_passes", "median"),
        dispatches_per_submit=("dispatches_per_submit", "median"),
        draws=("webgl_draws", "median"),
        upload_bytes=("upload_bytes", "median"),
        readback_bytes=("readback_bytes", "median"),
        map_read_wait_ms=("webgpu_map_read_wait_ms", "median"),
    )
    if not gpu.empty:
        summary = summary.join(gpu.groupby(g).agg(
            gpu_phase_wall_ms=("dur_ms", "median"),
            gpu_busy_ms=("gpu_busy_ms", "median"),
            gpu_span_ms=("gpu_span_ms", "median"),
        ))
    mem = ok.groupby(g).agg(
        gpu_live_mb_after_create=("gpu_live_bytes_after_session_create", lambda x: x.median() / 2**20),
        gpu_peak_live_mb=("gpu_peak_live_bytes", lambda x: x.median() / 2**20),
        page_memory_mb_before=("page_memory_before_fetch", lambda x: x.median() / 2**20),
        page_memory_mb_after_create=("page_memory_after_session_create", lambda x: x.median() / 2**20),
        page_memory_mb_after_runs=("page_memory_after_runs", lambda x: x.median() / 2**20),
        memcpy_to_host_nodes=("memcpy_to_host_nodes", "median"),
        memcpy_from_host_nodes=("memcpy_from_host_nodes", "median"),
    )
    summary = summary.join(mem)

    chrome_path = out_dir / "chrome_memory_samples.csv"
    if chrome_path.exists():
        cm = pd.read_csv(chrome_path).merge(ok[KEY], on=KEY)
        peaks = cm.groupby(KEY + ["process_type"])["private_mb"].max().unstack("process_type")
        peaks = peaks.groupby(level=["model", "backend"]).median()
        summary = summary.join(peaks.rename(columns=lambda c: f"chrome_{c}_private_peak_mb"))

    summary = summary.reset_index()

    top = pd.DataFrame()
    kpath = out_dir / "kernels.csv"
    if kpath.exists():
        k = pd.read_csv(kpath).merge(ok[KEY], on=KEY)
        if "op" not in k:
            k["op"] = None
        # WebGL records carry the ORT node name only; look its op type up in the ONNX graph.
        node_ops = node_op_types(out_dir, k.loc[k["op"].isna() & k["node"].notna(), "model"].unique())
        k["op"] = k["op"].fillna(k.apply(lambda r: node_ops.get((r["model"], r["node"])), axis=1))
        k["label"] = k["op"].fillna("node:" + k["node"].astype(str))
        per = (k.groupby(KEY + ["source", "phase", "run", "label"])["dur_us"].sum().reset_index()
               .groupby(["model", "backend", "source", "label"])["dur_us"].median().reset_index())
        per["dur_ms_per_run"] = per["dur_us"] / 1e3
        per["share"] = per["dur_us"] / per.groupby(["model", "backend", "source"])["dur_us"].transform("sum")
        top = (per.sort_values("dur_us", ascending=False).groupby(["model", "backend", "source"]).head(10)
               .drop(columns="dur_us").sort_values(["model", "backend", "source", "dur_ms_per_run"], ascending=[True, True, True, False]))
    return summary, top


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    args = ap.parse_args()
    summary, top = summarize(args.dir)
    out = Path(args.dir)
    summary.to_csv(out / "profile_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 40):
        print(summary.round(3).T.to_string())
        if not top.empty:
            top.to_csv(out / "top_kernels.csv", index=False)
            print("\nTop op types by time per run:")
            print(top.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
