"""Result files. experiments.jsonl is the raw source of truth; the CSVs are derived from it."""
import csv
import json
from pathlib import Path

import pandas as pd


class ResultWriter:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "logs").mkdir(exist_ok=True)
        self.jsonl = self.out_dir / "experiments.jsonl"

    def write_json(self, name, obj):
        (self.out_dir / name).write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")

    def append_rows(self, name, rows):
        """Append rows to a CSV in the output dir, writing the header on first use."""
        if not rows:
            return
        path = self.out_dir / name
        header = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if header:
                w.writeheader()
            w.writerows(rows)

    def append(self, record):
        """Append one experiment immediately, so a crash later in the sweep loses nothing."""
        with open(self.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def export_csvs(self):
        return export_csvs(self.out_dir)


def load_experiments(out_dir: Path):
    path = Path(out_dir) / "experiments.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _flatten(record):
    """Flatten nested per-run dicts (counters, memory) into prefixed columns; lists become JSON."""
    out = {}
    for k, v in record.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                out[k2 if k == "counters" else f"{k}_{k2}"] = v2
        elif isinstance(v, list):
            out[k] = json.dumps(v)
        else:
            out[k] = v
    return out


def _mem(e, *path):
    node = e
    for key in path:
        node = (node or {}).get(key)
    return node


def export_csvs(out_dir: Path):
    """(Re)build sessions.csv, runs.csv and (profile mode) kernels.csv from experiments.jsonl."""
    out_dir = Path(out_dir)
    experiments = load_experiments(out_dir)
    sessions, runs, kernels = [], [], []
    extra = {name: [] for name in ("frames", "interactions", "clicks", "loaf", "phases", "load_requests", "load_frames")}
    for e in experiments:
        # "backend" is the config label (e.g. wasm-4t, webgpu-capture); "ep" the execution provider.
        key = {"trial": e["trial"], "model": e["model"], "backend": e.get("config_label", e["backend"])}
        v = e.get("verification") or {}
        c = v.get("counters") or {}
        f = e.get("failure") or {}
        all_runs = e.get("runs", [])
        measured = [r for r in all_runs if r["phase"] == "measure"]
        sessions.append({
            **key,
            "mode": e.get("mode", "latency"),
            "ep": e["backend"],
            "ep_impl": (e.get("config") or {}).get("ep_impl"),
            "wasm_threads": (e.get("config") or {}).get("wasm_threads"),
            "gpu_io": (e.get("config") or {}).get("gpu_io"),
            "graph_capture": (e.get("config") or {}).get("graph_capture"),
            "outputs": (e.get("config") or {}).get("outputs"),
            "status": e["status"],
            "failure_stage": f.get("stage"),
            "failure_category": f.get("category"),
            "failure_message": f.get("message"),
            # File the page loaded (e.g. the fixed-shape WebGL copy); empty for records predating it.
            "model_path": e.get("model_path"),
            "model_bytes": e.get("model_bytes"),
            "fetch_ms": e.get("fetch_ms"),
            "session_create_ms": e.get("session_create_ms"),
            # The first and second inferences of the session (warmup runs 0 and 1).
            "first_run_ms": all_runs[0]["dur_ms"] if len(all_runs) > 0 else None,
            "second_run_ms": all_runs[1]["dur_ms"] if len(all_runs) > 1 else None,
            "n_measure": len(measured),
            "backend_verified": v.get("verified"),
            "verification_detail": v.get("detail"),
            "webgpu_submits": c.get("webgpu_submits"),
            "webgpu_dispatches": c.get("webgpu_dispatches"),
            "webgl_draws": c.get("webgl_draws"),
            # From ORT's node-placement report (WebGPU only; separate inspection session).
            "placement_nodes_per_ep": json.dumps((e.get("placement") or {}).get("nodes_per_ep")) if e.get("placement") else None,
            "cpu_nodes": (e.get("placement") or {}).get("cpu_nodes"),
            "memcpy_to_host_nodes": ((e.get("placement") or {}).get("memcpy_nodes") or {}).get("MemcpyToHost"),
            "memcpy_from_host_nodes": ((e.get("placement") or {}).get("memcpy_nodes") or {}).get("MemcpyFromHost"),
            "browser_version": e.get("browser_version"),
            # Profile mode: logical GPU memory (bytes of live WebGPU buffers / WebGL textures)
            # and page memory from measureUserAgentSpecificMemory().
            "gpu_live_bytes_after_session_create": _mem(e, "profile", "memory", "after_session_create", "live_bytes"),
            "gpu_peak_live_bytes": _mem(e, "gpu_memory_final", "peak_live_bytes"),
            "gpu_live_bytes_final": _mem(e, "gpu_memory_final", "live_bytes"),
            "page_memory_before_fetch": _mem(e, "page_memory", "before_fetch", "bytes"),
            "page_memory_after_session_create": _mem(e, "page_memory", "after_session_create", "bytes"),
            "page_memory_after_runs": _mem(e, "page_memory", "after_runs", "bytes"),
            "kernel_assignment": _mem(e, "profile", "kernel_assignment"),
            "ort_trace_assignment": _mem(e, "profile", "ort_trace_assignment"),
            "ort_session_events": json.dumps(_mem(e, "profile", "session_events")) if _mem(e, "profile", "session_events") else None,
        })
        for r in all_runs:
            runs.append({**key, "mode": e.get("mode", "latency"), **_flatten(r)})
        for k in _mem(e, "profile", "kernels") or []:
            kernels.append({**key, **k})

        # Responsiveness mode: times are page-relative ms (performance.now()); phase windows too.
        resp = e.get("responsiveness")
        if resp:
            phases = resp["phases"]

            def phase_of(t):
                for name, w in phases.items():
                    if w["start"] <= t <= w["end"]:
                        return name
                return "outside"
            fr = resp["frames"]
            for a, b in zip(fr, fr[1:]):
                extra["frames"].append({**key, "t_ms": b, "interval_ms": b - a, "phase": phase_of(b)})
            for ck in resp.get("clicks", []):
                extra["clicks"].append({**key, **ck, "phase": phase_of(ck["event_ts"]),
                                        "input_delay_ms": ck["handler_start"] - ck["event_ts"],
                                        "to_next_frame_ms": (ck["next_frame"] - ck["event_ts"]) if ck["next_frame"] is not None else None})
            for it in resp["interactions"]:
                extra["interactions"].append({**key, **it, "phase": phase_of(it["start"]),
                                              "input_delay_ms": it["processing_start"] - it["start"]})
            for lf in resp["loaf"]:
                # A long frame belongs to the phase containing its midpoint (one that starts just
                # before inference and spans it is caused by inference).
                extra["loaf"].append({**key, **lf, "phase": phase_of(lf["start"] + lf["duration"] / 2)})
            for name, w in phases.items():
                extra["phases"].append({**key, "phase": name, "start_ms": w["start"], "end_ms": w["end"]})

        load = e.get("load")
        if load:
            for r in load["requests"]:
                extra["load_requests"].append({**key, **r})
            for w in load["windows"]:
                fr = w.get("frames") or []
                for a, b in zip(fr, fr[1:]):
                    extra["load_frames"].append({**key, "rate": w["rate"], "t_ms": b, "interval_ms": b - a})

    pd.DataFrame(sessions).to_csv(out_dir / "sessions.csv", index=False)
    pd.DataFrame(runs).to_csv(out_dir / "runs.csv", index=False)
    if kernels:
        pd.DataFrame(kernels).to_csv(out_dir / "kernels.csv", index=False)
    for name, rows in extra.items():
        if rows:
            pd.DataFrame(rows).to_csv(out_dir / f"{name}.csv", index=False)
    return len(sessions), len(runs)
