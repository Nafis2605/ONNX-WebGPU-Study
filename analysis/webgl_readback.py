"""
Where WebGL's per-inference time goes (stage webgl_readback) and what the readback path costs
by itself (stage readback_micro). Every number is a direct measurement; the cross-checks between
independent measurements are reported alongside.

  M1  profile_gltime     CPU time inside WebGL calls by category; js_ms = wall - GL time
  M2  profile_readsplit  readPixels = drain (1x1 read first: waits for queued draws) + transfer
  M3  profile_cpuprof    V8 sampling profile (logs/<tag>.cpuprofile): self time by function
  M4  trace              GPU-process "WebGL" command-buffer slices inside the cpuprof window
  GPU profile_gpu        summed per-draw GPU time (timer queries)
  M5  readback_micro     readback ms per API path and size; ORT's decode vs a strided loop

Usage:
    python analysis/webgl_readback.py benchmark_results/study_full
"""
import argparse
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["trial", "model", "backend"]
GL_COLS = ["gl_readpixels_ms", "gl_draw_ms", "gl_upload_ms", "gl_state_ms", "gl_sync_ms", "gl_other_ms"]


def newest(stage_dir, marker):
    subs = [s for s in Path(stage_dir).iterdir() if s.is_dir() and (s / marker).exists()] if Path(stage_dir).exists() else []
    return max(subs, key=lambda s: s.stat().st_mtime) if subs else None


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    meds = np.median(rng.choice(x, (n, len(x))), axis=1)
    return tuple(np.percentile(meds, [2.5, 97.5]))


# ---------------------------------------------------------------- M1, M2, GPU (runs.csv)
def phase_tables(d):
    s = pd.read_csv(d / "sessions.csv")
    ok = s[s["status"] == "ok"][KEY]
    r = pd.read_csv(d / "runs.csv", low_memory=False).merge(ok, on=KEY)
    rows = []
    for m, g in r.groupby("model"):
        gl, rs = g[g.phase == "profile_gltime"], g[g.phase == "profile_readsplit"]
        cnt, gpu, cp = g[g.phase == "profile_counters"], g[g.phase == "profile_gpu"], g[g.phase == "profile_cpuprof"]
        row = {"model": m, "runs_gltime": len(gl), "trials": g["trial"].nunique(),
               "wall_counters_ms": cnt.dur_ms.median(), "wall_gltime_ms": gl.dur_ms.median(),
               "wall_readsplit_ms": rs.dur_ms.median(), "wall_cpuprof_ms": cp.dur_ms.median() if len(cp) else np.nan,
               "gl_calls": gl.gl_calls.median(), "gl_total_ms": gl.gl_ms_total.median(), "js_ms": gl.js_ms.median(),
               "gpu_busy_ms": gpu.gpu_busy_ms.median() if "gpu_busy_ms" in gpu else np.nan,
               "drain_ms": rs.readpixels_drain_ms.median(), "transfer_ms": rs.readpixels_transfer_ms.median(),
               "readpixels_calls": rs.readpixels_calls.median(), "readpixels_bytes": cnt.webgl_read_pixels_bytes.median()}
        for c in GL_COLS:
            row[c] = gl[c].median()
        row["js_ms_ci"] = boot_ci(gl.js_ms)
        row["gl_readpixels_ms_ci"] = boot_ci(gl.gl_readpixels_ms)
        row["drain_ms_ci"] = boot_ci(rs.readpixels_drain_ms)
        row["transfer_ms_ci"] = boot_ci(rs.readpixels_transfer_ms)
        # Cross-checks.
        row["m1_sum_le_wall_violations"] = int((gl.gl_ms_total > gl.dur_ms + 1e-6).sum())
        row["m2_split_over_m1_readpixels"] = (row["drain_ms"] + row["transfer_ms"]) / row["gl_readpixels_ms"] if row["gl_readpixels_ms"] else np.nan
        row["gltime_over_counters"] = row["wall_gltime_ms"] / row["wall_counters_ms"]
        row["readsplit_over_counters"] = row["wall_readsplit_ms"] / row["wall_counters_ms"]
        row["cpuprof_over_counters"] = row["wall_cpuprof_ms"] / row["wall_counters_ms"]
        row["runs_cpuprof_per_trial"] = cp.groupby("trial").size().median() if len(cp) else 0
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- M3 (V8 CPU profile)
def js_category(fn, url):
    if fn == "(garbage collector)":
        return "gc"
    if fn in ("(idle)",):
        return "idle"
    if fn in ("(program)", "(root)"):
        return "program"
    if not url:
        return "readPixels (native)" if fn == "readPixels" else "other WebGL/Web API (native)"
    if "ort.all" in url or "ort-wasm" in url:
        if fn == "decode":
            return "ORT decode (RGBA unpack)"
        if fn == "encode":
            return "ORT encode (upload pack)"
        return "ORT other JS"
    return "harness JS"


def cpu_profile_table(d):
    exps = [json.loads(l) for l in (d / "experiments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows, fns = [], {}
    for e in exps:
        if e.get("status") != "ok":
            continue
        tag = f"t{e['trial']}_{e['model']}_{e.get('config_label', e['backend'])}"
        f = d / "logs" / f"{tag}.cpuprofile"
        if not f.exists():
            continue
        runs = sum(1 for r in e["runs"] if r["phase"] == "profile_cpuprof")
        cat, fn_ms = Counter(), Counter()
        deltas = []
        for item in json.loads(f.read_text(encoding="utf-8")):
            p = item["profile"]
            nodes = {n["id"]: n["callFrame"] for n in p["nodes"]}
            # The first sample's delta spans the profiler start-up, not a run: dropped.
            for sid, dt in list(zip(p["samples"], p["timeDeltas"]))[1:]:
                cf = nodes[sid]
                c = js_category(cf["functionName"], cf["url"])
                cat[c] += dt / 1e3
                fn_ms[(cf["functionName"] or "(anonymous)", cf["url"].rsplit("/", 1)[-1])] += dt / 1e3
                deltas.append(dt)
        row = {"trial": e["trial"], "model": e["model"], "runs": runs,
               "sample_interval_us_median": float(np.median(deltas)) if deltas else np.nan}
        row.update({f"{k}_ms_per_run": v / runs for k, v in cat.items()})
        row["sampled_ms_per_run"] = sum(cat.values()) / runs
        row["busy_ms_per_run"] = sum(v for k, v in cat.items() if k != "idle") / runs
        rows.append(row)
        fns[(e["trial"], e["model"])] = {f"{a} [{b}]": v / runs for (a, b), v in fn_ms.most_common(12)}
    t = pd.DataFrame(rows)
    if len(t):
        t = t.fillna(0)
    return t, fns


# ---------------------------------------------------------------- M4 (Chrome trace)
def trace_table(d):
    rows, tops = [], {}
    exps = [json.loads(l) for l in (d / "experiments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for e in exps:
        if e.get("status") != "ok":
            continue
        tag = f"t{e['trial']}_{e['model']}_{e.get('config_label', e['backend'])}"
        f = d / "logs" / f"{tag}.trace.json.gz"
        if not f.exists():
            continue
        runs = sum(1 for r in e["runs"] if r["phase"] == "profile_cpuprof")
        t = json.load(gzip.open(f))
        ev = t["traceEvents"] if isinstance(t, dict) else t
        procs = {x["pid"]: x["args"]["name"] for x in ev if x.get("ph") == "M" and x.get("name") == "process_name"}
        # The page's performance.measure("cpuprof") gives the window (async b/e pair, µs).
        b = [x["ts"] for x in ev if x.get("name") == "cpuprof" and x.get("ph") == "b"]
        en = [x["ts"] for x in ev if x.get("name") == "cpuprof" and x.get("ph") == "e"]
        if not b or not en or not runs:
            rows.append({"trial": e["trial"], "model": e["model"], "window_found": False})
            continue
        w0, w1 = b[0], en[-1]
        gpu = [x for x in ev if procs.get(x.get("pid")) == "GPU Process" and x.get("ph") == "X" and "dur" in x
               and x["ts"] < w1 and x["ts"] + x["dur"] > w0]
        clip = lambda x: min(x["ts"] + x["dur"], w1) - max(x["ts"], w0)
        webgl = [x for x in gpu if x["name"] == "WebGL"]
        # Busy time of the GPU process's command-buffer thread(s): union of top-level slices.
        by_name = Counter()
        for x in gpu:
            by_name[x["name"]] += clip(x) / 1e3
        rows.append({"trial": e["trial"], "model": e["model"], "window_found": True, "runs": runs,
                     "window_ms_per_run": (w1 - w0) / 1e3 / runs,
                     "gpu_process_webgl_slices_ms_per_run": sum(clip(x) for x in webgl) / 1e3 / runs,
                     "gpu_process_webgl_slices_per_run": len(webgl) / runs})
        tops[(e["trial"], e["model"])] = {k: v / runs for k, v in by_name.most_common(12)}
    return pd.DataFrame(rows), tops


# ---------------------------------------------------------------- M5 (microbenchmark)
def micro_table(d):
    r = pd.read_csv(d / "readback_micro.csv")
    exps = [json.loads(l) for l in (d / "experiments.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    bad = set()
    for e in exps:
        for k in e["env"]:
            if k.startswith("gl_error_"):
                bad.add((e["trial"], int(k[len("gl_error_"):-3]) << 20))
    r = r[~r.apply(lambda x: (x["trial"], x["bytes"]) in bad and x["path"].startswith("webgl"), axis=1)]
    t = r.groupby(["path", "bytes"]).agg(
        n=("total_ms", "size"), total_ms=("total_ms", "median"),
        wait_ms=("wait_ms", "median") if "wait_ms" in r else ("total_ms", "size"),
        copy_ms=("copy_ms", "median") if "copy_ms" in r else ("total_ms", "size")).reset_index()
    t["mb_per_s"] = t["bytes"] / 1e6 / (t["total_ms"] / 1e3)
    env = exps[0]["env"] if exps else {}
    return t, {"red_float_read": env.get("red_float_read"), "gl_error_sizes": sorted({b for _, b in bad}),
               "renderer": env.get("webgl_renderer")}


def psutil_reference(study):
    """Per-inference CPU time by Chrome process from the latency stage (analysis/figures fig16)."""
    f = study / "figures" / "fig16_cpu_cost.csv"
    alt = Path("study_report/figures/fig16_cpu_cost.csv")
    f = f if f.exists() else alt
    if not f.exists():
        return {}
    c = pd.read_csv(f)
    c = c[c["backend"] == "webgl"]
    return {(m, p): v for m, p, v in zip(c["model"], c["process"], c["cpu_ms_per_inference"])}


def analyse(study):
    study = Path(study)
    out = {}
    wr = newest(study / "webgl_readback", "sessions.csv")
    mi = newest(study / "readback_micro", "readback_micro.csv")
    if wr is not None:
        ph = phase_tables(wr)
        cpu, fns = cpu_profile_table(wr)
        tr, tops = trace_table(wr)
        ref = psutil_reference(study)
        if len(cpu):
            agg = cpu.groupby("model").median(numeric_only=True).drop(columns="trial").reset_index()
            agg["psutil_renderer_cpu_ms_per_inference"] = [ref.get((m, "renderer")) for m in agg["model"]]
            ph = ph.merge(agg.add_prefix("prof_").rename(columns={"prof_model": "model"}), on="model", how="left")
        if len(tr) and tr.get("window_found", pd.Series(dtype=bool)).any():
            ta = tr[tr["window_found"]].groupby("model").median(numeric_only=True).drop(columns="trial").reset_index()
            ta["psutil_gpu_process_cpu_ms_per_inference"] = [ref.get((m, "gpu-process")) for m in ta["model"]]
            ph = ph.merge(ta.add_prefix("trace_").rename(columns={"trace_model": "model"}), on="model", how="left")
        out.update({"dir": wr, "phases": ph, "cpu_profile": cpu, "top_functions": fns, "trace": tr, "trace_top": tops})
    if mi is not None:
        mt, menv = micro_table(mi)
        out.update({"micro_dir": mi, "micro": mt, "micro_env": menv})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study")
    a = analyse(ap.parse_args().study)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        if "phases" in a:
            a["phases"].to_csv(a["dir"] / "webgl_readback_summary.csv", index=False)
            print(a["phases"].round(2).T.to_string())
            for k, v in a["top_functions"].items():
                print(k, {n: round(ms, 1) for n, ms in v.items()})
            for k, v in a["trace_top"].items():
                print(k, {n: round(ms, 1) for n, ms in list(v.items())[:8]})
        if "micro" in a:
            a["micro"].to_csv(a["micro_dir"] / "readback_micro_summary.csv", index=False)
            print(a["micro"].round(2).to_string(index=False))
            print(a["micro_env"])


if __name__ == "__main__":
    sys.exit(main())
