"""§3 Why is WebGL slow on DeepLabV3+? Per-inference CPU split, V8 profile, readback microbenchmark.

Three panels, each one direct measurement (no values combined across phases):
  (a) profile_gltime: wall = time inside readPixels + inside other WebGL calls + JS between calls;
      GPU busy (timer queries, profile_gpu) marked; drain/transfer (profile_readsplit) annotated
  (b) profile_cpuprof: V8 sampling profile self time by category, per run
  (c) readback_micro: GPU->JS readback time vs bytes per API path, and ORT's decode vs a loop
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import style

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from webgl_readback import analyse  # noqa: E402

ORDER = ["resnet50-v2", "yolov2-coco-9", "deeplabv3p-resnet50-human"]
PROF_CATS = [("readPixels (native)", "readPixels (native)", "#E69F00"),
             ("ORT decode (RGBA unpack)", "ORT decode (RGBA unpack)", "#D55E00"),
             ("gc", "garbage collection", "#CC79A7"),
             ("ORT other JS", "other ORT JS", "#56B4E9"),
             ("other WebGL/Web API (native)", "other WebGL calls", "#009E73"),
             ("harness JS", "harness + (program)", "#BBBBBB"),
             ("program", None, "#BBBBBB")]
MICRO = [("webgl_sync_rgba", "WebGL readPixels RGBA (ORT's path)", "#E69F00", "D", "-"),
         ("webgl_sync_red", "WebGL readPixels RED", "#F0E442", "d", "-"),
         ("webgl_pbo_rgba", "WebGL PBO + fence, RGBA", "#999999", "s", "-"),
         ("webgpu_map", "WebGPU mapAsync", "#0072B2", "o", "-"),
         ("decode_ort_filter", "ORT decode: filter(i % 4 == 0)", "#D55E00", "v", "--"),
         ("decode_strided_loop", "strided loop (same output)", "#009E73", "^", "--")]


def webgl_readback(study, out):
    a = analyse(study.root)
    if "phases" not in a and "micro" not in a:
        return {"skipped": "no webgl_readback / readback_micro stage"}
    fig, axes = plt.subplots(1, 3, figsize=(style.DOUBLE, 3.3), gridspec_kw={"width_ratios": [1.0, 1.0, 1.2], "wspace": 0.45})
    res = {}

    if "phases" in a:
        ph = a["phases"].set_index("model")
        models = [m for m in ORDER if m in ph.index]
        ax = axes[0]
        for i, m in enumerate(models):
            r = ph.loc[m]
            wall = r["wall_gltime_ms"]
            other_gl = r["gl_total_ms"] - r["gl_readpixels_ms"]
            segs = [(r["gl_readpixels_ms"], "#E69F00", "inside readPixels"),
                    (other_gl, "#009E73", "inside other WebGL calls"),
                    (r["js_ms"], "#666666", "JS between WebGL calls")]
            left = 0
            for v, c, lab in segs:
                ax.barh(i, 100 * v / wall, left=left, color=c, label=lab if i == 0 else None, height=0.6)
                left += 100 * v / wall
            if np.isfinite(r["gpu_busy_ms"]):
                ax.plot(100 * r["gpu_busy_ms"] / wall, i + 0.36, marker="v", color="k", ms=4,
                        label="GPU busy (timer queries)" if i == 0 else None, clip_on=False)
            ax.text(99, i - 0.36, f"wall {wall:.0f} ms", ha="right", va="bottom", fontsize=6)
        ax.set_yticks(range(len(models)), [style.short(m) for m in models])
        ax.set_xlim(0, 100); ax.set_xlabel("% of inference wall time")
        ax.set_title("(a) CPU view of one WebGL inference", fontsize=7)
        ax.legend(fontsize=5.5, loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=1)
        ax.grid(False, axis="y")

        ax = axes[1]
        prof_cols = {c: f"prof_{c}_ms_per_run" for c, *_ in PROF_CATS}
        for i, m in enumerate(models):
            r = ph.loc[m]
            tot = r.get("prof_busy_ms_per_run", np.nan)
            left = 0
            for c, lab, col in PROF_CATS:
                v = r.get(prof_cols[c], 0.0)
                v = 0.0 if not np.isfinite(v) else v
                ax.barh(i, 100 * v / tot if tot else 0, left=left, color=col, height=0.6,
                        label=lab if (i == 0 and lab) else None)
                left += 100 * v / tot if tot else 0
        ax.set_ylim(-0.6, len(models) - 0.4)
        ax.set_yticks(range(len(models)), [""] * len(models))
        ax.set_xlim(0, 100); ax.set_xlabel("% of sampled main-thread time")
        ax.set_title("(b) V8 profile of the same page", fontsize=7)
        ax.legend(fontsize=5.5, loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=1)
        ax.grid(False, axis="y")

        keep = ["wall_counters_ms", "wall_gltime_ms", "gl_calls", "gl_total_ms", "gl_readpixels_ms", "js_ms",
                "gpu_busy_ms", "drain_ms", "transfer_ms", "readpixels_bytes", "m1_sum_le_wall_violations",
                "m2_split_over_m1_readpixels", "gltime_over_counters", "readsplit_over_counters", "cpuprof_over_counters",
                "js_ms_ci", "drain_ms_ci", "transfer_ms_ci"] + \
               [c for c in ph.columns if c.startswith(("prof_", "trace_"))]
        ph[[c for c in keep if c in ph.columns]].to_csv(out / "fig22_webgl_split.csv")
        res["split"] = {m: {k: (list(np.round(v, 2)) if isinstance(v, tuple) else
                                (round(float(v), 3) if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(v) else None))
                            for k, v in ph.loc[m, [c for c in keep if c in ph.columns]].items()} for m in models}
        res["top_functions_ms_per_run"] = {f"t{t}|{m}": {k: round(v, 1) for k, v in d.items()}
                                           for (t, m), d in a["top_functions"].items()}
        res["gpu_process_trace_ms_per_run"] = {f"t{t}|{m}": {k: round(v, 1) for k, v in list(d.items())[:8]}
                                               for (t, m), d in a["trace_top"].items()}

    if "micro" in a:
        mt = a["micro"]
        ax = axes[2]
        for path, lab, col, mk, ls in MICRO:
            g = mt[mt["path"] == path].sort_values("bytes")
            if len(g):
                ax.plot(g["bytes"] / 2**20, g["total_ms"], marker=mk, color=col, ls=ls, label=lab, ms=3.5)
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.axvline(80, color="k", lw=0.6, ls=":")
        ax.text(78, ax.get_ylim()[1], "DeepLabV3+\nRGBA readback", fontsize=5.5, ha="right", va="top")
        ax.set_xlabel("bytes read back (MiB, log)"); ax.set_ylabel("ms (log)")
        ax.set_title("(c) readback and unpack cost, no ORT", fontsize=7)
        ax.legend(fontsize=5.5, loc="upper left", bbox_to_anchor=(0.0, -0.22), ncol=1)
        mt.to_csv(out / "fig22_readback_micro.csv", index=False)
        at = lambda p, mib: mt[(mt["path"] == p) & (mt["bytes"] == mib * 2**20)]["total_ms"]
        res["micro_ms"] = {p: {f"{mib}MiB": (round(float(at(p, mib).iloc[0]), 2) if len(at(p, mib)) else None)
                               for mib in sorted(set(mt["bytes"] // 2**20))} for p, *_ in MICRO}
        res["micro_env"] = {k: v for k, v in a["micro_env"].items() if k != "renderer"}

    res["files"] = style.save(fig, out, "fig22_webgl_readback")
    return res
