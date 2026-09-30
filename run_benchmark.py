"""
ONNX Runtime Web benchmark driver (Windows + NVIDIA + Google Chrome Dev).

Each (trial, model, config) experiment runs in a brand-new Chrome Dev process with an empty
profile, so no shader, pipeline or HTTP cache carries over. The page (src/main.js) exposes
window.__bench; results come back as structured JSON, not scraped text.

Backends are strict: exactly one execution provider, and WebGPU sessions refuse to place any
node on the CPU unless --allow-cpu-nodes is given. A run is only marked "ok" when graphics-API
counters confirm the requested backend did the work.

Output (benchmark_results/<timestamp>/):
    env.json            host, GPU, browser, ORT and run configuration
    experiments.jsonl   one raw record per experiment (source of truth)
    sessions.csv        one row per experiment (status, load times, verification)
    runs.csv            one row per inference (phase, index, start time, duration)
    logs/               browser console per experiment

Usage:
    npm run bench:serve    # production build + preview server (port 4173), in another terminal
    python run_benchmark.py --config wasm-4t webgpu webgl --filter mobilenetv2 --warmup 10 --measure 200

Configs (see CONFIGS): an EP backend plus runtime options. In the CSVs the "backend" column holds
the config label and "ep" the execution provider.
"""
import argparse
import asyncio
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

from playwright.async_api import async_playwright

from benchkit.chrome import CHROME_FLAGS, chrome_page, find_chrome_dev
from benchkit.correctness import compare as compare_outputs
from benchkit.devtools import DiagCollector
from benchkit.envinfo import host_info
from benchkit.results import ResultWriter

# Named experiment configurations: execution provider + runtime options.
CONFIGS = {
    "wasm-1t": {"backend": "wasm", "wasmThreads": 1},
    "wasm-4t": {"backend": "wasm", "wasmThreads": 4},       # ORT Web's own default thread count here
    "wasm-8t": {"backend": "wasm", "wasmThreads": 8},       # = P-core count of the i7-12700K
    "webgl": {"backend": "webgl"},
    "webgpu": {"backend": "webgpu"},                        # JSEP, CPU I/O
    "webgpu-native": {"backend": "webgpu", "epImpl": "native"},   # C++ WebGPU EP (onnxruntime-web/webgpu)
    "webgpu-gpuio": {"backend": "webgpu", "gpuIo": True},        # inputs/outputs stay on the GPU
    "webgpu-capture": {"backend": "webgpu", "gpuIo": True, "graphCapture": True},
}
BASE_CONFIGS = ["wasm-1t", "wasm-4t", "wasm-8t", "webgl", "webgpu"]
MODES = ("latency", "shapes", "batch", "profile", "correctness", "responsiveness", "llm", "load")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:4173/", help="benchmark page; start it with: npm run bench:serve")
    p.add_argument("--models", default="./public/models/models.json")
    p.add_argument("--config", nargs="+", default=BASE_CONFIGS, choices=list(CONFIGS),
                   help="experiment configs (EP + runtime options), see CONFIGS")
    p.add_argument("--outputs", default="all", choices=("all", "primary"),
                   help="fetch every model output, or only the primary one (logits_outputs[0] / first output)")
    p.add_argument("--exclude", nargs="+", default=[], metavar="MODEL:CONFIG",
                   help="skip these (model substring, config) pairs, e.g. vit_large:wasm-1t")
    p.add_argument("--batch-values", type=int, nargs="+", default=None,
                   help="batch mode: override the model's shape_sweep values")
    p.add_argument("--cool-to-c", type=float, default=None,
                   help="before each launch, wait (max 180 s) until the GPU is at or below this temperature")
    p.add_argument("--animate", action="store_true", help="responsiveness: also run the rAF canvas animation")
    p.add_argument("--filter", nargs="+", default=None, help="keep models whose name contains any of these substrings")
    p.add_argument("--mode", default="latency", choices=MODES,
                   help="latency: closed-loop runs at the base shape (groups 1-2); "
                        "shapes: first-seen vs revisited input shapes (group 1, models with shape_sweep only); "
                        "batch: per-sample latency across the shape_sweep batch sizes; "
                        "profile: instrumented pass - API counters, per-kernel GPU/CPU time, memory (groups 5-7); "
                        "correctness: outputs vs native ONNX Runtime CPU on identical inputs (group 10); "
                        "responsiveness: frame timing + interaction latency with/without inference (group 4); "
                        "llm: token generation timing, models with an llm spec (group 3); "
                        "load: open-loop Poisson arrivals with deadlines (group 8)")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--measure", type=int, default=200)
    p.add_argument("--shape-rounds", type=int, default=3, help="shapes mode: passes over the sweep values")
    p.add_argument("--shape-repeats", type=int, default=3, help="shapes mode: consecutive runs per shape block")
    p.add_argument("--page-baseline-s", type=float, default=5.0, help="responsiveness: animation-only phase length")
    p.add_argument("--load-s", type=float, default=10.0, help="responsiveness: inference phase; load: arrival window per rate")
    p.add_argument("--click-interval-s", type=float, default=0.25, help="responsiveness: time between scripted clicks")
    p.add_argument("--pacing", default="yield", choices=("yield", "back_to_back"),
                   help="responsiveness/load: yield to the browser between inferences, or run them back to back")
    p.add_argument("--prompt-tokens", type=int, default=32, help="llm: prompt length")
    p.add_argument("--gen-tokens", type=int, default=32, help="llm: tokens generated per generation")
    p.add_argument("--rates", type=float, nargs="+", default=[1, 2, 5, 10, 20, 50], help="load: arrival rates (requests/s)")
    p.add_argument("--max-queue", type=int, default=16, help="load: queue capacity; arrivals beyond it are rejected")
    p.add_argument("--drain-s", type=float, default=5.0, help="load: time allowed to drain the queue after each window")
    p.add_argument("--with-animation", action="store_true", help="load: run the page animation to record frame timing under load")
    p.add_argument("--trials", type=int, default=3, help="independent browser launches per (model, backend)")
    p.add_argument("--seed", type=int, default=1, help="input-data seed and experiment-order seed")
    p.add_argument("--no-shuffle", action="store_true", help="run experiments in fixed model/backend order")
    p.add_argument("--webgl-pack", action="store_true", help="enable ORT WebGL texture packing (ORT default: off)")
    p.add_argument("--gl-timing", action="store_true",
                   help="profile, WebGL: add phases timing every WebGL call and splitting readPixels (drain/transfer)")
    p.add_argument("--cpu-profile", action="store_true",
                   help="profile, WebGL with --gl-timing: V8 sampling profile of extra runs (logs/<tag>.cpuprofile); "
                        "use the unminified diagnostic build (vite build --mode diag, port 4174) for function names")
    p.add_argument("--trace", action="store_true",
                   help="record a Chrome trace with GPU-process categories for each experiment (logs/<tag>.trace.json.gz)")
    p.add_argument("--allow-cpu-nodes", action="store_true",
                   help="let ORT place unsupported WebGPU nodes on the CPU (default: such models fail)")
    p.add_argument("--session-timeout-s", type=int, default=600)
    p.add_argument("--experiment-timeout-s", type=int, default=3600)
    p.add_argument("--cooldown-s", type=float, default=5.0, help="idle time between browser launches")
    p.add_argument("--no-telemetry", action="store_true", help="disable NVML GPU telemetry")
    p.add_argument("--telemetry-interval-s", type=float, default=0.05)
    p.add_argument("--no-pcie", action="store_true", help="skip PCIe throughput sampling (each query blocks ~25 ms)")
    p.add_argument("--baseline-s", type=float, default=3.0, help="idle GPU telemetry recorded before each launch")
    p.add_argument("--executable-path", default=None, help="Chrome Dev path (default: auto-detect)")
    p.add_argument("--headless", action="store_true", help="headless Chrome (default: headed, recommended for GPU runs)")
    p.add_argument("--output-dir", default="benchmark_results")
    return p.parse_args()


def load_models(path, filters):
    models = json.loads(Path(path).read_text(encoding="utf-8"))["models"]
    if filters:
        models = [m for m in models if any(f.lower() in m["name"].lower() for f in filters)]
    return models


def check_server(url):
    try:
        with urlopen(url, timeout=10) as r:
            coop = r.headers.get("Cross-Origin-Opener-Policy")
            coep = r.headers.get("Cross-Origin-Embedder-Policy")
    except Exception as e:
        sys.exit(f"Dev server not reachable at {url} ({e}). Start it with: npm run bench:serve")
    if coop != "same-origin" or coep != "require-corp":
        print(f"WARNING: page is not cross-origin isolated (COOP={coop}, COEP={coep}); "
              "timer resolution will be 100 µs instead of 5 µs.")


async def click_while_running(page, experiment, interval_s):
    """Responsiveness mode: click the page's interaction target with real input events until the
    experiment finishes. Returns the send time and round-trip of each click (epoch ms)."""
    clicks = []
    while not experiment.done():
        if await page.evaluate("() => !!window.__bench.interactionTarget"):
            break
        await asyncio.sleep(0.05)
    target = page.locator("#interactBtn")
    await target.scroll_into_view_if_needed()
    box = await target.bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    if not (0 <= y <= page.viewport_size["height"] and 0 <= x <= page.viewport_size["width"]):
        raise RuntimeError(f"interaction target at ({x:.0f}, {y:.0f}) is outside the viewport")
    while not experiment.done():
        t0 = time.time() * 1000.0
        await page.mouse.click(x, y)
        clicks.append({"t_send_epoch_ms": t0, "roundtrip_ms": time.time() * 1000.0 - t0})
        await asyncio.sleep(interval_s)
    return clicks


async def wait_for_gpu_temp(limit_c, timeout_s=180):
    """Hold the next launch until the GPU has cooled to limit_c (returns seconds waited)."""
    import pynvml as nv
    nv.nvmlInit()
    h = nv.nvmlDeviceGetHandleByIndex(0)
    t0 = time.time()
    while nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU) > limit_c and time.time() - t0 < timeout_s:
        await asyncio.sleep(2)
    return time.time() - t0


async def run_one(pw, args, chrome, model, config, trial, writer, telemetry=None):
    backend = CONFIGS[config]["backend"]
    if backend == "webgl" and model.get("webgl_path"):
        # Same graph and weights with input dims pinned: WebGL rejects symbolic input dims
        # (public/patch/pin_input_dims.py). Page and correctness reference both use this file.
        model = {**model, "path": model["webgl_path"]}
    tag = f"t{trial}_{model['name']}_{config}"
    log_path = writer.out_dir / "logs" / f"{tag}.log"
    cfg = {
        **CONFIGS[config], "configLabel": config, "outputs": args.outputs,
        "batchValues": args.batch_values, "animate": args.animate,
        "model": model, "backend": backend, "mode": args.mode,
        "warmup": args.warmup, "measure": args.measure, "seed": args.seed,
        "shapeRounds": args.shape_rounds, "shapeRepeats": args.shape_repeats,
        "webglPack": args.webgl_pack,
        "allowCpuNodes": args.allow_cpu_nodes,
        "sessionTimeoutMs": args.session_timeout_s * 1000,
        "baselineSeconds": args.page_baseline_s, "loadSeconds": args.load_s,
        "promptTokens": args.prompt_tokens, "genTokens": args.gen_tokens,
        "rates": args.rates, "maxQueue": args.max_queue, "drainSeconds": args.drain_s,
        "withAnimation": args.with_animation, "pacing": args.pacing,
        "glTiming": args.gl_timing, "cpuProfile": args.cpu_profile and args.gl_timing,
    }
    diag = None
    if args.cpu_profile or args.trace:
        diag = DiagCollector(writer.out_dir / "logs" / tag, cpu_profile=cfg["cpuProfile"], trace=args.trace)
    started = datetime.now().isoformat(timespec="seconds")
    cooled_s = await wait_for_gpu_temp(args.cool_to_c) if args.cool_to_c is not None else None
    if telemetry:
        telemetry.start()
        telemetry.mark("baseline_start")
        await asyncio.sleep(args.baseline_s)
        telemetry.mark("baseline_end")
    try:
        # Profile mode measures page memory; this flag makes measureUserAgentSpecificMemory() run
        # immediately instead of waiting for the next garbage collection.
        extra = ["--enable-blink-features=ForceEagerMeasureMemory"] if args.mode == "profile" else []
        on_launch = telemetry.watch_chrome if telemetry else None
        async with chrome_page(pw, chrome, args.url, args.headless, extra_flags=extra, log_path=log_path,
                               on_launch=on_launch, cdp_setup=diag.setup if diag else None) as (page, version):
            page_env = await page.evaluate("() => window.__bench.probeEnv()")
            experiment = asyncio.ensure_future(page.evaluate("cfg => window.__bench.runExperiment(cfg)", cfg))
            clicks = await click_while_running(page, experiment, args.click_interval_s) if args.mode == "responsiveness" else None
            result = await asyncio.wait_for(experiment, timeout=args.experiment_timeout_s)
            if clicks is not None:
                result["python_clicks"] = clicks
            result["browser_version"] = version
            result["page_env"] = page_env
    except Exception as e:
        # The browser itself failed (crash, OOM kill, hang): recorded as a failed experiment.
        result = {
            "schema": 2, "model": model["name"], "backend": backend, "config_label": config, "status": "failed",
            "failure": {"stage": "browser", "category": "browser_failure", "message": f"{type(e).__name__}: {e}"},
            "runs": [],
        }
    result.update({"trial": trial, "started_at": started, "cooled_s": cooled_s, "model_path": model["path"]})
    if diag:
        result["devtools"] = diag.status
    if result.get("io"):
        # Group 10: compare against native ONNX Runtime on the exact same input bytes.
        io = result.pop("io")
        model_file = Path(args.models).resolve().parents[1] / model["path"].lstrip("/")
        try:
            rows = compare_outputs(model_file, io, model.get("logits_outputs", ()))
        except Exception as e:
            rows = [{"error": f"reference run failed: {type(e).__name__}: {e}"}]
        result["correctness"] = rows
        writer.append_rows("correctness.csv", [{"trial": trial, "model": model["name"], "backend": config, **r} for r in rows])
    if telemetry:
        telemetry.mark("experiment_end")
        data = telemetry.stop()
        key = {"trial": trial, "model": model["name"], "backend": config}
        writer.append_rows("gpu_samples.csv", [{**key, **s} for s in data["samples"]])
        writer.append_rows("gpu_util_samples.csv", [{**key, **s} for s in data["util"]])
        writer.append_rows("gpu_pcie_samples.csv", [{**key, **s} for s in data["pcie"]])
        writer.append_rows("gpu_marks.csv", [{**key, **s} for s in data["marks"]])
        writer.append_rows("chrome_memory_samples.csv", [{**key, **s} for s in data["chrome"]])
        writer.append_rows("system_cpu_samples.csv", [{**key, **s} for s in data["system_cpu"]])
        result["telemetry"] = {"samples": len(data["samples"]), "util_samples": len(data["util"]),
                               "pcie_samples": len(data["pcie"]), "chrome_samples": len(data["chrome"]),
                               "errors": data["errors"][:20]}
    writer.append(result)
    return result


async def main():
    args = parse_args()
    chrome = args.executable_path or find_chrome_dev()
    if not chrome:
        sys.exit("Google Chrome Dev not found. Install it (winget install Google.Chrome.Dev), "
                 "set CHROME_DEV_PATH, or pass --executable-path.")
    check_server(args.url)

    models = load_models(args.models, args.filter)
    if args.mode == "llm":
        models = [m for m in models if "llm" in m]
    if args.mode in ("shapes", "batch"):
        static = [m["name"] for m in models if "shape_sweep" not in m]
        if static:
            print(f"{args.mode} mode: skipping static-shape models (no symbolic input dims): {', '.join(static)}")
        models = [m for m in models if "shape_sweep" in m]
    if not models:
        sys.exit("No models matched.")

    out_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = ResultWriter(out_dir)
    env = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "args": vars(args),
        "chrome_executable": chrome,
        "chrome_flags": CHROME_FLAGS,
        "host": host_info(),
        "page_env": None,
    }
    writer.write_json("env.json", env)

    configs = list(args.config)
    if args.mode == "llm":
        dropped = [c for c in configs if CONFIGS[c].get("gpuIo")]
        if dropped:
            print(f"llm mode uses CPU I/O only; skipping {', '.join(dropped)}")
        configs = [c for c in configs if not CONFIGS[c].get("gpuIo")]

    def excluded(model, config):
        return any(sub.lower() in model["name"].lower() and cfg == config
                   for sub, cfg in (e.split(":", 1) for e in args.exclude))

    plan = [(t, m, c) for t in range(1, args.trials + 1) for m in models for c in configs if not excluded(m, c)]
    if not args.no_shuffle:
        rng = random.Random(args.seed)
        # Shuffle within each trial so trials stay sequential but order effects (thermal state,
        # background activity) don't line up with a fixed model/config order.
        plan = [x for t in range(1, args.trials + 1) for x in rng.sample([p for p in plan if p[0] == t], k=sum(p[0] == t for p in plan))]

    telemetry = None
    if not args.no_telemetry:
        from benchkit.telemetry import NvmlTelemetry
        telemetry = NvmlTelemetry(interval_s=args.telemetry_interval_s, pcie=not args.no_pcie)

    print(f"Output: {out_dir}\nChrome: {chrome}\n{len(plan)} experiments")
    async with async_playwright() as pw:
        for n, (trial, model, config) in enumerate(plan, 1):
            t0 = time.time()
            print(f"[{n}/{len(plan)}] trial {trial} {model['name']} / {config} ...", flush=True)
            r = await run_one(pw, args, chrome, model, config, trial, writer, telemetry)
            measured = [x["dur_ms"] for x in r.get("runs", []) if x["phase"] == "measure"]
            detail = (r.get("verification") or {}).get("detail") or (r.get("failure") or {}).get("message", "")
            median = sorted(measured)[len(measured) // 2] if measured else None
            print(f"    {r['status']}"
                  + (f", median {median:.3f} ms over {len(measured)} runs" if median is not None else "")
                  + f" ({time.time() - t0:.0f}s) {detail[:160]}", flush=True)
            if env["page_env"] is None and r.get("page_env"):
                env["page_env"] = r["page_env"]
                env["browser_version"] = r.get("browser_version")
                writer.write_json("env.json", env)
            if n < len(plan):
                await asyncio.sleep(args.cooldown_s)

    n_sessions, n_runs = writer.export_csvs()
    print(f"Done: {n_sessions} experiments, {n_runs} inference records -> {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
