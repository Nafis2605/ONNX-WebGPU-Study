"""
Run the whole measurement study: every mode of run_benchmark.py, then every analysis script.

Each stage writes its own timestamped folder under <study dir>/<stage>/ and the analysis output
lands next to the raw data. A stage that fails is reported and the study continues.

Prerequisite (separate terminal):  npm run bench:serve

Usage:
    python run_study.py                      # full study (several hours; see README)
    python run_study.py --quick              # smoke run: 1 trial, few runs, small models
    python run_study.py --stages latency profile

    # Supplement: run only the given configs x models inside each stage's own scope, then merge
    # into the stage's newest result (same keys replaced), e.g. WebGL on the fixed-shape copies:
    python run_study.py --output-dir benchmark_results/study_full --supplement --config webgl --models resnet50-v2 deeplabv3p
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PY = sys.executable
sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchkit.merge import merge  # noqa: E402

ALL = ["wasm-1t", "wasm-4t", "wasm-8t", "webgl", "webgpu", "webgpu-native", "webgpu-gpuio", "webgpu-capture"]
GPU_EPS = ["webgpu", "webgpu-native", "webgpu-gpuio"]
COMMON = ["--cool-to-c", "50"]   # every launch starts from a comparable GPU temperature

# (stage, run_benchmark.py arguments, analysis commands). "{dir}" is the stage's result folder.
FULL = {
    # Groups 1-2 and the "when" question: every model x every config.
    "latency": (["--mode", "latency", "--config", *ALL, "--warmup", "10", "--measure", "200", "--trials", "3",
                 "--exclude", "vit_large:wasm-1t"],
                [["analysis/summarize.py", "{dir}"], ["analysis/init_reuse.py", "{dir}"],
                 ["analysis/gpu_telemetry.py", "{dir}"], ["analysis/predict.py", "{dir}", "--slo-ms", "100"]]),
    # Utilisation: per-sample latency vs batch (graph capture needs static shapes; WebGL rejects
    # these models' symbolic dims; 1-thread WASM is too slow at batch 16).
    "batch": (["--mode", "batch", "--config", "wasm-4t", "wasm-8t", *GPU_EPS, "--warmup", "5", "--measure", "30",
               "--trials", "3"], []),
    # Recompilation on first-seen input shapes.
    "shapes": (["--mode", "shapes", "--config", "wasm-8t", "webgpu", "webgpu-native", "--warmup", "5",
                "--shape-rounds", "3", "--shape-repeats", "3", "--trials", "3"],
               [["analysis/init_reuse.py", "{dir}"]]),
    # Readback: only the primary output of the multi-output models.
    "readback": (["--mode", "latency", "--outputs", "primary", "--config", "wasm-8t", "webgpu", "webgpu-native",
                  "--filter", "bert_Opset17", "mobilebert", "gpt2", "--warmup", "10", "--measure", "200", "--trials", "3"],
                 [["analysis/summarize.py", "{dir}"]]),
    # Groups 5-7: where the time goes.
    "profile": (["--mode", "profile", "--config", "wasm-8t", "webgl", "webgpu", "webgpu-native", "webgpu-capture",
                 "--warmup", "10", "--measure", "20", "--trials", "3"],
                [["analysis/profile_summary.py", "{dir}"]]),
    "correctness": (["--mode", "correctness", "--config", *ALL, "--warmup", "2", "--trials", "1", "--no-telemetry",
                     "--exclude", "vit_large:wasm-1t"], []),
    # Main-thread availability while inference runs (no animation: DL cost, not rendering).
    "responsiveness": (["--mode", "responsiveness", "--config", "wasm-1t", "wasm-8t", "webgl", "webgpu", "webgpu-capture",
                        "--filter", "mobilenetv2", "resnet50-v2", "efficientnet", "bert_Opset17", "yolov2",
                        "--warmup", "5", "--page-baseline-s", "3", "--load-s", "8", "--trials", "3", "--no-telemetry"],
                       [["analysis/responsiveness.py", "{dir}"]]),
    "llm": (["--mode", "llm", "--config", "wasm-1t", "wasm-8t", "webgpu", "webgpu-native", "--warmup", "1", "--measure", "5",
             "--prompt-tokens", "32", "--gen-tokens", "32", "--trials", "3"],
            [["analysis/llm.py", "{dir}"]]),
    # Operating conditions: open-loop load, telemetry on (SM clock vs arrival rate).
    "load": (["--mode", "load", "--config", "wasm-8t", "webgl", "webgpu", "--filter", "mobilenetv2", "resnet50-v2",
              "bert_Opset17", "--warmup", "10", "--rates", "1", "5", "20", "50", "100", "200", "--load-s", "8",
              "--max-queue", "16", "--drain-s", "5", "--trials", "3"],
             [["analysis/load.py", "{dir}", "--slo-ms", "100", "--max-miss", "0.01"]]),
}

# Diagnostic stages: run only when named in --stages (not part of the default study).
DIAGNOSTIC = {
    # Where WebGL's per-inference time goes (CPU inside GL calls vs JS, readPixels drain vs
    # transfer, V8 profile, GPU-process trace). Needs the unminified diagnostic build on :4174:
    #   npx vite build --mode diag && npx vite preview --mode diag
    "webgl_readback": (["--mode", "profile", "--config", "webgl", "--filter", "deeplabv3p", "resnet50-v2", "yolov2",
                        "--warmup", "5", "--measure", "10", "--trials", "3", "--gl-timing", "--cpu-profile", "--trace",
                        "--url", "http://localhost:4174/"], []),
    # GPU -> JS readback cost per API path and size, without ORT (run_readback_micro.py).
    "readback_micro": (["--trials", "3", "--reps", "20"], []),
}
SCRIPT = {"readback_micro": "run_readback_micro.py"}   # default: run_benchmark.py
STAGES = {**FULL, **DIAGNOSTIC}

QUICK_OVERRIDES = {
    "latency": ["--measure", "30", "--trials", "1", "--filter", "mobilenetv2", "resnet50-v2", "bert_Opset17"],
    "batch": ["--measure", "10", "--trials", "1", "--filter", "resnet50-v2", "--batch-values", "1", "2", "4"],
    "shapes": ["--shape-rounds", "2", "--trials", "1", "--filter", "resnet50-v2"],
    "readback": ["--measure", "30", "--trials", "1", "--filter", "bert_Opset17"],
    "profile": ["--measure", "5", "--trials", "1", "--filter", "mobilenetv2", "alexnet"],
    "correctness": ["--filter", "mobilenetv2", "bert_Opset17"],
    "responsiveness": ["--load-s", "4", "--trials", "1", "--filter", "mobilenetv2"],
    "llm": ["--measure", "2", "--gen-tokens", "8", "--trials", "1"],
    "load": ["--rates", "5", "50", "--load-s", "4", "--trials", "1", "--filter", "mobilenetv2"],
}


def newest_subdir(path):
    subdirs = [p for p in Path(path).iterdir() if p.is_dir() and (p / "experiments.jsonl").exists()]
    return max(subdirs, key=lambda p: p.stat().st_mtime) if subdirs else None


def flag_values(args, flag):
    """Values after the last occurrence of `flag` (argparse: the last one wins), or None."""
    idx = [i for i, a in enumerate(args) if a == flag]
    if not idx:
        return None
    vals = []
    for a in args[idx[-1] + 1:]:
        if a.startswith("--"):
            break
        vals.append(a)
    return vals


def supplement_scope(bench_args, configs, model_names):
    """(configs, models) of the supplement that fall inside this stage's own --config / --filter."""
    cfgs = [c for c in configs if c in (flag_values(bench_args, "--config") or [])]
    stage_filter = flag_values(bench_args, "--filter")
    models = [m for m in model_names if stage_filter is None or any(f in m for f in stage_filter)]
    return cfgs, models


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=list(FULL), choices=list(STAGES))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--output-dir", default=None, help="default: benchmark_results/study_<timestamp>")
    ap.add_argument("--supplement", action="store_true",
                    help="run only --config x --models within each stage's scope and merge into its newest result")
    ap.add_argument("--config", nargs="+", default=[], help="--supplement: configs to (re)run")
    ap.add_argument("--models", nargs="+", default=[], help="--supplement: model name substrings to (re)run")
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="passed to every run_benchmark.py call")
    args = ap.parse_args()

    study = Path(args.output_dir or f"benchmark_results/study_{datetime.now():%Y%m%d_%H%M%S}")
    study.mkdir(parents=True, exist_ok=True)
    if args.supplement:
        if not (args.config and args.models):
            ap.error("--supplement needs --config and --models")
        registry = [m["name"] for m in json.loads(Path("public/models/models.json").read_text())["models"]]
        wanted = [m for m in registry if any(s in m for s in args.models)]
    summary = []
    for stage in args.stages:
        bench_args, analyses = STAGES[stage]
        bench_args = list(bench_args)
        if args.quick:
            # Later occurrences of a flag win in argparse; --filter is replaced, not merged.
            bench_args += QUICK_OVERRIDES.get(stage, [])
        stage_dir = study / stage
        main_dir = None
        if args.supplement and stage in DIAGNOSTIC:
            print(f"\n=== {stage}: diagnostic stage, not supplemented")
            continue
        if args.supplement:
            cfgs, models = supplement_scope(bench_args, args.config, wanted)
            main_dir = newest_subdir(stage_dir) if stage_dir.exists() else None
            if not cfgs or not models or main_dir is None:
                print(f"\n=== {stage}: nothing to supplement (configs {cfgs}, models {models}, main {main_dir})")
                continue
            bench_args += ["--config", *cfgs, "--filter", *models]
        script = SCRIPT.get(stage, "run_benchmark.py")
        common = COMMON if script == "run_benchmark.py" else []
        cmd = [PY, script, *bench_args, *common, "--output-dir", str(stage_dir), *args.extra]
        print(f"\n=== {stage}: {' '.join(cmd[1:])}", flush=True)
        t0 = time.time()
        rc = subprocess.call(cmd)
        result_dir = newest_subdir(stage_dir) if stage_dir.exists() else None
        if main_dir is not None:
            result_dir = merge(main_dir, result_dir, stage_dir / f"{result_dir.name}_merged") \
                if rc == 0 and result_dir and result_dir != main_dir else None
        status = "ok" if rc == 0 and result_dir else f"failed (exit {rc})"
        for a in analyses if result_dir else []:
            acmd = [PY] + [x.replace("{dir}", str(result_dir)) for x in a]
            print(f"--- analysis: {' '.join(acmd[1:])}", flush=True)
            if subprocess.call(acmd) != 0:
                status += f"; analysis {a[0]} failed"
        summary.append((stage, status, time.time() - t0, result_dir))

    print("\n=== Study summary")
    for stage, status, secs, d in summary:
        print(f"{stage:15s} {status:30s} {secs / 60:6.1f} min  {d}")


if __name__ == "__main__":
    main()
