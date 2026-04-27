"""
Native ONNX Runtime Benchmark
==============================
Measures per-trial first-call latency and steady-state inference latency
for ONNX models using onnxruntime directly (no browser, no Playwright).

Per-trial protocol
------------------
  Phase 1 — Setup        : Create fresh InferenceSession; prepare dummy inputs
                            (both outside timed inference region).
  Phase 2 — 1st-Call     : Exactly one timed sess.run() -> first_call_ms.
                            Captures one-time runtime overhead (graph execution
                            engine warmup, kernel selection, memory allocation).
  Phase 3 — Stabilization: --stabilization untimed sess.run() calls.
                            Not reported; exists purely to let any late JIT or
                            caching settle before the measurement window.
  Phase 4 — Measured loop: --measure-iters timed sess.run() calls.
                            Yields inference_mean_ms and inference_std_ms.
  Phase 5 — Cleanup      : Session deleted; optional cooldown sleep.

Synchronization
---------------
  CPUExecutionProvider : sess.run() is blocking and fully synchronous.
                         When it returns, all computation is complete and
                         results reside in NumPy arrays.  Direct
                         time.perf_counter() wrapping is correct.

  CUDAExecutionProvider: By default (no explicit IOBinding), ORT copies
                         device outputs back to host (CPU) NumPy arrays
                         before sess.run() returns.  The call therefore
                         blocks until the device pipeline completes.
                         Direct time.perf_counter() wrapping is correct;
                         no external cudaDeviceSynchronize is required.

  DmlExecutionProvider : Same semantics as CUDA — outputs are synchronised
                         to CPU before the call returns.

Usage
-----
  python run_onnx_native_benchmark.py \\
      --models public/models/models.json \\
      --provider cpu \\
      --trials 5 \\
      --stabilization 3 \\
      --measure-iters 20

Requirements
------------
  pip install onnxruntime          # CPU
  pip install onnxruntime-gpu      # CUDA
  pip install numpy
"""

import argparse
import csv
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime is not installed.")
    print("  CPU:  pip install onnxruntime")
    print("  CUDA: pip install onnxruntime-gpu")
    sys.exit(1)

# Optional: GPU metrics collection (reuses existing gpu_metrics.py)
try:
    from gpu_metrics import GPUMetricsCollector
    _GPU_METRICS_AVAILABLE = True
except ImportError:
    _GPU_METRICS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Constants and mappings
# ---------------------------------------------------------------------------

#: Maps the CLI provider short-name to the ORT provider string.
PROVIDER_MAP: dict = {
    "cpu":    "CPUExecutionProvider",
    "cuda":   "CUDAExecutionProvider",
    "dml":    "DmlExecutionProvider",
    "trt":    "TensorrtExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}

#: Maps models.json dtype strings to NumPy dtypes.
DTYPE_MAP: dict = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "int8":    np.int8,
    "int16":   np.int16,
    "int32":   np.int32,
    "int64":   np.int64,
    "uint8":   np.uint8,
    "uint16":  np.uint16,
    "uint32":  np.uint32,
    "uint64":  np.uint64,
    "bool":    np.bool_,
}

#: Ordered CSV columns written for every row.
CSV_FIELDS = [
    "timestamp",
    "model",
    "provider",        # requested provider key (e.g. "cuda")
    "provider_used",   # first active provider reported by sess.get_providers()
    "trial_id",
    "session_create_ms",
    "first_call_ms",
    "stabilization_iters",
    "inference_mean_ms",
    "inference_std_ms",
    "inference_iters",
    "status",
    "provider_mismatch",
    "actual_providers",
    "notes",
    # GPU metrics from gpu_metrics.py (appended when available)
    "gpu_vendor",
    "gpu_name",
    "gpu_utilization_percent",
    "gpu_memory_utilization_percent",
    "gpu_power_draw_watts",
]


# ---------------------------------------------------------------------------
# Input generation
# ---------------------------------------------------------------------------

def _resolve_dim(d) -> int:
    """Convert a dynamic/symbolic dimension to a concrete integer (1)."""
    if isinstance(d, int) and d > 0:
        return d
    return 1  # symbolic or zero-valued dims become 1


def make_dummy_inputs_from_session(
    sess: "ort.InferenceSession",
    json_specs: list,
    seed: int = 42,
) -> dict:
    """
    Build {input_name: np.ndarray} by introspecting the live ORT session.

    Why session introspection instead of models.json only
    -------------------------------------------------------
    The models.json input names sometimes diverge from the actual graph input
    names embedded in the .onnx file (e.g. models.json lists "input" but the
    file uses "x").  sess.get_inputs() is always authoritative.

    Strategy
    --------
    1.  Use sess.get_inputs() to get the ground-truth input names, ORT element
        types, and declared shapes.
    2.  For each input, if models.json has a matching spec (by name) with
        concrete dims, prefer those dims (they may fix dynamic 0-dims).
    3.  Fall back to session-declared shapes, replacing any dynamic dims with 1.

    Parameters
    ----------
    sess      : live ort.InferenceSession
    json_specs: list of {"name", "type", "dims"} from models.json (may be
                empty or have wrong names — handled gracefully)
    seed      : RNG seed for reproducibility

    Returns
    -------
    dict {input_name: np.ndarray}
    """
    # Build a lookup of models.json specs keyed by input name
    json_by_name: dict = {s["name"]: s for s in json_specs}

    # ORT element type string -> NumPy dtype
    ORT_ETYPE_TO_NP: dict = {
        "tensor(float)":   np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)":  np.float64,
        "tensor(int8)":    np.int8,
        "tensor(int16)":   np.int16,
        "tensor(int32)":   np.int32,
        "tensor(int64)":   np.int64,
        "tensor(uint8)":   np.uint8,
        "tensor(uint16)":  np.uint16,
        "tensor(uint32)":  np.uint32,
        "tensor(uint64)":  np.uint64,
        "tensor(bool)":    np.bool_,
        "tensor(string)":  np.object_,
    }

    rng = np.random.default_rng(seed)
    feeds: dict = {}

    for inp in sess.get_inputs():
        name: str = inp.name
        ort_type: str = inp.type  # e.g. "tensor(float)"
        np_dtype = ORT_ETYPE_TO_NP.get(ort_type, np.float32)

        # Determine shape: prefer models.json concrete dims, then session dims
        if name in json_by_name:
            raw_dims = json_by_name[name].get("dims", inp.shape or [1])
            # Also pick up dtype from json if it's more specific
            json_dtype_str = json_by_name[name].get("type", "")
            if json_dtype_str in DTYPE_MAP:
                np_dtype = DTYPE_MAP[json_dtype_str]
        else:
            raw_dims = inp.shape or [1]

        shape = tuple(_resolve_dim(d) for d in raw_dims)

        if np_dtype == np.object_:
            # String tensors: fill with empty bytes
            feeds[name] = np.full(shape, b"", dtype=np.object_)
        elif np.issubdtype(np_dtype, np.floating):
            feeds[name] = rng.random(shape).astype(np_dtype)
        elif np.issubdtype(np_dtype, np.integer):
            feeds[name] = rng.integers(0, 100, size=shape).astype(np_dtype)
        elif np_dtype == np.bool_:
            feeds[name] = rng.integers(0, 2, size=shape).astype(np.bool_)
        else:
            feeds[name] = rng.random(shape).astype(np_dtype)

    return feeds


# ---------------------------------------------------------------------------
# ORT session helpers
# ---------------------------------------------------------------------------

def _build_session_options() -> "ort.SessionOptions":
    """
    Standard SessionOptions for reproducible benchmarking.
    Graph optimisation is fully enabled (matches production use).
    """
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    # Disable verbose logging to avoid timing interference
    opts.log_severity_level = 3  # ERROR only
    return opts


def _build_provider_list(provider_key: str) -> list:
    """
    Return an ORT provider list for the requested key.
    CPUExecutionProvider is always appended as a fallback.
    """
    if provider_key not in PROVIDER_MAP:
        raise ValueError(
            f"Unknown provider '{provider_key}'. "
            f"Valid choices: {list(PROVIDER_MAP.keys())}"
        )
    primary = PROVIDER_MAP[provider_key]
    if primary == "CPUExecutionProvider":
        return ["CPUExecutionProvider"]
    if primary == "CUDAExecutionProvider":
        # Pass explicit CUDA options so device selection is unambiguous.
        # cudnn_conv_algo_search=HEURISTIC: cuDNN picks an algorithm via
        # heuristic on the first call (avoids exhaustive search overhead
        # while still causing measurable first-call JIT latency).
        cuda_opts = {
            "device_id": 0,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": True,
        }
        return [("CUDAExecutionProvider", cuda_opts), "CPUExecutionProvider"]
    return [primary, "CPUExecutionProvider"]


def _check_provider_active(
    sess: "ort.InferenceSession",
    requested_key: str,
) -> tuple:
    """
    Verify whether the session's primary active provider matches the request.

    Returns
    -------
    (actual_providers_str, mismatch_detected)
        actual_providers_str : comma-separated list of session providers
        mismatch_detected    : True if primary provider != requested
    """
    requested_ort = PROVIDER_MAP[requested_key]
    active = sess.get_providers()
    primary_active = active[0] if active else "unknown"
    mismatch = primary_active != requested_ort
    # Returns: all_providers_str, primary_active_provider, mismatch_bool
    return ",".join(active), primary_active, mismatch


# ---------------------------------------------------------------------------
# Core per-trial benchmark
# ---------------------------------------------------------------------------

def _run_trial(
    model_path: str,
    input_specs: list,
    provider_key: str,
    stabilization_iters: int,
    measure_iters: int,
    trial_id: int,
) -> dict:
    """
    Execute one complete benchmark trial for a single model.

    Phases
    ------
    1. Session creation + input preparation  (outside timed region)
    2. 1st-Call: exactly one timed sess.run()
    3. Stabilization: `stabilization_iters` untimed sess.run() calls
    4. Measured loop: `measure_iters` timed sess.run() calls
    5. Cleanup: session deleted

    Returns
    -------
    dict with keys:
        trial_id, session_create_ms, first_call_ms,
        stabilization_iters, inference_mean_ms, inference_std_ms,
        inference_iters, status, notes, provider_mismatch, actual_providers
    """
    result = {
        "trial_id": trial_id,
        "session_create_ms": float("nan"),
        "first_call_ms": float("nan"),
        "stabilization_iters": stabilization_iters,
        "inference_mean_ms": float("nan"),
        "inference_std_ms": float("nan"),
        "inference_iters": measure_iters,
        "status": "FAILED",
        "notes": "",
        "provider_mismatch": False,
        "provider_used": "",
        "actual_providers": "",
    }

    sess = None
    try:
        # ----------------------------------------------------------------
        # Phase 1 — Setup (not timed as inference; session_create_ms tracked
        #            separately as metadata)
        # ----------------------------------------------------------------
        opts = _build_session_options()
        providers = _build_provider_list(provider_key)

        t0 = time.perf_counter()
        sess = ort.InferenceSession(
            model_path, sess_options=opts, providers=providers
        )
        t1 = time.perf_counter()
        result["session_create_ms"] = (t1 - t0) * 1000.0

        # Verify actual provider (detect silent fallback)
        actual_str, provider_used, mismatch = _check_provider_active(sess, provider_key)
        result["actual_providers"] = actual_str
        result["provider_used"] = provider_used
        result["provider_mismatch"] = mismatch
        if mismatch:
            requested_ort = PROVIDER_MAP[provider_key]
            result["notes"] = (
                f"Provider fallback detected: requested={requested_ort}, "
                f"active_primary={provider_used}"
            )

        # Prepare dummy inputs outside the timed region.
        # Uses session introspection so input names always match the .onnx file.
        # Seed is offset per trial for variety while remaining deterministic.
        feeds = make_dummy_inputs_from_session(sess, input_specs, seed=42 + trial_id)

        # ----------------------------------------------------------------
        # Phase 2 — 1st-Call measurement
        #
        # Exactly one timed inference executed on a fresh session.
        # Captures: runtime graph preparation, kernel selection, memory
        # allocation, and any provider-specific JIT startup overhead.
        #
        # Synchronisation note:
        #   CPUExecutionProvider : sess.run() is fully synchronous.
        #   CUDAExecutionProvider: outputs are copied back to CPU host
        #     NumPy arrays before the call returns, making it effectively
        #     synchronous.  No explicit cudaDeviceSynchronize needed.
        #   DmlExecutionProvider : same semantics as CUDA.
        # ----------------------------------------------------------------
        t0 = time.perf_counter()
        _ = sess.run(None, feeds)
        t1 = time.perf_counter()
        result["first_call_ms"] = (t1 - t0) * 1000.0

        # ----------------------------------------------------------------
        # Phase 3 — Stabilization (untimed, not reported)
        # ----------------------------------------------------------------
        for _ in range(stabilization_iters):
            _ = sess.run(None, feeds)

        # ----------------------------------------------------------------
        # Phase 4 — Measured inference loop
        # ----------------------------------------------------------------
        iter_times: list = []
        for _ in range(measure_iters):
            t0 = time.perf_counter()
            _ = sess.run(None, feeds)
            t1 = time.perf_counter()
            iter_times.append((t1 - t0) * 1000.0)

        result["inference_mean_ms"] = statistics.mean(iter_times)
        result["inference_std_ms"] = (
            statistics.stdev(iter_times) if len(iter_times) > 1 else 0.0
        )
        result["inference_iters"] = len(iter_times)
        result["status"] = "SUCCESS"

    except Exception as exc:
        result["notes"] += f" | Exception: {str(exc)[:400]}"
        result["status"] = "FAILED"

    finally:
        # ----------------------------------------------------------------
        # Phase 5 — Cleanup
        # ----------------------------------------------------------------
        del sess  # Release session resources before next trial

    return result


# ---------------------------------------------------------------------------
# Model-level orchestration
# ---------------------------------------------------------------------------

def _fmt_float(v) -> str:
    """Format a float to 4 decimal places; return 'NaN' for NaN values."""
    if isinstance(v, float):
        if v != v:  # IEEE NaN
            return "NaN"
        return f"{v:.4f}"
    return str(v)


def benchmark_model(
    model_name: str,
    model_path: str,
    input_specs: list,
    provider_key: str,
    trials: int,
    stabilization_iters: int,
    measure_iters: int,
    cooldown_sec: float,
    gpu_collector,
    fail_on_fallback: bool = False,
) -> list:
    """
    Run `trials` independent benchmark trials for one model.

    Parameters
    ----------
    gpu_collector : GPUMetricsCollector or None
        If not None, GPU metrics are collected during each trial.

    Returns
    -------
    list[dict]
        One dict per trial, ready to be written as a CSV row.
    """
    rows: list = []

    for trial_id in range(1, trials + 1):
        print(f"  Trial {trial_id}/{trials} ...", end="", flush=True)

        # Start GPU monitoring
        if gpu_collector is not None:
            gpu_collector.start_monitoring()

        trial = _run_trial(
            model_path=model_path,
            input_specs=input_specs,
            provider_key=provider_key,
            stabilization_iters=stabilization_iters,
            measure_iters=measure_iters,
            trial_id=trial_id,
        )

        # Stop GPU monitoring and collect metrics.
        # gpu_metrics.py's stop_monitoring() returns the averaged metrics dict
        # directly; there is no separate get_metrics() method.
        gpu_metrics: dict = {}
        if gpu_collector is not None:
            gpu_metrics = gpu_collector.stop_monitoring()
            gpu_collector.clear_interval_samples()

        # Console progress
        if trial["status"] == "SUCCESS":
            print(
                f"  first_call={trial['first_call_ms']:.3f} ms  "
                f"inference={trial['inference_mean_ms']:.3f}"
                f"±{trial['inference_std_ms']:.3f} ms"
            )
        else:
            print(f"  FAILED — {trial['notes'][:80]}")

        # Detect and warn about provider fallback
        if trial["provider_mismatch"]:
            print(f"  WARNING: {trial['notes']}")
            if fail_on_fallback:
                print("  ABORT: --fail-on-fallback is set. Exiting.")
                sys.exit(2)

        # Build CSV row
        row: dict = {
            "timestamp":            datetime.now().isoformat(),
            "model":                model_name,
            "provider":             provider_key,
            "provider_used":        trial["provider_used"],
            "trial_id":             trial["trial_id"],
            "session_create_ms":    _fmt_float(trial["session_create_ms"]),
            "first_call_ms":        _fmt_float(trial["first_call_ms"]),
            "stabilization_iters":  trial["stabilization_iters"],
            "inference_mean_ms":    _fmt_float(trial["inference_mean_ms"]),
            "inference_std_ms":     _fmt_float(trial["inference_std_ms"]),
            "inference_iters":      trial["inference_iters"],
            "status":               trial["status"],
            "provider_mismatch":    trial["provider_mismatch"],
            "actual_providers":     trial["actual_providers"],
            "notes":                trial["notes"],
        }
        row.update(gpu_metrics)
        rows.append(row)

        # Inter-trial cooldown (not applied after the last trial of the model)
        if cooldown_sec > 0 and trial_id < trials:
            time.sleep(cooldown_sec)

    return rows


# ---------------------------------------------------------------------------
# CSV writing
# ---------------------------------------------------------------------------

def save_results(rows: list, output_path: Path) -> None:
    """Write all result rows to a CSV file.

    The column order follows CSV_FIELDS; any extra GPU metric columns discovered
    at runtime are appended after the predefined columns.
    """
    if not rows:
        print("No results to save.")
        return

    # Union of all keys, preserving CSV_FIELDS order then extras
    extra_keys: list = []
    for row in rows:
        for k in row:
            if k not in CSV_FIELDS and k not in extra_keys:
                extra_keys.append(k)
    all_fields = CSV_FIELDS + extra_keys

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "N/A") for k in all_fields})

    print(f"\nResults saved -> {output_path}")
    print(f"Rows written  : {len(rows)}")
    print(f"Columns       : {len(all_fields)}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(rows: list) -> None:
    """Print a per-model aggregated summary after all trials complete."""
    model_data: dict = defaultdict(list)
    for row in rows:
        if row.get("status") == "SUCCESS":
            model_data[row["model"]].append(row)

    if not model_data:
        print("\nNo successful trials to summarise.")
        return

    hdr = f"{'Model':<48} {'Provider(used)':<14} {'N':>3}  {'1st-Call ms':>22}  {'Inference ms':>22}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))

    for model, trials in model_data.items():
        provider = trials[0]["provider"]
        used = trials[0].get("provider_used", provider)
        provider_label = used if used and used != provider else provider
        n = len(trials)

        fc_vals  = [float(t["first_call_ms"])     for t in trials]
        inf_vals = [float(t["inference_mean_ms"]) for t in trials]

        fc_mean  = statistics.mean(fc_vals)
        fc_std   = statistics.stdev(fc_vals)  if n > 1 else 0.0
        inf_mean = statistics.mean(inf_vals)
        inf_std  = statistics.stdev(inf_vals) if n > 1 else 0.0

        # Show which provider actually ran (may differ from requested)
        prov_display = provider_label[:12]
        print(
            f"{model:<48} {prov_display:<14} {n:>3}  "
            f"{fc_mean:>9.2f} +/- {fc_std:>7.2f} ms  "
            f"{inf_mean:>9.2f} +/- {inf_std:>7.2f} ms"
        )

    print("=" * len(hdr))


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Native ONNX Runtime benchmark — first-call and steady-state latency.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Providers: cpu | cuda | dml | trt | coreml

Examples
--------
  # CPU — all models, 5 trials, 20 measured iterations:
  python run_onnx_native_benchmark.py --models public/models/models.json --provider cpu

  # CUDA — single model, 10 trials, 50 measured iterations:
  python run_onnx_native_benchmark.py \\
      --models public/models/models.json \\
      --provider cuda \\
      --trials 10 --measure-iters 50 \\
      --filter alexnet

  # List which providers are available on this machine:
  python run_onnx_native_benchmark.py --list-providers

  # List all registered models (checks if .onnx files exist):
  python run_onnx_native_benchmark.py --models public/models/models.json --list-models
        """,
    )

    p.add_argument("--models", "-m", required=False,
                   help="Path to models.json registry (e.g. public/models/models.json)")
    p.add_argument("--provider", "-p", default="cpu",
                   choices=list(PROVIDER_MAP.keys()),
                   help="Execution provider (default: cpu)")
    p.add_argument("--trials", "-t", type=int, default=5,
                   help="Independent trials per model (default: 5)")
    p.add_argument("--stabilization", type=int, default=3,
                   help="Untimed stabilisation iterations before measured loop (default: 3)")
    p.add_argument("--measure-iters", "-r", type=int, default=20,
                   help="Timed inference iterations per trial (default: 20)")
    p.add_argument("--cooldown-sec", type=float, default=1.0,
                   help="Cooldown between trials in seconds (default: 1.0)")
    p.add_argument("--models-dir", default="public/models",
                   help="Directory containing .onnx files (default: public/models)")
    p.add_argument("--output-dir", "-o", default="benchmark_results",
                   help="Output directory for CSV (default: benchmark_results)")
    p.add_argument("--filename", "-n", default=None,
                   help="Custom CSV filename (auto-generated if omitted)")
    p.add_argument("--filter", nargs="+", default=None,
                   help="Only benchmark models whose name contains one of these strings "
                        "(case-insensitive)")
    p.add_argument("--list-models", action="store_true",
                   help="List models from registry and check file existence, then exit")
    p.add_argument("--list-providers", action="store_true",
                   help="List ORT providers available on this system, then exit")
    p.add_argument("--fail-on-fallback", action="store_true", default=False,
                   help="Exit with error if requested provider is unavailable or "
                        "falls back to CPU (default: warn and continue)")

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_arg_parser().parse_args()

    # --list-providers: no models.json needed
    if args.list_providers:
        available = ort.get_available_providers()
        print(f"ORT version : {ort.__version__}")
        print("Available providers on this system:")
        for prov in available:
            print(f"  {prov}")
        return

    # Require --models for everything else
    if not args.models:
        print("ERROR: --models is required (unless using --list-providers).")
        sys.exit(1)

    # Load model registry
    models_json_path = Path(args.models)
    if not models_json_path.exists():
        print(f"ERROR: models.json not found: {models_json_path}")
        sys.exit(1)

    with open(models_json_path, "r", encoding="utf-8") as fh:
        registry = json.load(fh)

    models_dir = Path(args.models_dir)
    all_models: list = registry.get("models", [])

    # --list-models
    if args.list_models:
        print(f"Models in registry ({models_json_path}):")
        for m in all_models:
            onnx_filename = Path(m["path"]).name
            onnx_path = models_dir / onnx_filename
            status = "OK" if onnx_path.exists() else "MISSING"
            print(f"  [{status}] {m['name']:<50}  {onnx_path}")
        return

    # Filter models
    if args.filter:
        models_to_run = [
            m for m in all_models
            if any(f.lower() in m["name"].lower() for f in args.filter)
        ]
        print(f"Filtered to {len(models_to_run)} model(s) matching: {args.filter}")
    else:
        models_to_run = all_models

    if not models_to_run:
        print("No models selected. Check --filter or run --list-models.")
        sys.exit(1)

    # Validate provider availability
    requested_ort_provider = PROVIDER_MAP[args.provider]
    available_providers = ort.get_available_providers()
    provider_available = requested_ort_provider in available_providers

    if not provider_available:
        lines = [
            f"",
            f"{'ERROR' if args.fail_on_fallback else 'WARNING'}: "
            f"Requested provider '{requested_ort_provider}' is NOT in "
            f"ort.get_available_providers().",
            f"  Available : {available_providers}",
        ]
        if args.provider == "cuda":
            lines += [
                "  Fix: install onnxruntime-gpu (not the CPU-only onnxruntime wheel):",
                "    pip uninstall onnxruntime -y",
                "    pip install onnxruntime-gpu",
                "  Your NVIDIA driver must support CUDA 12 (driver >= 525.60).",
            ]
        print("\n".join(lines))
        if args.fail_on_fallback:
            sys.exit(1)
        print("  Continuing with CPU fallback. Use --fail-on-fallback to abort.\n")

    # GPU metrics collector (reuses gpu_metrics.py)
    gpu_collector = None
    if _GPU_METRICS_AVAILABLE:
        gpu_collector = GPUMetricsCollector()
        gpu_name = gpu_collector.metrics.get("gpu_name", "N/A")
        print(f"GPU metrics  : enabled ({gpu_name})")
    else:
        print("GPU metrics  : disabled (gpu_metrics.py not importable)")

    # Collect CUDA device info for the header (requires nvidia-smi)
    cuda_device_info = "N/A"
    if args.provider == "cuda":
        try:
            import subprocess as _sp
            _r = _sp.run(
                ["nvidia-smi",
                 "--query-gpu=name,driver_version,memory.total,compute_cap",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if _r.returncode == 0:
                cuda_device_info = _r.stdout.strip().split("\n")[0].strip()
        except Exception:
            pass

    # Print configuration header
    print("\n" + "=" * 70)
    print("Native ONNX Runtime Benchmark")
    print("=" * 70)
    print(f"  Registry        : {models_json_path}")
    print(f"  Models dir      : {models_dir}")
    print(f"  Provider        : {args.provider}  ({requested_ort_provider})")
    print(f"  Provider avail  : {provider_available}")
    print(f"  Fail on fallbk  : {args.fail_on_fallback}")
    print(f"  CUDA device     : {cuda_device_info}")
    print(f"  Trials          : {args.trials}")
    print(f"  Stabilisation   : {args.stabilization} iters (untimed)")
    print(f"  Measure iters   : {args.measure_iters} per trial")
    print(f"  Cooldown        : {args.cooldown_sec}s between trials")
    print(f"  ORT version     : {ort.__version__}")
    print(f"  Available provs : {available_providers}")
    print("=" * 70 + "\n")

    all_rows: list = []

    for idx, model_cfg in enumerate(models_to_run, 1):
        model_name = model_cfg["name"]
        onnx_filename = Path(model_cfg["path"]).name
        model_path = str(models_dir / onnx_filename)
        input_specs = model_cfg.get("inputs", [])

        print(f"\n[{idx}/{len(models_to_run)}] {model_name}")
        print(f"  File  : {model_path}")
        print(f"  Inputs: {[s['name'] for s in input_specs]}")

        if not Path(model_path).exists():
            print(f"  ERROR : model file not found — skipping.")
            all_rows.append({
                "timestamp":  datetime.now().isoformat(),
                "model":      model_name,
                "provider":   args.provider,
                "trial_id":   0,
                "status":     "MISSING",
                "notes":      f"File not found: {model_path}",
            })
            continue

        rows = benchmark_model(
            model_name=model_name,
            model_path=model_path,
            input_specs=input_specs,
            provider_key=args.provider,
            trials=args.trials,
            stabilization_iters=args.stabilization,
            measure_iters=args.measure_iters,
            cooldown_sec=args.cooldown_sec,
            gpu_collector=gpu_collector,
            fail_on_fallback=args.fail_on_fallback,
        )
        all_rows.extend(rows)

        # Inter-model gap (2× the inter-trial cooldown)
        if idx < len(models_to_run) and args.cooldown_sec > 0:
            time.sleep(args.cooldown_sec * 2)

    # Determine output path
    output_dir = Path(args.output_dir)
    if args.filename:
        output_path = output_dir / args.filename
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"onnx_native_{args.provider}_{ts}.csv"

    save_results(all_rows, output_path)
    print_summary(all_rows)

    # Exit summary counts
    successful = sum(1 for r in all_rows if r.get("status") == "SUCCESS")
    failed = sum(1 for r in all_rows if r.get("status") not in ("SUCCESS", "MISSING"))
    missing = sum(1 for r in all_rows if r.get("status") == "MISSING")
    print(f"\nTotal rows : {len(all_rows)}")
    print(f"Successful : {successful}")
    print(f"Failed     : {failed}")
    print(f"Missing    : {missing}")


if __name__ == "__main__":
    main()
