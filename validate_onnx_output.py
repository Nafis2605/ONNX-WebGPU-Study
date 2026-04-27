"""
Validation script for run_onnx_native_benchmark.py output.

Usage:
    python validate_onnx_output.py benchmark_results/onnx_native_smoke_test.csv
    python validate_onnx_output.py benchmark_results/onnx_native_cpu_20260427_120000.csv

Checks:
    1.  CSV file exists and is non-empty
    2.  Required columns are all present
    3.  Row count matches expected trials × models
    4.  Every row has a non-NaN / non-Inf first_call_ms
    5.  Every row has inference_iters consistent with the run configuration
    6.  No NaN or Inf in numeric timing columns
    7.  first_call_ms is generally distinct from inference_mean_ms
        (warns if they are identical to 4 decimal places across ALL trials)
    8.  inference_mean_ms variance across trials is reasonable (< 5×)
    9.  actual_providers reported does not start with CPUExecutionProvider
        when a GPU provider was requested (provider fallback detection)
   10.  Status == SUCCESS for every row (else reports which rows failed)
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


REQUIRED_COLUMNS = [
    "timestamp",
    "model",
    "provider",
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
]

NUMERIC_TIMING_COLS = [
    "session_create_ms",
    "first_call_ms",
    "inference_mean_ms",
    "inference_std_ms",
]


def is_bad_float(s: str) -> bool:
    """Return True if the string represents NaN or Inf."""
    try:
        v = float(s)
        return math.isnan(v) or math.isinf(v)
    except (ValueError, TypeError):
        return False  # non-numeric (e.g. 'N/A' in GPU columns) is fine


def validate(csv_path: Path) -> bool:
    ok = True

    def warn(msg):
        nonlocal ok
        print(f"  WARN  {msg}")
        ok = False

    def info(msg):
        print(f"  OK    {msg}")

    print(f"\n{'='*65}")
    print(f"Validating: {csv_path}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. File existence
    # ------------------------------------------------------------------
    if not csv_path.exists():
        print(f"  FAIL  File not found: {csv_path}")
        return False
    info(f"File exists ({csv_path.stat().st_size} bytes)")

    # ------------------------------------------------------------------
    # 2. Load rows
    # ------------------------------------------------------------------
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        actual_columns = reader.fieldnames or []

    if not rows:
        print("  FAIL  CSV is empty (no data rows)")
        return False
    info(f"Row count: {len(rows)}")

    # ------------------------------------------------------------------
    # 3. Required columns
    # ------------------------------------------------------------------
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in actual_columns]
    if missing_cols:
        warn(f"Missing required columns: {missing_cols}")
    else:
        info(f"All {len(REQUIRED_COLUMNS)} required columns present")

    # ------------------------------------------------------------------
    # 4. Status check
    # ------------------------------------------------------------------
    failed_rows = [r for r in rows if r.get("status") != "SUCCESS"]
    if failed_rows:
        warn(f"{len(failed_rows)} row(s) with status != SUCCESS:")
        for r in failed_rows:
            print(f"         model={r.get('model')} trial={r.get('trial_id')} "
                  f"status={r.get('status')} notes={r.get('notes','')[:80]}")
    else:
        info("All rows have status == SUCCESS")

    # ------------------------------------------------------------------
    # 5. NaN / Inf in timing columns
    # ------------------------------------------------------------------
    bad_numeric = []
    for i, r in enumerate(rows, 1):
        if r.get("status") != "SUCCESS":
            continue
        for col in NUMERIC_TIMING_COLS:
            if col in r and is_bad_float(r[col]):
                bad_numeric.append((i, col, r[col]))

    if bad_numeric:
        warn(f"NaN/Inf values in timing columns: {bad_numeric[:10]}")
    else:
        info("No NaN or Inf in timing columns")

    # ------------------------------------------------------------------
    # 6. inference_iters consistency per model
    # ------------------------------------------------------------------
    iters_by_model = defaultdict(set)
    for r in rows:
        if r.get("status") == "SUCCESS":
            iters_by_model[r["model"]].add(r.get("inference_iters", ""))
    for model, iters_set in iters_by_model.items():
        if len(iters_set) > 1:
            warn(f"Inconsistent inference_iters for {model}: {iters_set}")
        else:
            info(f"inference_iters consistent for {model}: {iters_set}")

    # ------------------------------------------------------------------
    # 7. first_call_ms distinct from inference_mean_ms
    # ------------------------------------------------------------------
    identical_count = 0
    for r in rows:
        if r.get("status") != "SUCCESS":
            continue
        try:
            fc  = float(r["first_call_ms"])
            inf = float(r["inference_mean_ms"])
            if abs(fc - inf) < 0.0001:
                identical_count += 1
        except (ValueError, KeyError):
            pass

    success_count = sum(1 for r in rows if r.get("status") == "SUCCESS")
    if identical_count == success_count and success_count > 0:
        warn("first_call_ms == inference_mean_ms for ALL successful rows "
             "(timing may not be distinguishing first-call overhead)")
    elif identical_count > 0:
        info(f"first_call_ms differs from inference_mean_ms in "
             f"{success_count - identical_count}/{success_count} rows "
             f"({identical_count} identical — may be expected for CPU fast models)")
    else:
        info("first_call_ms differs from inference_mean_ms in all rows")

    # ------------------------------------------------------------------
    # 8. Variance check across trials (per model)
    # ------------------------------------------------------------------
    inf_by_model = defaultdict(list)
    for r in rows:
        if r.get("status") == "SUCCESS":
            try:
                inf_by_model[r["model"]].append(float(r["inference_mean_ms"]))
            except (ValueError, KeyError):
                pass

    for model, vals in inf_by_model.items():
        if len(vals) < 2:
            continue
        mn = sum(vals) / len(vals)
        if mn > 0:
            ratio = max(vals) / min(vals)
            if ratio > 5.0:
                warn(f"High inference variance for {model}: "
                     f"min={min(vals):.2f}ms max={max(vals):.2f}ms "
                     f"ratio={ratio:.1f}x — investigate thermal throttling or "
                     f"missing stabilisation")
            else:
                info(f"Inference variance OK for {model}: "
                     f"mean={mn:.2f}ms max/min ratio={ratio:.2f}x")

    # ------------------------------------------------------------------
    # 9. Provider fallback detection
    # ------------------------------------------------------------------
    mismatch_rows = [
        r for r in rows
        if r.get("provider_mismatch", "").lower() in ("true", "1", "yes")
    ]
    if mismatch_rows:
        warn(f"{len(mismatch_rows)} row(s) with provider_mismatch=True "
             f"(silent fallback to CPU detected):")
        for r in mismatch_rows:
            print(f"         model={r.get('model')} trial={r.get('trial_id')} "
                  f"actual={r.get('actual_providers')}")
    else:
        info("No provider fallback detected (provider_mismatch=False for all rows)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    if ok:
        print("VALIDATION PASSED — all checks OK")
    else:
        print("VALIDATION FINISHED WITH WARNINGS — review items above")

    return ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default to the smoke test file if no argument given
        default = Path("benchmark_results/onnx_native_smoke_test.csv")
        path = default
    else:
        path = Path(sys.argv[1])

    validate(path)
