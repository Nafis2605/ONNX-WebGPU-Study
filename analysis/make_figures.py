"""
Generate every study figure (PNG + PDF) and findings.json from a run_study.py output directory.

Figures follow the study's narrative:
    §0 validity   §1 coverage   §2 when   §3 why   §4 cost   §5 conditions   §6 correctness   §7 prediction

Usage:
    python analysis/make_figures.py benchmark_results/study_<timestamp> [--out study_report/figures]
"""
import argparse
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figures import style  # noqa: E402
from figures.data import Study  # noqa: E402
from figures import f_validity, f_coverage, f_when, f_why, f_cost, f_conditions, f_correct_predict, f_readback  # noqa: E402

STEPS = [
    ("0_validity", f_validity.validity),
    ("1_support_matrix", f_coverage.support_matrix),
    ("2a_speedup_vs_compute", f_when.speedup_vs_compute),
    ("2b_latency_ecdf", f_when.latency_ecdf),
    ("2c_tail_ratio", f_when.tail_ratio),
    ("2d_batch_scaling", f_when.batch_scaling),
    ("3a_time_decomposition", f_why.time_decomposition),
    ("3b_dispatch_overhead", f_why.dispatch_overhead),
    ("3c_op_speedup", f_why.op_speedup),
    ("3d_runtime_variants", f_why.runtime_variants),
    ("3e_readback", f_why.readback),
    ("3f_webgl_readback", f_readback.webgl_readback),
    ("4a_startup_costs", f_cost.startup_costs),
    ("4b_break_even", f_cost.break_even),
    ("4c_shape_penalty", f_cost.shape_penalty),
    ("5a_idle_penalty", f_conditions.idle_penalty),
    ("5b_goodput", f_conditions.goodput),
    ("5c_cpu_cost", f_conditions.cpu_cost),
    ("5d_energy", f_conditions.energy),
    ("5e_main_thread", f_conditions.main_thread),
    ("5f_llm", f_conditions.llm),
    ("6_correctness", f_correct_predict.correctness),
    ("7_prediction", f_correct_predict.prediction),
]


def _jsonable(o):
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and o != o:
        return None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study")
    ap.add_argument("--out", default=None, help="default: <study>/figures")
    ap.add_argument("--only", nargs="+", default=None, help="run only steps whose name contains these substrings")
    args = ap.parse_args()
    style.apply()
    study = Study(args.study)
    out = Path(args.out or Path(args.study) / "figures")
    out.mkdir(parents=True, exist_ok=True)
    print(f"stages: {', '.join(study.stages)}")
    findings = {"study": str(Path(args.study).resolve()), "stages": {k: str(v.path) for k, v in study.stages.items()}}
    for name, fn in STEPS:
        if args.only and not any(o in name for o in args.only):
            continue
        try:
            findings[name] = fn(study, out)
            print(f"ok     {name}")
        except Exception as e:
            findings[name] = {"error": f"{type(e).__name__}: {e}"}
            print(f"FAILED {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
    (out / "findings.json").write_text(json.dumps(_jsonable(findings), indent=1, default=str), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
