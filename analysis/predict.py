"""
Group 9: latency prediction and configuration (backend) selection, validated on held-out models.

Everything the predictor uses is either static (from the ONNX file) or measured on this device:

  Static model features   log10 MACs (calculate_flops.count_model_macs), log10 weight bytes
                          (sum of initializer sizes), log10 node count
  Latency model           per backend: least-squares fit of log(p50 latency) on those features
                          (with a small ridge penalty; few models per backend)
  WebGL feasibility rule  static: every op must be in ORT's WebGL op table (parsed from the installed
                          onnxruntime-web bundle, with opset ranges), no int64 tensors anywhere in
                          the graph (see below), no symbolic graph-input dims in the file WebGL loads. The rule is checked against the
                          measured WebGL outcomes (rule accuracy is reported, not assumed).
                          int64 is only fatal as a graph input or with values outside the int32
                          range (e.g. INT64_MAX Slice bounds); Constant nodes are folded by the loader.
  WASM / WebGPU           predicted feasible (every tested model ran on both in strict mode); the
                          measured outcome is still what selection is scored against

Validation: leave-one-model-out. For each held-out model the latency models are refit on the
other models only, then:
  prediction error        |predicted - measured| / measured, per (model, backend)
  selection               among backends predicted feasible (and, with --slo-ms, predicted p50
                          within the SLO), pick the lowest predicted latency
  violation               the picked backend failed on this model, or its measured p50 > SLO
  regret                  measured p50 of the pick minus that of the best backend that really ran
                          (and met the SLO); 0 when the pick is the best
  calibration_s           measurement time behind the training data of each fold (sum of all
                          measured inference durations of the training models)

Usage:
    python analysis/predict.py benchmark_results/<latency run> [more latency runs ...] [--slo-ms 50]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import onnx
import pandas as pd
from onnx import TensorProto

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))
from calculate_flops import count_model_macs  # noqa: E402
from summarize import load as load_runs, summarize  # noqa: E402

FEATURES = ["log_macs", "log_weight_bytes", "log_nodes"]


def webgl_op_table():
    """[(op, domain, version selector)] from ORT's WEBGL_OP_RESOLVE_RULES in the installed bundle."""
    src = (ROOT / "node_modules/onnxruntime-web/dist/ort.all.mjs").read_text(encoding="utf-8")
    start = src.index("WEBGL_OP_RESOLVE_RULES = [")
    end = src.index("];", start)
    return re.findall(r'\["([A-Za-z]+)", "([a-z.]*)", "([0-9+-]+)"', src[start:end])


def selector_matches(version, sel):
    if sel.endswith("+"):
        return version >= int(sel[:-1])
    if "-" in sel:
        lo, hi = sel.split("-")
        return int(lo) <= version <= int(hi)
    return version == int(sel)


def static_features(model_entry, public_dir, webgl_rules):
    path = public_dir / model_entry["path"].lstrip("/")
    m = onnx.load(path)
    g = m.graph
    opsets = {o.domain or "": o.version for o in m.opset_import}
    init_names = {t.name for t in g.initializer}
    true_inputs = [i for i in g.input if i.name not in init_names]   # old opsets list initializers as inputs

    # WebGL rule components (checked before count_model_macs pins symbolic dims in place).
    # Constant nodes are folded into initializers by the WebGL backend's graph loader, so they
    # aren't looked up in the op table.
    unsupported = sorted({n.op_type for n in g.node if n.op_type != "Constant"
                          and not any(op == n.op_type and dom == (n.domain or "")
                                      and selector_matches(opsets.get(n.domain or "", 0), sel)
                                      for op, dom, sel in webgl_rules)})
    int64_input = any(i.type.tensor_type.elem_type == TensorProto.INT64 for i in true_inputs)
    lim = np.iinfo(np.int32)
    int64_tensors = [t for t in g.initializer if t.data_type == TensorProto.INT64] + \
        [a.t for n in g.node if n.op_type == "Constant" for a in n.attribute
         if a.name == "value" and a.t.data_type == TensorProto.INT64]
    int64_beyond_int32 = sum(1 for t in int64_tensors
                             if (v := onnx.numpy_helper.to_array(t)).size and (v.max() > lim.max or v.min() < lim.min))
    symbolic = any(not d.HasField("dim_value") for i in true_inputs for d in i.type.tensor_type.shape.dim)
    # WebGL loads the fixed-shape copy when the registry has one (same graph and weights, input
    # dims pinned; public/patch/pin_input_dims.py), so the dim check applies to that file.
    webgl_symbolic = symbolic
    if model_entry.get("webgl_path"):
        wg = onnx.load(public_dir / model_entry["webgl_path"].lstrip("/"), load_external_data=False).graph
        wg_inits = {t.name for t in wg.initializer}
        webgl_symbolic = any(not d.HasField("dim_value") for i in wg.input if i.name not in wg_inits
                             for d in i.type.tensor_type.shape.dim)

    macs = count_model_macs(m, {i["name"]: i["dims"] for i in model_entry["inputs"]})["macs"]
    weight_bytes = sum(onnx.numpy_helper.to_array(t).nbytes for t in g.initializer)
    return {
        "model": model_entry["name"],
        "macs": macs, "weight_bytes": weight_bytes, "nodes": len(g.node),
        "log_macs": np.log10(max(macs, 1)), "log_weight_bytes": np.log10(max(weight_bytes, 1)),
        "log_nodes": np.log10(len(g.node)),
        "webgl_unsupported_ops": ",".join(unsupported), "int64_graph_input": int64_input,
        "int64_values_beyond_int32": int64_beyond_int32, "symbolic_input_dims": symbolic,
        "webgl_path": model_entry.get("webgl_path"), "symbolic_input_dims_webgl_file": webgl_symbolic,
        "webgl_predicted_feasible": not unsupported and not int64_input and not int64_beyond_int32 and not webgl_symbolic,
    }


def fit_predict(train, test, ridge=1e-3):
    X = np.column_stack([np.ones(len(train))] + [train[f] for f in FEATURES])
    y = np.log(train["p50_ms"].to_numpy())
    A = X.T @ X + ridge * np.eye(X.shape[1])
    A[0, 0] -= ridge  # don't penalise the intercept
    w = np.linalg.solve(A, X.T @ y)
    Xt = np.column_stack([np.ones(len(test))] + [test[f] for f in FEATURES])
    return np.exp(Xt @ w)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--slo-ms", type=float, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs, sessions = load_runs(args.dirs)
    _, lat, _ = summarize(runs, sessions)
    env = json.loads((Path(args.dirs[0]) / "env.json").read_text())
    registry = Path(env["args"]["models"])
    if not registry.is_absolute():
        registry = ROOT / registry
    public = registry.resolve().parents[1]
    entries = json.loads(registry.read_text())["models"]
    tested = set(sessions["model"])
    rules = webgl_op_table()
    feats = pd.DataFrame([static_features(e, public, rules) for e in entries if e["name"] in tested])

    # Measured feasibility: a (model, backend) is feasible if any trial verified ok.
    outcome = sessions.groupby(["model", "backend"])["status"].apply(lambda s: (s == "ok").any()).rename("measured_feasible").reset_index()
    wg = outcome[outcome["backend"] == "webgl"].merge(feats[["model", "webgl_predicted_feasible", "webgl_unsupported_ops", "int64_graph_input", "int64_values_beyond_int32", "symbolic_input_dims", "symbolic_input_dims_webgl_file"]], on="model")
    rule_acc = (wg["measured_feasible"] == wg["webgl_predicted_feasible"]).mean() if len(wg) else np.nan

    measure_time = runs[runs["phase"] == "measure"].groupby("model")["dur_ms"].sum() / 1000.0
    data = lat.merge(feats, on="model")
    preds, picks = [], []
    for held in sorted(data["model"].unique()):
        cand = []
        for backend, grp in data.groupby("backend"):
            train, test = grp[grp["model"] != held], grp[grp["model"] == held]
            if len(train) < len(FEATURES) + 1:
                continue
            f = feats[feats["model"] == held]
            pred = fit_predict(train, f)[0]
            feasible_pred = bool(f["webgl_predicted_feasible"].iloc[0]) if backend == "webgl" else True
            measured = test["p50_ms"].iloc[0] if len(test) else np.nan
            preds.append({"model": held, "backend": backend, "train_models": len(train),
                          "predicted_p50_ms": pred, "measured_p50_ms": measured,
                          "ape": abs(pred - measured) / measured if len(test) else np.nan,
                          "predicted_feasible": feasible_pred})
            cand.append((backend, pred, feasible_pred, measured))
        ok = [c for c in cand if c[2] and (args.slo_ms is None or c[1] <= args.slo_ms)]
        real = [c for c in cand if not np.isnan(c[3]) and (args.slo_ms is None or c[3] <= args.slo_ms)]
        if not cand:
            continue
        pick = min(ok, key=lambda c: c[1]) if ok else None
        best = min(real, key=lambda c: c[3]) if real else None
        violation = pick is not None and (np.isnan(pick[3]) or (args.slo_ms is not None and pick[3] > args.slo_ms))
        picks.append({
            "model": held, "picked": pick[0] if pick else None, "best_feasible": best[0] if best else None,
            "picked_measured_p50_ms": pick[3] if pick else np.nan, "best_measured_p50_ms": best[3] if best else np.nan,
            "violation": violation,
            "regret_ms": (pick[3] - best[3]) if pick and best and not violation else np.nan,
            "calibration_s": measure_time.drop(held, errors="ignore").sum(),
        })
    preds, picks = pd.DataFrame(preds), pd.DataFrame(picks)
    out = Path(args.out or args.dirs[0])
    feats.to_csv(out / "model_features.csv", index=False)
    preds.to_csv(out / "prediction_lomo.csv", index=False)
    picks.to_csv(out / "selection_lomo.csv", index=False)
    wg.to_csv(out / "webgl_feasibility_rule.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 60):
        print(f"WebGL static feasibility rule vs measured: {rule_acc:.0%} agreement over {len(wg)} models")
        print(wg[["model", "measured_feasible", "webgl_predicted_feasible", "webgl_unsupported_ops", "int64_graph_input", "int64_values_beyond_int32", "symbolic_input_dims", "symbolic_input_dims_webgl_file"]].to_string(index=False))
        if len(preds):
            print("\nLeave-one-model-out latency prediction error by backend:")
            print(preds.dropna(subset=["ape"]).groupby("backend")["ape"].agg(["count", "median", "mean"]).round(3).to_string())
        if len(picks):
            print("\nBackend selection on held-out models:")
            print(picks.round(3).to_string(index=False))
            print(f"\nviolations: {int(picks['violation'].sum())}/{len(picks)}, "
                  f"mean regret: {picks['regret_ms'].mean():.3f} ms, picks matching best: {(picks['picked'] == picks['best_feasible']).mean():.0%}")


if __name__ == "__main__":
    main()
