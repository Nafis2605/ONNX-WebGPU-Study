"""Group 10: compare browser outputs with native ONNX Runtime (CPU EP) on the identical inputs.

The reference runs the same model file with the exact input bytes the browser used, so any
difference comes from the backend (kernels, precision, fusion), not from the data.

Per output:
    max_abs_err, mean_abs_err, rmse
    max_rel_err     max |a-b| / max(|b|, 1e-6)   (unstable where the reference is near zero)
    norm_max_err    max |a-b| / max |b|          (error relative to the output's range; used in figures)
    cosine          cosine similarity of the flattened tensors
    nonfinite       NaN/Inf count in the browser output
    top1_agreement  only for outputs listed in the model's models.json "logits_outputs" (class or
                    vocabulary scores on the last axis): fraction of rows (all other axes) whose
                    argmax matches the reference. Agreement with the reference, not task accuracy
                    (no labelled data is involved).
    top5_overlap    mean |top-5 browser ∩ top-5 reference| / 5 over the same rows
"""
import base64
from functools import lru_cache

import numpy as np
import onnxruntime as ort

DTYPES = {
    "float32": np.float32, "float16": np.float16, "int32": np.int32, "int64": np.int64,
    "uint8": np.uint8, "int8": np.int8, "bool": np.bool_,
}


def decode(t):
    arr = np.frombuffer(base64.b64decode(t["b64"]), dtype=DTYPES[t["type"]])
    return arr.reshape(t["dims"])


@lru_cache(maxsize=2)
def reference_session(model_path):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])


def compare(model_path, io, logits_outputs=()):
    inputs = {t["name"]: decode(t) for t in io["inputs"]}
    sess = reference_session(str(model_path))
    names = [o.name for o in sess.get_outputs()]
    ref = dict(zip(names, sess.run(names, inputs)))
    rows = []
    for t in io["outputs"]:
        got = decode(t)
        exp = ref.get(t["name"])
        row = {"output": t["name"], "dims": "x".join(map(str, t["dims"])), "dtype": t["type"]}
        if exp is None or exp.shape != got.shape:
            row["error"] = f"reference shape {None if exp is None else exp.shape} vs browser {got.shape}"
            rows.append(row)
            continue
        a = got.astype(np.float64).ravel()
        b = exp.astype(np.float64).ravel()
        finite = np.isfinite(a)
        d = np.abs(a - b)
        row.update({
            "nonfinite": int((~finite).sum()),
            "max_abs_err": float(d.max()) if d.size else 0.0,
            "mean_abs_err": float(d.mean()) if d.size else 0.0,
            "rmse": float(np.sqrt((d ** 2).mean())) if d.size else 0.0,
            "max_rel_err": float((d / np.maximum(np.abs(b), 1e-6)).max()) if d.size else 0.0,
            "ref_abs_max": float(np.abs(b).max()) if b.size else 0.0,
            "norm_max_err": float(d.max() / np.abs(b).max()) if d.size and np.abs(b).max() > 0 else None,
            "cosine": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) if np.linalg.norm(a) * np.linalg.norm(b) > 0 else None,
        })
        if t["name"] in logits_outputs and got.shape[-1] >= 2:
            g2 = got.reshape(-1, got.shape[-1])
            e2 = exp.reshape(-1, exp.shape[-1])
            row["top1_agreement"] = float((g2.argmax(1) == e2.argmax(1)).mean())
            if got.shape[-1] >= 5:
                tg = np.argsort(-g2, axis=1)[:, :5]
                te = np.argsort(-e2, axis=1)[:, :5]
                row["top5_overlap"] = float(np.mean([len(set(x) & set(y)) / 5 for x, y in zip(tg, te)]))
        rows.append(row)
    return rows
