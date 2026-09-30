"""
Write a copy of a model with its graph-input dims pinned to the shapes in models.json.

ORT Web's WebGL backend rejects symbolic input dims (`expected shape '[,3,224,224]'`) even when
the fed tensor matches. The copy has the same graph and weights; only the input/output shape
declarations change. The script fails unless native ONNX Runtime (CPU) gives bit-identical outputs
for the original and the copy on the same seeded input.

Usage:
    python public/patch/pin_input_dims.py resnet50-v2 deeplabv3p-resnet50-human
    -> public/models/<file>_b1.onnx ; add "webgl_path": "/models/<file>_b1.onnx" to models.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort

PUBLIC = Path(__file__).resolve().parents[1]
REGISTRY = PUBLIC / "models" / "models.json"
NP_TYPES = {"float32": np.float32, "int32": np.int32, "int64": np.int64}


def pin_dims(model, inputs, batch):
    """Set the named graph inputs to `inputs` dims and a symbolic leading output dim to `batch`.

    (onnx.tools.update_model_dims can't be used: older exports list initializers as graph inputs.)
    """
    for i in model.graph.input:
        if i.name in inputs:
            for d, v in zip(i.type.tensor_type.shape.dim, inputs[i.name]):
                d.Clear()
                d.dim_value = int(v)
    for o in model.graph.output:
        dims = o.type.tensor_type.shape.dim
        if len(dims) and not dims[0].HasField("dim_value"):
            dims[0].Clear()
            dims[0].dim_value = int(batch)
    return model


def run_cpu(path, feeds):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return dict(zip([o.name for o in s.get_outputs()], s.run(None, feeds)))


def pin(entry):
    src = PUBLIC / entry["path"].lstrip("/")
    dst = src.with_name(src.stem + "_b1.onnx")
    model = onnx.load(str(src))
    inputs = {i["name"]: list(i["dims"]) for i in entry["inputs"]}
    graph_inputs = {i.name for i in model.graph.input} - {t.name for t in model.graph.initializer}
    if set(inputs) != graph_inputs:
        sys.exit(f"{entry['name']}: models.json inputs {sorted(inputs)} != graph inputs {sorted(graph_inputs)}")
    batch = next(iter(inputs.values()))[0]
    pinned = pin_dims(model, inputs, batch)
    onnx.checker.check_model(pinned)
    left = [i.name for i in pinned.graph.input if any(d.HasField("dim_param") for d in i.type.tensor_type.shape.dim)]
    if left:
        sys.exit(f"{entry['name']}: symbolic dims remain on inputs {left}")
    onnx.save(pinned, str(dst))

    rng = np.random.default_rng(0)
    feeds = {i["name"]: (np.ones(i["dims"], NP_TYPES[i["type"]]) if i.get("fill") == "ones"
                         else rng.standard_normal(i["dims"]).astype(NP_TYPES[i["type"]]))
             for i in entry["inputs"]}
    a, b = run_cpu(src, feeds), run_cpu(dst, feeds)
    worst = max(float(np.max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)))) for k in a)
    if a.keys() != b.keys() or worst != 0.0:
        dst.unlink()
        sys.exit(f"{entry['name']}: outputs differ (max abs diff {worst}); copy removed")
    print(f"{entry['name']}: {dst.name}  inputs {inputs}  outputs bit-identical ({len(a)} tensors)")
    return dst


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    models = {m["name"]: m for m in json.loads(REGISTRY.read_text())["models"]}
    for name in sys.argv[1:]:
        if name not in models:
            sys.exit(f"unknown model {name}; known: {sorted(models)}")
        pin(models[name])


if __name__ == "__main__":
    main()
