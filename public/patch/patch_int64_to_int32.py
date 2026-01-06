import sys
import onnx
from onnx import TensorProto

if len(sys.argv) < 2:
    print("Usage: python patch_int64_to_int32.py model1.onnx [model2.onnx ...]")
    sys.exit(1)

def patch_model(path):
    print(f"\nPatching {path}...")
    model = onnx.load(path)

    def patch_vi(v):
        t = v.type.tensor_type
        if t.elem_type == TensorProto.INT64:
            t.elem_type = TensorProto.INT32

    # Patch inputs
    for inp in model.graph.input:
        patch_vi(inp)

    # Patch intermediate tensors
    for vi in model.graph.value_info:
        patch_vi(vi)

    # Patch outputs
    for out in model.graph.output:
        patch_vi(out)

    # Patch initializers (constants)
    for init in model.graph.initializer:
        if init.data_type == TensorProto.INT64:
            init.data_type = TensorProto.INT32

    out_path = path.replace(".onnx", "_int32.onnx")
    onnx.save(model, out_path)
    print(f"Saved → {out_path}")

for p in sys.argv[1:]:
    patch_model(p)
