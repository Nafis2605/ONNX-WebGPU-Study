import onnx
from onnx import AttributeProto
import numpy as np
import csv
import os

MODEL_DIR = "public/models"
OUTPUT_CSV = "flops_table.csv"


def extract_attrs(node):
    """
    Extract ONNX node attributes into a python dict.
    Includes ints, int, floats, float, strings/string for things like auto_pad.
    """
    attrs = {}
    for a in node.attribute:
        if a.type == AttributeProto.INTS:
            attrs[a.name] = list(a.ints)
        elif a.type == AttributeProto.INT:
            attrs[a.name] = a.i
        elif a.type == AttributeProto.FLOATS:
            attrs[a.name] = list(a.floats)
        elif a.type == AttributeProto.FLOAT:
            attrs[a.name] = a.f
        elif a.type == AttributeProto.STRINGS:
            attrs[a.name] = [s.decode("utf-8", errors="ignore") for s in a.strings]
        elif a.type == AttributeProto.STRING:
            attrs[a.name] = a.s.decode("utf-8", errors="ignore")
    return attrs


def safe_int_list(x, default):
    if x is None:
        return default
    if isinstance(x, (int, np.integer)):
        return [int(x)]
    try:
        return [int(v) for v in x]
    except Exception:
        return default


def safe_str(x, default=None):
    if x is None:
        return default
    if isinstance(x, str):
        return x
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, list) and x and isinstance(x[0], str):
        return x[0]
    return default


def normalize_shape(shape):
    """
    Replace None/0/strings with 1 so we can compute.
    """
    if shape is None:
        return None
    out = []
    for d in shape:
        if d is None:
            out.append(1)
        elif isinstance(d, str):
            out.append(1)
        else:
            try:
                v = int(d)
                out.append(1 if v == 0 else v)
            except Exception:
                out.append(1)
    return out


def infer_shapes(model):
    """
    Run ONNX shape inference and build a dict: tensor_name -> shape(list[int]).
    """
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

    shape_dict = {}

    # initializers
    for init in model.graph.initializer:
        shape_dict[init.name] = [int(d) for d in init.dims]

    # value_info
    for vi in model.graph.value_info:
        dims = vi.type.tensor_type.shape.dim
        shape = [d.dim_value for d in dims]
        shape_dict[vi.name] = normalize_shape(shape)

    # inputs
    for inp in model.graph.input:
        dims = inp.type.tensor_type.shape.dim
        shape = [d.dim_value for d in dims]
        shape_dict[inp.name] = normalize_shape(shape)

    # outputs
    for out in model.graph.output:
        dims = out.type.tensor_type.shape.dim
        shape = [d.dim_value for d in dims]
        shape_dict[out.name] = normalize_shape(shape)

    return shape_dict


def conv_output_hw(Hin, Win, kH, kW, pads, strides, dilations):
    pad_top, pad_left, pad_bottom, pad_right = pads
    sH, sW = strides
    dH, dW = dilations

    Hout = (Hin + pad_top + pad_bottom - dH * (kH - 1) - 1) // sH + 1
    Wout = (Win + pad_left + pad_right - dW * (kW - 1) - 1) // sW + 1
    return Hout, Wout


def compute_same_pads(Hin, Win, kH, kW, strides, dilations, auto_pad):
    """
    Approximate SAME_UPPER / SAME_LOWER pads.
    This is enough for FLOPs (output size) estimation.
    """
    sH, sW = strides
    dH, dW = dilations

    eff_kH = dH * (kH - 1) + 1
    eff_kW = dW * (kW - 1) + 1

    Hout = int(np.ceil(Hin / sH))
    Wout = int(np.ceil(Win / sW))

    pad_h = max((Hout - 1) * sH + eff_kH - Hin, 0)
    pad_w = max((Wout - 1) * sW + eff_kW - Win, 0)

    if auto_pad == "SAME_UPPER":
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
    else:  # SAME_LOWER
        pad_bottom = pad_h // 2
        pad_top = pad_h - pad_bottom
        pad_right = pad_w // 2
        pad_left = pad_w - pad_right

    return [pad_top, pad_left, pad_bottom, pad_right], Hout, Wout


def count_conv_flops(input_shape, weight_shape, attrs):
    """
    FLOPs for Conv as MACs (1 MAC = 1 FLOP here, per your choice).
    Correctly handles grouped/depthwise conv via 'group'.
    """
    # Expect NCHW input and OIHW weights
    if input_shape is None or len(input_shape) != 4:
        return 0

    N, Cin, Hin, Win = [int(x) for x in input_shape]
    Cout, _, kH, kW = [int(x) for x in weight_shape]

    strides = safe_int_list(attrs.get("strides"), [1, 1])
    dilations = safe_int_list(attrs.get("dilations"), [1, 1])
    pads = safe_int_list(attrs.get("pads"), [0, 0, 0, 0])
    groups = int(attrs.get("group", 1)) if attrs.get("group", None) is not None else 1
    auto_pad = safe_str(attrs.get("auto_pad"), None)

    # Ensure list lengths
    if len(strides) == 1:
        strides = [strides[0], strides[0]]
    if len(dilations) == 1:
        dilations = [dilations[0], dilations[0]]
    if len(pads) != 4:
        pads = [0, 0, 0, 0]

    # Handle SAME_* if pads not explicitly provided / or if model uses auto_pad
    if auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        pads, Hout, Wout = compute_same_pads(Hin, Win, kH, kW, strides, dilations, auto_pad)
    else:
        Hout, Wout = conv_output_hw(Hin, Win, kH, kW, pads, strides, dilations)

    if Hout <= 0 or Wout <= 0:
        return 0

    # grouped conv
    if groups <= 0:
        groups = 1
    Cin_per_group = Cin // groups if groups else Cin

    # MACs
    return int(N * Cout * Hout * Wout * Cin_per_group * kH * kW)


def count_matmul_flops(a_shape, b_shape):
    """
    FLOPs for MatMul as MACs (1 MAC = 1 FLOP).
    Supports:
      - 2D: [M,K] x [K,N] -> [M,N]
      - Batched: [..., M, K] x [..., K, N] -> [..., M, N]
    """
    a = normalize_shape(a_shape)
    b = normalize_shape(b_shape)
    if a is None or b is None or len(a) < 2 or len(b) < 2:
        return 0

    # Align ranks by left-padding with 1s for broadcasting
    ra, rb = len(a), len(b)
    r = max(ra, rb)
    a2 = [1] * (r - ra) + a
    b2 = [1] * (r - rb) + b

    # Last two dims are matrix dims
    M = a2[-2]
    K_a = a2[-1]
    K_b = b2[-2]
    N = b2[-1]

    if K_a != K_b:
        # If shapes don't line up, skip (safer than forcing)
        return 0

    # Broadcasted batch dims product
    batch = 1
    for da, db in zip(a2[:-2], b2[:-2]):
        if da == 1:
            batch *= db
        elif db == 1:
            batch *= da
        elif da == db:
            batch *= da
        else:
            # incompatible broadcast
            return 0

    return int(batch * M * K_a * N)


def main():
    rows = [["Model", "MFLOPs"]]

    for fname in sorted(os.listdir(MODEL_DIR)):
        if not fname.endswith(".onnx"):
            continue

        path = os.path.join(MODEL_DIR, fname)
        print(f"Processing {fname}...")

        try:
            model = onnx.load(path)
            onnx.checker.check_model(model)

            shape_dict = infer_shapes(model)
            initializer = {i.name: i for i in model.graph.initializer}

            total_flops = 0

            for node in model.graph.node:
                op = node.op_type

                if op == "Conv":
                    if len(node.input) < 2:
                        continue

                    x_name = node.input[0]
                    w_name = node.input[1]

                    x_shape = shape_dict.get(x_name)
                    if x_shape is None:
                        continue
                    x_shape = normalize_shape(x_shape)

                    W = initializer.get(w_name)
                    if W is None:
                        # weight might not be an initializer; skip
                        continue
                    w_shape = [int(d) for d in W.dims]

                    attrs = extract_attrs(node)
                    total_flops += count_conv_flops(x_shape, w_shape, attrs)

                elif op in ("Gemm", "MatMul"):
                    if len(node.input) < 2:
                        continue
                    a_name = node.input[0]
                    b_name = node.input[1]
                    a_shape = shape_dict.get(a_name)
                    b_shape = shape_dict.get(b_name)
                    if a_shape is None or b_shape is None:
                        continue
                    node_flops = count_matmul_flops(a_shape, b_shape)
                    total_flops += node_flops

            rows.append([fname, f"{total_flops / 1e6:.2f}"])

        except Exception as e:
            print(f"  Error: {e}")
            rows.append([fname, "Error"])

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\nDone! Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
