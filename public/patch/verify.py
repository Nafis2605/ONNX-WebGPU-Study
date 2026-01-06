import onnx
m = onnx.load("adv_inception_v3_Opset17_int32.onnx")
for i in m.graph.input:
    print(i.name, i.type.tensor_type.elem_type)
