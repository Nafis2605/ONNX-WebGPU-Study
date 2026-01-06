import onnx
m = onnx.load("bert_Opset17_int32.onnx")
for i in m.graph.input:
    print(i.name, i.type.tensor_type.elem_type)
