import onnx
m = onnx.load("deeplabv3p-resnet50-human.onnx")
for i in m.graph.input:
    print(i.name, i.type.tensor_type.elem_type)
