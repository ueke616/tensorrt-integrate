import onnx
import onnx.helper

# 加载主模型和后处理模型
model = onnx.load("workspace/ultra_fast_lane_detection_culane_288x800.onnx")
postprocess = onnx.load("workspace/postprocess.onnx")

# 遍历后处理模型中的所有节点
for n in postprocess.graph.node:
    # 重命名节点，添加"post/"前缀
    n.name = "post/" + n.name

    # 更新节点的输入名称
    for i, v in enumerate(n.input):
        if v == "0":
            # 特殊处理名为"0"的输入，映射到"200"
            n.input[i] = "200"
        else:
            # 其他输入名称添加"post/"前缀
            n.input[i] = "post/" + v

    # 更新节点的输出名称
    for i, v in enumerate(n.output):
        if v == "18":
            # 特殊处理名为"18"的输出，重命名为"points"
            n.output[i] = "points"
        else:
            # 其他输出名称添加"post/"前缀
            n.output[i] = "post/" + v

# 将后处理模型的节点添加到主模型的图中
model.graph.node.extend(postprocess.graph.node)

# 清除原模型的输出信息
while len(model.graph.output) > 0:
    model.graph.output.pop()

# 将后处理模型的输出设置为新模型的输出
model.graph.output.extend(postprocess.graph.output)

# 设置新模型的输入和输出信息，为了匹配特定的输入输出形状和类型
model.graph.input[0].CopyFrom(onnx.helper.make_tensor_value_info("input.1", 1, ["batch", 3, 288, 800]))
model.graph.output[0].CopyFrom(onnx.helper.make_tensor_value_info("points", 1, ["batch", 18, 4]))

# 清除图中的所有value_info信息（中间变量的形状信息），这在某些情况下可以减小模型文件的大小
while len(model.graph.value_info) > 0:
    model.graph.value_info.pop()
    
# 保存合并后的模型
onnx.save(model, "workspace/new-lane.onnx")
