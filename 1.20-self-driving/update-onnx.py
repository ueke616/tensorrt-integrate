import onnx

# 加载老版本的ONNX模型
model_path = 'workspace/postprocess.onnx'
model = onnx.load(model_path)

# 你可以在这里对模型进行任何所需的操作或检查

# 使用当前ONNX版本重新保存模型
new_model_path = 'workspace/postprocess.onnx'
onnx.save(model, new_model_path)
