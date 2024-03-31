import onnx
import onnxruntime
import numpy as np
import matplotlib.pyplot as plt
import cv2

def depth_inference():
    # providers : `['CUDAExecutionProvider', 'CPUExecutionProvider']`
    # session = onnxruntime.InferenceSession("workspace/road-segmentation-adas.onnx", providers=["CPUExecutionProvider"])
    session = onnxruntime.InferenceSession("workspace/ldrn_kitti_resnext101_pretrained_data_grad_256x512.onnx", providers=["CPUExecutionProvider"])
    
    image = cv2.imread("workspace/imgs/dashcam_00.jpg")
    image = cv2.resize(image, (512, 256))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image_tensor = (image / 255.0)
    mean = [0.485, 0.456, 0.406]
    norm = [0.229, 0.224, 0.225]
    image_tensor = ((image_tensor - mean) / norm).astype(np.float32)
    image_tensor = image_tensor.transpose(2, 0, 1)[None]
    
    prob = session.run(
        ["2499"], {"input.1": image_tensor}
    )[0]
    
    print(prob.shape)
    # prob = prob[0, 0] * -5 + 255
    prob = prob[0, 0]
    y = int(prob.shape[0] * 0.18)
    prob = prob[y:]
    
    # 保存概率图
    # prob最后一个维度 [不可行驶区域, 可行驶区域,马路牙子,车道线]
    # cv2.imwrite("workspace/road_depth_map.jpg", prob[0, 0] / 80 * 255)   # prob 最后一个维度是区域下标
    # cv2.imwrite("workspace/road_depth_map.jpg", prob)   # prob 最后一个维度是区域下标
    plt.imsave("workspace/road_depth_map.jpg", prob, cmap="plasma_r")


if __name__ == "__main__":
    depth_inference()




