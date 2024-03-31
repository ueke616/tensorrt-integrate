import cv2
import numpy as np
import onnxruntime


# 道路分割图
def road_segmentation():
    session = onnxruntime.InferenceSession("workspace/road-segmentation-adas.onnx", providers=["CPUExecutionProvider"])
    
    image = cv2.imread("workspace/imgs/dashcam_00.jpg")
    image = cv2.resize(image, (896, 512))
    image_tensor = image.astype(np.float32)
    image_tensor = image_tensor.transpose(2, 0, 1)[None]
    
    prob = session.run(
        ["tf.identity"], {"data": image_tensor}
    )[0]
    
    print(prob.shape)
    
    # 保存概率图
    # prob最后一个维度 [不可行驶区域, 可行驶区域,马路牙子,车道线]
    cv2.imwrite("workspace/road_segmentation.jpg", prob[0, :, :, 3] * 255)   # prob 最后一个维度是区域下标


def ldrn_kitti_depth():
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
    prob = prob[0, 0] * -5 + 255
    y = int(prob.shape[0] * 0.18)
    prob = prob[y:]
    
    # 保存概率图
    # prob最后一个维度 [不可行驶区域, 可行驶区域,马路牙子,车道线]
    # cv2.imwrite("workspace/road_depth_map.jpg", prob[0, 0] / 80 * 255)   # prob 最后一个维度是区域下标
    cv2.imwrite("workspace/road_depth_map.jpg", prob)   # prob 最后一个维度是区域下标


if __name__ == "__main__":
    ldrn_kitti_depth()
    # road_segmentation()



