import onnx
import onnxruntime
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy

"""
## 对车道线检测模型的分析
1. 输入是: 1x3x288x800
2. 输出是: 1x201x18x4
3. 对于车道线检测任务而言有一些定义或者说是先验
    - 只需要识别 4 根线
    - 对于车道线基本是在地面上的, 因此, y 方向可以从图像中心开始, 也就是 achor 起始坐标是图像中心到图像底部
    - 对于对于车道线的检测, 因为线是连续的, 因此这里可以转变为离散的点检测, 对于一根线可以设计为 18 个点来描述
    - 因此回归一个点, 其 y 坐标已知, x 坐标需要回归出来
    - 对于 x 的回归, 采用了位置概率来表示, 划分为 200 个网格表示其坐标
    - 对于车道线的点是否存在这个问题, 采用第 201 个概率表示。 若这个点不存在, 则 201 个位置的值是最大的
### 下面操作在 1.20-self-driving/self-driving-ish_computer_vision_system/image_processor/lane_engine.cpp 也可找到
我们的实现 lane-detection.py
4. 图像的预处理直接是 image / 255
5. 图像需要从 BGR 到 RGB
6. 图像 resize 到 (288, 800)
7. 后处理部分:
    - 对 0-200 (纵向网格)维度进行 softmax, 此时得到的是位置概率
    - 对位置概率和位置索引点乘相加, 得到 location, 此时 location 是 18x4(点、线)
    - 对原始输出的最大值进行判断, 决定该点是否存在(基于位置概率的最大值与预定义阈值的比较，来决定该点是否存在)
    - 最后通过过滤得到 4 根线的坐标
参考代码: https://github.com/cfzd/Ultra-Fast-Lane-Detection/blob/master/demo.py
"""

def lane_detection():
    # providers : `['CUDAExecutionProvider', 'CPUExecutionProvider']`
    # session = onnxruntime.InferenceSession("workspace/road-segmentation-adas.onnx", providers=["CPUExecutionProvider"])
    session = onnxruntime.InferenceSession("workspace/ultra_fast_lane_detection_culane_288x800.onnx", providers=["CPUExecutionProvider"])
    
    image = cv2.imread("workspace/imgs/dashcam_00.jpg")
    show  = image.copy()
    image = cv2.resize(image, (800, 288))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = (image / 255.0).astype(np.float32)
    image_tensor = image_tensor.transpose(2, 0, 1)[None]
    
    pred = session.run(
        ["200"], {"input.1": image_tensor}
    )[0][0]        # 200: output ops name, input.1: input ops name
    
    print(pred.shape)
    # 对 0-200 (纵向网格)维度进行 softmax, 此时得到的是位置概率
    out_j = pred
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(200) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    print(loc.shape)
    
    # 判断点是否该存在, 计算 mask 坐标
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == 200] = 0  # x坐标
    
    col_sample = np.linspace(0, 800-1, 200)  # y坐标 800个像素划分为200个格子
    col_sample_w = col_sample[1] - col_sample[0]  # 横坐标轴上每个格子的间隔
    print(col_sample_w)
    
    culane_row_anchor = [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
    ys = np.array(culane_row_anchor)
    # 恢复图片中的点的坐标
    xs = loc * col_sample_w * show.shape[1] / 800
    ys = ys * show.shape[0] / 288
    
    colors = [(0, 0, 255), (0, 255, 0), (0, 255, 0), (0, 0, 255)]
    
    # 画车道线
    for iline in range(4):
        for x, y in zip(xs[:, iline], ys):
            if x == 0:
                continue
            
            cv2.circle(show, (int(x), int(y)), 5, colors[iline], -1, 16)
    # 保存概率图
    plt.imsave("workspace/lane_detection.jpg", show)
    # 修改后代码的模型导出: change-lane-onnx.py


if __name__ == "__main__":
    lane_detection()




