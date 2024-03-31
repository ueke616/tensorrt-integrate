import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
from torch import nn
from torchvision.models import resnet18


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用torchvision自带的与训练模型, 更多模型请参考:https://tensorvision.readthedocs.io/en/master/
        self.backbone = resnet18(pretrained=True)
    
    def forward(self, x):
        feature         = self.backbone(x)
        probability     = F.softmax(feature, dim=1)
        return probability


if __name__ == "__main__":
    # 对每个通道进行归一化有助于模型的训练
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    
    # 对图像进行预处理
    image = cv.imread("workspace/dog.jpg")
    image = cv.resize(image, (224, 224))        # resize
    image = image[..., ::-1]                    # BGR -> RGB
    image = image / 255.0
    image = (image - imagenet_mean) / imagenet_std  # normalize
    image = image.astype(np.float32)            # float64 -> float32
    image = image.transpose(2, 0, 1)            # HWC -> CHW
    image = np.ascontiguousarray(image)         # contiguous array memory
    image = image[None, ...]                    # CHW -> 1CHW
    image = torch.from_numpy(image)             # numpy -> torch
    model = Classifier().eval()
    
    with torch.no_grad():
        probability     = model(image)
    
    predict_class   = probability.argmax(dim=1).item()
    confidence      = probability[0, predict_class]

    labels = open("workspace/labels.imagenet.txt").readlines()
    labels = [item.strip() for item in labels]
    
    print("Predict: {predict_class}, {confidence}, {labels[predict_class]}")

    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        model, (dummy,), "workspace/classifier.onnx",
        input_names=["images"],
        output_names=["prob"],
        dynamic_axes={"image": {0: "batch"}, "prob": {0: "batch"}},
        opset_version=11
    )




