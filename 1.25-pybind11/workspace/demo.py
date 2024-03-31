
import yolo
import os
import cv2

if not os.path.exists("workspace/yolov5s.trtmodel"):
    yolo.compileTRT(
        max_batch_size=1,
        source="workspace/yolov5s.onnx",
        output="workspace/yolov5s.trtmodel",
        fp16=False,
        device_id=0
    )

infer = yolo.Yolo("workspace/yolov5s.trtmodel")
if not infer.valid:
    print("invalid trtmodel")
    exit(0)

image = cv2.imread("workspace/rq.jpg")
boxes = infer.commit(image).get()

for box in boxes:
    l, t, r, b = map(int, [box.left, box.top, box.right, box.bottom])
    cv2.rectangle(image, (l, t), (r, b), (0, 255, 0), 2, 16)

cv2.imwrite("workspace/detect.jpg", image)