#!/bin/bash

git clone https://github.com/mohenghui/yolov5_6.0
mv yolov5_6.0 yolov5-6.0

cd yolov5-6.0
python export.py --weights=../yolov5s.pt --dynamic --include=onnx --opset=11

# mv ../yolov5s.onnx ../workspace/yolov5s-raw.onnx
mv ../yolov5s.onnx ../workspace/yolov5s.onnx

rm -rf yolov5-6.0
