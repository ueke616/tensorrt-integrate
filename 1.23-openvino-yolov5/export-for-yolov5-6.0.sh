#!/bin/bash
git clone git@github.com:ueke616/openvino_2022.1.0.643.git

cd yolov5-6.0
python export.py --weights=../yolov5s.pt --dynamic --include=onnx --opset=11

mv ../yolov5s.onnx ../workspace/