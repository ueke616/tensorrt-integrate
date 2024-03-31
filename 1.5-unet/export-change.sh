#!/bin/bash

git clone https://github.com/ueke616/unet-pytorch-change

cd unet-pytorch-change
export PYTHONPATH=$PYTHONPATH:.

python export.py

mv unet.onnx ../workspace/unet.onnx