#!/bin/bash
git clone git@github.com:ueke616/AlphaPose-change.git

cd AlphaPose-change

export PYTHONPATH=$PYTHONPATH:.

python scripts/export.py

mv alpha-pose-136.onnx ../workspace/