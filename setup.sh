#!/bin/bash#

export DIRECTORY=weight_onnx
if [ ! -d "$DIRECTORY" ]; then
  mkdir $DIRECTORY
  cd weight_onnx
  wget https://github.com/onnx/models/raw/main/vision/classification/mobilenet/model/mobilenetv2-12.onnx 
  wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/yolov4/model/yolov4.onnx
   wget https://github.com/onnx/models/raw/main/vision/object_detection_segmentation/fcn/model/fcn-resnet50-12.onnx

  cd ..
fi
