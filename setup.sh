#!/bin/bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install tesseract-ocr-rus tesseract-ocr-eng
curl -L -o ~/Downloads/yolo-weights-for-licence-plate-detector.zip\
  https://www.kaggle.com/api/v1/datasets/download/achrafkhazri/yolo-weights-for-licence-plate-detector