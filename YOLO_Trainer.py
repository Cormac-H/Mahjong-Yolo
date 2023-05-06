import ultralytics
from ultralytics import YOLO
from ultralytics.yolo.utils import ROOT, yaml_load
from ultralytics.yolo.utils.checks import check_yaml

from IPython.display import Image
import torch
import sys
import numpy as np
import cv2

BEST_MODEL = "Current_Best_Model.pt"
DEFAULT_MODEL = "yolov8n.pt"

#Train function will train a YOLO model according to MahjongTiles.yaml
#Predict will form predictions using BEST_MODEL on data in Test-Screenshots and Videos
CLASSES = yaml_load(check_yaml('MahjongTiles.yaml'))['names']

def main():
    torch.cuda.is_available = lambda : False
    # ultralytics.checks()
    # trainModel()
    predictOnBestModel()
    return

def trainModel():
    model = YOLO(DEFAULT_MODEL)
    model.train(data="MahjongTiles.yaml", epochs=1, cache=True, save=True, save_period=1, optimizer='SGD', seed=42, workers=12, batch=8)
    metrics = model.val()
    print(metrics)
    success = model.export(format="onnx")

def predictOnBestModel():
    model = YOLO(BEST_MODEL)
    results = model.predict(source="Test-Screenshots", save=True) # requires stream=True for below
    # model.predict(source="Test-Videos", save=True)
    # for result in results:
        # print(result.boxes.xywh)   # box with xywh format, (N, 4)
        # print(result.boxes.conf)
        # # classification
        # print(result.probs)


    
if __name__ == "__main__": main()