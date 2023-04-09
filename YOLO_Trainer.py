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

colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

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

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
if __name__ == "__main__": main()