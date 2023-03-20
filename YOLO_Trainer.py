from ultralytics import YOLO
from IPython.display import Image
import torch
import sys

BEST_MODEL = "Mojiang.pt"
DEFAULT_MODEL = "yolov8m.pt"

#Train function will train a YOLO model according to MahjongTiles.yaml
#Predict will form predictions using BEST_MODEL on data in Test-Screenshots and Videos

def main():
    torch.cuda.is_available = lambda : False
    # trainModel()
    predictOnBestModel()
    return

def trainModel():
    model = YOLO(DEFAULT_MODEL)
    model.train(data="MahjongTiles.yaml", epochs=10, cache=True)
    metrics = model.val()
    print(metrics)

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