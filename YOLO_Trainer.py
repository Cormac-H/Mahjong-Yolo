from ultralytics import YOLO
from IPython.display import Image
import torch
import sys

BEST_MODEL = "Current_Best_Model.pt"
DEFAULT_MODEL = "yolov8n.pt"

def trainModel():
    model = YOLO(DEFAULT_MODEL)
    model.train(data="MahjongTiles.yaml", epochs=3, cache=True)
    metrics = model.val()
    print(metrics)

def predictOnBestModel():
    model = YOLO(BEST_MODEL)
    results = model.predict(source="Test-Screenshots", save=True, stream=True)
    # model.predict(source="Test-Videos", save=True)
    for result in results:
        print(result.boxes.xywh)   # box with xywh format, (N, 4)
        print(result.boxes.conf)
        # classification
        print(result.probs)
    
def main():
    torch.cuda.is_available = lambda : False
    predictOnBestModel()
    return


if __name__ == "__main__": main()