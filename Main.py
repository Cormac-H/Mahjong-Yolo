from ultralytics import YOLO
from IPython.display import Image
import torch
import sys

def main():
    print(sys.prefix)

    model = YOLO("yolov8n.pt")
    model.train(data="MahjongTiles.yaml", epochs=3, cache=True)
    metrics = model.val()
    print(metrics)
    print("done")
    return

if __name__ == "__main__": main()