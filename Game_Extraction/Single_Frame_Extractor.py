import ultralytics
from ultralytics import YOLO
from ultralytics.yolo.utils import ROOT, yaml_load
from ultralytics.yolo.utils.checks import check_yaml
# from IPython.display import Image
import onnx
from onnx.defs import onnx_opset_version
import onnxruntime
import numpy as np
import torch
import sys
import cv2
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BEST_MODEL = "../Current_Best_Model.pt"
BEST_MODEL_ONNX = ROOT_DIR + "\Current_Best_Model.onnx"
TEST_IMAGE = ROOT_DIR + "\Game_Extraction\Example_Frame.png"

CLASSES = yaml_load(check_yaml('MahjongTiles.yaml'))['names']

#Boundary values to segment image

PLAYER_HAND = []
PLAYER_PENG = []
PLAYER_DISCARD = []
PLAYER_HU = []

EAST_PENG = []
EAST_DISCARD = []
EAST_HU = []

NORTH_PENG = []
NORTH_DISCARD = []
NORTH_HU = []

WEST_PENG = []
WEST_DISCARD = []
WEST_HU = []


def main():
    # success = exportModel()
    detections = classify_image(BEST_MODEL_ONNX, TEST_IMAGE)
    extract_tiles(detections)
    print_state()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return
    
def exportModel():
    model = YOLO(BEST_MODEL)
    
    return model.export(format="onnx", opset=12)

def check_valid_opsets():
    model = YOLO(BEST_MODEL)
    for test_opset in range(1, onnx_opset_version() + 1):
        try:
            print("\n\nVERSION ATTEMPT: " + str(test_opset) + "\n\n")
            onx = model.export(format="onnx", opset=test_opset)
        except RuntimeError as e:
            print('target: %r error: %r' % (test_opset, e))
            continue

def classify_image(onnx_model, input_image):
    model= cv2.dnn.readNetFromONNX(onnx_model)
    original_image: np.ndarray = cv2.imread(input_image)
    [height, width, _] = original_image.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / 640

    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)
    outputs = model.forward()

    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2], outputs[0][i][3]]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

    detections = []
    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]

        draw_bounding_box(original_image, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                          round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))
        
        box[0] = box[0] * scale / width
        box[1] = box[1] * scale / height
        box[2] = box[2] * scale / width
        box[3] = box[3] * scale / height
        
        detection = {
            'class_id': class_ids[index],
            'class_name': CLASSES[class_ids[index]],
            'confidence': scores[index],
            'box': box
        }
        detections.append(detection)
    
    cv2.imshow('image', original_image)

    return detections

def extract_tiles(detections):
    for tile in detections:
        if tile["confidence"] < 0.3:
            pass
        elif tile["box"][2] > 0.05:
            PLAYER_HAND.append(tuple((tile["class_name"], round(tile["confidence"], 2))))
        elif tile["box"][0] > 0.4 and tile["box"][1] > 0.58:
            PLAYER_DISCARD.append(tuple((tile["class_name"], round(tile["confidence"], 2))))
            
        elif tile["box"][0] > 0.8 and tile["box"][1] < 0.5:
            EAST_PENG.append(tuple((tile["class_name"], round(tile["confidence"], 2))))
        elif tile["box"][0] > 0.57 and tile["box"][1] < 0.55:
            EAST_DISCARD.append(tuple((tile["class_name"], round(tile["confidence"], 2))))
            
        elif tile["box"][0] > 0.22 and tile["box"][1] < 0.15:
            NORTH_PENG.append(tuple((tile["class_name"], round(tile["confidence"], 2))))
        elif tile["box"][0] > 0.42 and tile["box"][1] < 0.25:
            NORTH_DISCARD.append(tuple((tile["class_name"], round(tile["confidence"], 2))))

        elif tile["box"][0] > 0.32:
            WEST_DISCARD.append(tuple((tile["class_name"], round(tile["confidence"], 2))))
        elif tile["box"][0] > 0.18:
            WEST_HU.append(tuple((tile["class_name"], round(tile["confidence"], 2))))
        else:
            WEST_PENG.append(tuple((tile["class_name"], round(tile["confidence"], 2))))
            
def print_state():
    print("Player Hand    " + str(PLAYER_HAND) + " total = " + str(len(PLAYER_HAND)))
    print("Player Discard " + str(PLAYER_DISCARD) + " total = " + str(len(PLAYER_DISCARD)))
    print("Player Peng    " + str(PLAYER_PENG) + " total = " + str(len(PLAYER_PENG)))
    print("Player Hu      " + str(PLAYER_HU) + " total = " + str(len(PLAYER_HU)) + "\n")
    
    print("East Peng      " + str(EAST_PENG) + " total = " + str(len(EAST_PENG)))
    print("East Discard   " + str(EAST_DISCARD) + " total = " + str(len(EAST_DISCARD)))
    print("East HU        " + str(EAST_HU) + " total = " + str(len(EAST_HU)) + "\n")
    
    print("North Peng     " + str(NORTH_PENG) + " total = " + str(len(NORTH_PENG)))
    print("North Discard  " + str(NORTH_DISCARD) + " total = " + str(len(NORTH_DISCARD)))
    print("North Hu       " + str(NORTH_HU) + " total = " + str(len(NORTH_HU)) + "\n")

    print("West Peng      " + str(WEST_PENG) + " total = " + str(len(WEST_PENG)))
    print("West Discard   " + str(WEST_DISCARD) + " total = " + str(len(WEST_DISCARD)))
    print("West Hu        " + str(WEST_HU) + " total = " + str(len(WEST_HU)))
    
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f'{CLASSES[class_id]} ({confidence:.2f})'
    colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


if __name__ == "__main__": main()
