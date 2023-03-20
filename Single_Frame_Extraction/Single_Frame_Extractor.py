from ultralytics import YOLO
from IPython.display import Image
import torch
import sys
import cv2

BEST_MODEL = "Current_Best_Model.pt"

#Boundary values to segment image
EAST_HAND_BOUNDARY = 0
EAST_HU_BOUNDARY = 0

WEST_HAND_BOUNDARY = 0
WEST_HU_BOUNDARY = 0

NORTH_HAND_BOUNDARY = 0
NORTH_HU_BOUNDARY = 0

PLAYER_HAND_BOUNDARY = 0.88105726872 # y-axis boundary of player hand with margin of 10 pixels

def main():
    torch.cuda.is_available = lambda : False
    prediction_results = predictOnBestModel()
    extractPlayerHand(prediction_results)
    return

def extractPlayerHand(results):
    player_hand = []
    for tile in results:
        if(int(tile.boxes.xywhn[1]) > PLAYER_HAND_BOUNDARY):
            player_hand.append(tile)
            cv2.imshow("Single_Frame_Extraction/Example_Frame.png", tile.plot())
    
    # cv2.imshow("Single_Frame_Extraction/Example_Frame.png", )
    # for i in len(player_hand):
    #     print(player_hand[i])

def predictOnBestModel():
    model = YOLO(BEST_MODEL)
    results = model.predict(source="Single_Frame_Extraction/Example_Frame.png", save=True, stream=True)
    for result in results:
        print(result.boxes.xywhn)
        print(result.probs)
        
    return results


if __name__ == "__main__": main()
