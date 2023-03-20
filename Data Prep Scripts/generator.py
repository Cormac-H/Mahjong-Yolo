import os
import cv2 
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

def get_random_height(height):
    vaild = False
    while(vaild == False):
        random_height = random.random()
        if (720-height)/720 < random_height:
            vaild = False
        elif (height/720) > random_height:
            vaild = False
        else:
            vaild = True
    return random_height

def get_random_width(width):
    vaild = False
    while(vaild == False):
        random_width = random.random()
        if (1520-width)/1520 < random_width:
            vaild = False
        elif (width/1520) > random_width:
            vaild = False
        else:
            vaild = True
    return random_width

def get_random_centre(height,width):
    return get_random_height(height),get_random_width(width)

def get_card_size():
    return random.uniform(0.8,1.2)

def label_number(card):
    cards = {
        "B1" : 0,
        "B2" : 1,
        "B3" : 2,
        "B4" : 3,
        "B5" : 4,
        "B6" : 5,
        "B7" : 6,
        "B8" : 7,
        "B9" : 8,
        "C1" : 9,
        "C2" : 10,
        "C3" : 11,
        "C4" : 12,
        "C5" : 13,
        "C6" : 14,
        "C7" : 15,
        "C8" : 16,
        "C9" : 17,
        "D1" : 18,
        "D2" : 19,
        "D3" : 20,
        "D4" : 21,
        "D5" : 22,
        "D6" : 23,
        "D7" : 24,
        "D8" : 25,
        "D9" : 26
    }
    return cards.get(card)

def generate_images(image_count, image_rotation):
    for x in range(image_count):
        for y in ["B","C","D"]:
            for z in range(1,10):
                background = cv2.imread("./MahjongTiles/background.jpg")
                background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
                card = cv2.imread("./MahjongTiles/" + y + str(z) + ".jpg")
                card = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
                
                if image_rotation == 90:
                    card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
                elif image_rotation == 180:
                    card = cv2.rotate(card, cv2.ROTATE_180)
                elif image_rotation == 270:
                    card = cv2.rotate(card, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    
                height = card.shape[0]
                width = card.shape[1]
                size = get_card_size()
                card = cv2.resize(card,(int(width*size),int(height*size)))
                height = card.shape[0]
                width = card.shape[1]
                
                centre_height,centre_width = get_random_centre(height,width)
                top_left_height = int((centre_height*720)-(height/2))
                top_left_width = int((centre_width*1520)-(width/2))
                background[top_left_height:top_left_height+height, top_left_width:top_left_width+width] = card
                
                cv2.imwrite("./test-generator/images/image_" + str(x) + "_" + y + str(z) + "_rotated_" + str(image_rotation) + ".jpg", cv2.cvtColor(background, cv2.COLOR_RGB2BGR))
                files = open("./test-generator/labels/image_" + str(x) + "_" + y + str(z) + "_rotated_" + str(image_rotation) + ".txt", 'w')
                yolo = str(label_number(y+str(z)))+" "+str(centre_width)+" "+str(centre_height)+" "+str(width/1520)+" "+str(height/720)
                files.write(yolo)
                
                print("Generated image_" + str(x) + "_" + y + str(z) + "_rotated_" + str(image_rotation) + ".jpg")
                
                
def get_random_tile(tile_counts):
    suits = ["B", "C", "D"]
    suit = random.choice(suits)
    tile_value = random.randint(1, 9)
    while(tile_counts[label_number(suit + str(tile_value))] >= 4):
        suit = random.choice(suits)
        tile_value = random.randint(1, 9)
        
    tile_counts[label_number(suit + str(tile_value))] += 1
    return suit, tile_value

def generate_player_hand(background, tile_counts, label):
    STARTING_X = 290
    STARTING_Y = 640
    
    for i in range(13):
        suit, tile_value = get_random_tile(tile_counts)
        card = cv2.imread("./MahjongTiles/" + suit + str(tile_value) + ".jpg")
        card = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
        
        height = card.shape[0]
        width = card.shape[1]
        
        card = cv2.resize(card,(int(width*1.67),int(height*1.67)))
        height = card.shape[0]
        width = card.shape[1]
        
        centre_height = STARTING_Y / 720
        centre_width = (STARTING_X + (i * width)) / 1520
        
        top_left_height = int((centre_height*720)-(height/2))
        top_left_width = int((centre_width*1520)-(width/2))
        background[top_left_height:top_left_height+height, top_left_width:top_left_width+width] = card

        yolo_bbox = str(label_number(suit+str(tile_value)))+" "+str(centre_width)+" "+str(centre_height)+" "+str(width/1520)+" "+str(height/720) + "\n"
        label.write(yolo_bbox)
        
    return background

def experimental_generate_images(image_count):
    tile_counts = np.zeros(27)
    for x in range(image_count):
        background = cv2.imread("./MahjongTiles/background.jpg")
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
        
        label = open("./image_" + str(x) + ".txt", 'w')
        
        background = generate_player_hand(background, tile_counts, label)
        
        cv2.imwrite("./image_" + str(x) + ".jpg" , cv2.cvtColor(background, cv2.COLOR_RGB2BGR))
        
        print("Generated image_" + str(x) + ".jpg")

if __name__ == "__main__":
    print("GENERATING DATASET")
    # generate_images(100, 0)
    # generate_images(100, 90)
    # generate_images(100, 180)
    # generate_images(100, 270)
    experimental_generate_images(1)