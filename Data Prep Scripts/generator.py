import os
import cv2 
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

# Get label number based on suit + value
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

# Return the first available tile that has enough to turn into a peng       
def get_peng_tile(tile_counts):
    suits=["B", "C", "D"]
    for suit in suits:
        for i in range(1,9):            
            if (tile_counts[label_number(suit + str(i))]) <= 1:
                tile_counts[label_number(suit + str(i))] += 3
                return suit, i
    return "NO_PENG", 0

# Return a random tile that hasn't already been used more than 4 times
def get_random_tile(tile_counts):
    suits = ["B", "C", "D"]
    suit = random.choice(suits)
    tile_value = random.randint(1, 9)
    while(tile_counts[label_number(suit + str(tile_value))] >= 4):
        suit = random.choice(suits)
        tile_value = random.randint(1, 9)
        
    tile_counts[label_number(suit + str(tile_value))] += 1
    return suit, tile_value

# Generate a row of 3 tiles given coordinates and a side of the player table
def generate_peng_row(background, tile_counts, label, STARTING_X, STARTING_Y, DIRECTION):
    suit, tile_value = get_peng_tile(tile_counts)
    if suit == "NO_PENG":
        return background # Refuse to generate a row if it can't be legally constructed
    
    for i in range(3):
        card = cv2.imread("./Mahjong-Tiles-Small/" + suit + str(tile_value) + ".jpg")
        card = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
        
        height = card.shape[0]
        width = card.shape[1]
                
        if DIRECTION == "EAST":
            card = cv2.resize(card,(int(width),int(height)))
            card = cv2.rotate(card, cv2.ROTATE_90_COUNTERCLOCKWISE)

        elif DIRECTION == "NORTH":
            card = cv2.resize(card,(int(width),int(height)))
            card = cv2.rotate(card, cv2.ROTATE_180)
        elif DIRECTION == "WEST":
            card = cv2.resize(card,(int(width),int(height)))
            card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)

        else:
            card = cv2.resize(card,(int(width),int(height)))

        height = card.shape[0]
        width = card.shape[1]
        
        if DIRECTION == "SOUTH" or DIRECTION == "NORTH":
            centre_height = STARTING_Y / 720
            centre_width = (STARTING_X + (i * width)) / 1520
        else:
            centre_height = (STARTING_Y + (i * height)) / 720
            centre_width = STARTING_X  / 1520
        
        top_left_height = int((centre_height*720)-(height/2))
        top_left_width = int((centre_width*1520)-(width/2))
        background[top_left_height:top_left_height+height, top_left_width:top_left_width+width] = card

        yolo_bbox = str(label_number(suit+str(tile_value)))+" "+str(centre_width)+" "+str(centre_height)+" "+str(width/1520)+" "+str(height/720) + "\n"
        label.write(yolo_bbox)
        
    return background

# Generate a row of 6 tiles given coordinates and a side of the player table
def generate_discard_row(background, tile_counts, label, STARTING_X, STARTING_Y, DIRECTION):
    for i in range(6):
        suit, tile_value = get_random_tile(tile_counts)
        card = cv2.imread("./Mahjong-Tiles-Small/" + suit + str(tile_value) + ".jpg")
        card = cv2.cvtColor(card, cv2.COLOR_BGR2RGB)
        
        height = card.shape[0]
        width = card.shape[1]
                
        if DIRECTION == "EAST":
            card = cv2.resize(card,(int(width),int(height)))
            card = cv2.rotate(card, cv2.ROTATE_90_COUNTERCLOCKWISE)

        elif DIRECTION == "NORTH":
            card = cv2.resize(card,(int(width),int(height)))
            card = cv2.rotate(card, cv2.ROTATE_180)
        elif DIRECTION == "WEST":
            card = cv2.resize(card,(int(width),int(height)))
            card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)

        else:
            card = cv2.resize(card,(int(width),int(height)))

        height = card.shape[0]
        width = card.shape[1]
        
        if DIRECTION == "SOUTH" or DIRECTION == "NORTH":
            centre_height = STARTING_Y / 720
            centre_width = (STARTING_X + (i * width)) / 1520
        else:
            centre_height = (STARTING_Y + (i * height)) / 720
            centre_width = STARTING_X  / 1520
        
        top_left_height = int((centre_height*720)-(height/2))
        top_left_width = int((centre_width*1520)-(width/2))
        background[top_left_height:top_left_height+height, top_left_width:top_left_width+width] = card

        yolo_bbox = str(label_number(suit+str(tile_value)))+" "+str(centre_width)+" "+str(centre_height)+" "+str(width/1520)+" "+str(height/720) + "\n"
        label.write(yolo_bbox)
        
    return background

# Fill in dedicated coordinates to generate peng tiles
def generate_peng_tiles(background, tile_counts, label):
    SOUTH_X = 1334
    SOUTH_Y = 650

    NORTH_X = 366
    NORTH_Y = 45
    
    EAST_X = 1269
    EAST_Y = 54
    
    WEST_X = 194
    WEST_Y = 280
    
    # Prioritize generating min one row of peng tiles
    background = generate_peng_row(background, tile_counts, label, SOUTH_X, SOUTH_Y, "SOUTH")
    background = generate_peng_row(background, tile_counts, label, NORTH_X, NORTH_Y, "NORTH")
    background = generate_peng_row(background, tile_counts, label, EAST_X, EAST_Y, "EAST")
    background = generate_peng_row(background, tile_counts, label, WEST_X, WEST_Y, "WEST")
    
    background = generate_peng_row(background, tile_counts, label, NORTH_X + 150, NORTH_Y, "NORTH")
    background = generate_peng_row(background, tile_counts, label, EAST_X, EAST_Y + 150, "EAST")
    background = generate_peng_row(background, tile_counts, label, WEST_X, WEST_Y + 150, "WEST")
    
    return background

# Fill in dedicated coordinates to generate discarded tiles for all players
def generate_discarded_tiles(background, tile_counts, label):
    SOUTH_X = 643
    SOUTH_Y = 450

    NORTH_X = 653
    NORTH_Y = 190
    
    EAST_X = 930
    EAST_Y = 220
    
    WEST_X = 580
    WEST_Y = 220
    background = generate_discard_row(background, tile_counts, label, SOUTH_X, SOUTH_Y, "SOUTH")
    background = generate_discard_row(background, tile_counts, label, SOUTH_X, SOUTH_Y+48, "SOUTH")

    background = generate_discard_row(background, tile_counts, label, NORTH_X, NORTH_Y, "NORTH")
    background = generate_discard_row(background, tile_counts, label, NORTH_X, NORTH_Y-48, "NORTH")

    background = generate_discard_row(background, tile_counts, label, EAST_X, EAST_Y, "EAST")
    background = generate_discard_row(background, tile_counts, label, EAST_X+48, EAST_Y, "EAST")
    
    background = generate_discard_row(background, tile_counts, label, WEST_X, WEST_Y, "WEST")
    background = generate_discard_row(background, tile_counts, label, WEST_X-48, WEST_Y, "WEST")
    

        
    return background

# Generate 12 tiles forming the player's hand
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

# Main script to generate images
def generate_images(image_num):
    tile_counts = np.zeros(27)
    background = cv2.imread("./MahjongTiles/background.jpg")
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    
    label = open("./Generated_Data/labels/generated_" + str(image_num) + ".txt", 'w')
    # label = open("./Data/labels/train/generated_" + str(image_num) + ".txt", 'w')
    
    background = generate_player_hand(background, tile_counts, label)
    background = generate_discarded_tiles(background, tile_counts, label)
    background = generate_peng_tiles(background, tile_counts, label)
    
    cv2.imwrite("./Generated_Data/images/generated_" + str(image_num) + ".jpg" , cv2.cvtColor(background, cv2.COLOR_RGB2BGR))
    # cv2.imwrite("./Data/images/train/generated_" + str(image_num) + ".jpg" , cv2.cvtColor(background, cv2.COLOR_RGB2BGR))
    
    print("Created " + str(image_num) + ".jpg")

if __name__ == "__main__":
    print("GENERATING DATASET")
    image_count = 1
    for i in range(image_count):
        generate_images(i)