import os
import sys
import argparse
import shutil
import math
from collections import OrderedDict

import json
import cv2
import PIL.Image
  
from sklearn.model_selection import train_test_split
from labelme import utils

def get_label_number(card):
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
        "D9" : 26,
        "HU" : 27,
        "PENG": 28,
        "KONG": 29
    }
    
    return cards.get(card)

def convert_labelme_to_yolo(input_json_folder, output_directory):

    for file in os.listdir(input_json_folder):
        
        with open(input_json_folder + '/' + file) as json_file:

            data = json.load(json_file)

            filename = os.path.splitext(os.path.basename(data['imagePath']))[0]
            
            output_label = open(output_directory + filename + ".txt","w+")
            
            img_width = data['imageWidth']
            img_height = data['imageHeight']
            
            for rectangle in data['shapes']:

                currentClass = rectangle['label']

                xmin = rectangle['points'][0][0]
                ymin = rectangle['points'][0][1]
                xmax = rectangle['points'][1][0]
                ymax = rectangle['points'][1][1]
                
                xmin, ymin, xmax, ymax = yolobbox2bbox(xmin, ymin, xmax, ymax, img_width, img_height)

                output_label.write(str(get_label_number(currentClass.upper())) + " " + str(xmin) + " " + str(ymin) + " " + str(xmax) + " " + str(ymax) + "\n")
                
            output_label.close()
        print("Converted " + file)
                    
                    
def yolobbox2bbox(x1, y1, x2, y2, image_w, image_h):
    centerX = (((x2 - x1) / 2) + x1 )/ image_w
    centerY = (((y2 - y1) / 2) + y1 )/ image_h
    width = (x2-x1) / image_w
    height = (y2 - y1) / image_h
    return centerX, centerY, width, height

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_dir',type=str,
                        help='Please input the path of the labelme json files.')
    args = parser.parse_args(sys.argv[1:])
    
    output_dir = args.json_dir + "\\Yolo_Labels\\"
    os.mkdir(output_dir)
    
    convert_labelme_to_yolo(args.json_dir, output_dir)