import os
import sys
import argparse

def add_number_to_filename(input_json_folder):
    
    for file in os.listdir(input_json_folder):
        filename = os.path.basename(file)
        os.rename(input_json_folder + "\\" + filename, input_json_folder + "\\" + os.path.splitext(filename)[0] + "_2" + os.path.splitext(filename)[1])
                    
                    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir',type=str,
                        help='Please input the path of the labelme json files.')
    args = parser.parse_args(sys.argv[1:])
    
    add_number_to_filename(args.input_dir)