from PIL import Image, ImageDraw
import numpy as np
import json
import argparse
import glob
import os
import pdb

def get_parser():
    """ This function returns a parser object """
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--images_dir', type=str)
    aparser.add_argument('--chip_size', type=int, default=224)
    aparser.add_argument('--overlap', type=int, default=0.1)
    aparser.add_argument('--output_dir', type=str, default=0)

    return aparser

def image_chipper(parent_img_name, chip_size, output_dir):
    """ Chips the full images and saves them into the output directory """
    img_arr = np.asarray(Image.open(parent_img_name))
    w, h, _ = img_arr.shape
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("{} created".format(output_dir)) 
    img_name = os.path.splitext(os.path.split(parent_img_name)[1])[0]
    s = min(w, h) 
    if chip_size>s:
        return
    for k in range(10):
        random_c = np.random.randint(chip_size, s) 
        #Sampling random top left corner
        random_i = np.random.randint(0, w-random_c+1)
        random_j = np.random.randint(0, h-random_c+1)
        img_chip = img_arr[random_i:random_i+random_c, random_j:random_j+random_c, :]
        chip_name = os.path.join(output_dir, '{}_random_{}.jpeg'.format(img_name, k))
        img_chip = Image.fromarray(img_chip)
        img_chip.thumbnail((chip_size, chip_size), Image.ANTIALIAS)
        img_chip.save(chip_name)
        yield chip_name, [random_i, random_j, random_i+random_c, random_j+random_c], random_c


def ground_truth_parser(parent_img_name, chip_name, ground_truth, coords, crop_size, chip_size=400):
    """ Parses the ground truth file for the dataset and finds the bounding box annotations
        in the chip of interest and saves them into a json file
    """
    chip_ground_truth = []
    offset_row = coords[0]
    offset_col = coords[1]
    for obj in ground_truth:
        bbox, label = obj
        if bbox[0]>= coords[0] and bbox[1]>= coords[1]:
            if bbox[2]<= coords[2] and bbox[3]<= coords[3]:
                x_min = max((float(bbox[0]) - offset_col) / float(crop_size), 0)
                y_min = max((float(bbox[1]) - offset_row) / float(crop_size), 0)
                x_max = min((float(bbox[2]) - offset_col) / float(crop_size), 1)
                y_max = min((float(bbox[3]) - offset_row) / float(crop_size), 1)
                chip_ground_truth.append([[x_min, y_min, x_max, y_max], label])

    with open('{}{}'.format(os.path.splitext(chip_name)[0], '.json'), 'w') as output_file:
        json.dump(chip_ground_truth, output_file)

def main():
    args_set = get_parser().parse_args()
    imgs_name = glob.glob(os.path.join(args_set.images_dir, '*.jpeg'))
    print(len(imgs_name))
    for parent_img_name in imgs_name:
        print("Treating", parent_img_name)
        chipper = image_chipper(parent_img_name, args_set.chip_size, args_set.output_dir)
        with open(os.path.splitext(parent_img_name)[0]+".json", 'r') as annot_file:
            ground_truth = json.load(annot_file)
        for i in chipper:
            chip_ground_truth = ground_truth_parser(parent_img_name, i[0], ground_truth, i[1], i[2], args_set.chip_size)

if __name__ == '__main__':
    print("GO")
    main()
