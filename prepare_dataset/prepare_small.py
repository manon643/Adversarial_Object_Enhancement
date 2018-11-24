from PIL import Image, ImageDraw
import numpy as np
import json
from pathos.multiprocessing import ProcessingPool as Pool
#from multiprocessing import Pool
import argparse
import glob
import os
import pdb
import sys

def get_parser():
    """ This function returns a parser object """
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--images_dir', type=str)
    aparser.add_argument('--overlap', type=int, default=0.1)
    aparser.add_argument('--output_dir', type=str, default=0)

    return aparser

def image_chipper(parent_img_name, output_dir):
    """ Chips the full images and saves them into the output directory """
    img_arr = np.asarray(Image.open(parent_img_name))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print("{} created".format(output_dir)) 
    img_name = os.path.splitext(os.path.split(parent_img_name)[1])[0]
    img_name = os.path.join(output_dir, img_name+'.jpeg')
    return img_name, img_arr.shape[0], img_arr.shape[1] 

def function_img(output_dir):
    def aux(parent_img_name):
        name, w, h = image_chipper(parent_img_name, output_dir)
        with open(os.path.splitext(parent_img_name)[0]+".json", 'r') as annot_file:
            ground_truth = json.load(annot_file)
        gt = []
        for obj in ground_truth:
            gt.append([obj[1], [obj[0][0]/h, obj[0][1]/w, obj[0][2]/h, obj[0][3]/w]])
        with open(name[:-5]+".json", "w") as output_file:
            json.dump(gt, output_file)
    return aux

if __name__ == '__main__':
    args_set = get_parser().parse_args()
    imgs_name = glob.glob(os.path.join(args_set.images_dir, '*.jpeg'))
    print(len(imgs_name))
    aux = function_img(args_set.output_dir)
    #for img in imgs_name:
    #    aux(img)
    with Pool(10) as p:
        for i,_ in enumerate(p.imap(aux, imgs_name)):
            sys.stderr.write('\rdone {0:%}'.format(float(i)/len(imgs_name)))
