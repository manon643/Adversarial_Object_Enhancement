import os 
import json
import argparse
import xml.etree.ElementTree as ET

def get_parser():
    """ This function returns a parser object """
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--images_dir', type=str, default="JPEG_Images",
                         help='Provide the training directory to the text file with file names and labels in it')
    aparser.add_argument('--splits_dir', type=str,
                         help='Provide the training directory to the text file with file names and labels in it')
    aparser.add_argument('--annot_dir', type=str,
                     help='Provide the checkpoint directory where the network parameters will be stored')

    return aparser

#def get_categories():
#    return ['aeroplane', 'bicycle', 'bird', 'boat',
#        'bottle', 'bus', 'car', 'cat', 'chair',
#        'cow', 'diningtable', 'dog', 'horse',
#        'motorbike', 'person', 'pottedplant',
#        'sheep', 'sofa', 'train',
#        'tvmonitor'] 

def main():
    # Parse the command line args
    args = get_parser().parse_args()

    data_dir = os.path.split(args.images_dir)[0]
    
    #categories = get_categories()
    for split in ["train", "val"]:
        directory = os.path.join(data_dir, split)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("{} created") 
        split_file = os.path.join(args.splits_dir, split+".txt")
        with open(split_file, "r") as f:
             for l in f.readlines():
                 name = l[:-1]
                 #Moving the image to the right directory
                 image_path = os.path.join(args.images_dir, name+".jpg")
                 #if not os.path.exists(image_path):
                 #    print(image_path)
                 #    return
                 #os.rename(image_path, os.path.join(directory, name+".jpeg"))
                 
                 #Reading xml and converting to JSON
                 json_list = [] 
                 annotations_path = os.path.join(args.annot_dir, name+".xml")
                 tree = ET.parse(annotations_path)
                 root = tree.getroot()

                 #Get size of image
                 size = root.find("size")
                 w, h = int(size.find("width").text), int(size.find("height").text)
                 for obj in root.findall("object"):
                      label = obj.find("name").text
                      bounding_box = obj.find("bndbox")
                      #Normalizing bbox
                      xmin = int(bounding_box.find("xmin").text)               
                      ymin = int(bounding_box.find("ymin").text)              
                      xmax = int(bounding_box.find("xmax").text)            
                      ymax = int(bounding_box.find("ymax").text)            
                      bbox = [xmin, ymin, xmax, ymax]
                      json_list.append((bbox, label)) 
                 #Writing to Json
                 json_name = os.path.join(directory, name+".json")
                 with open(json_name, "w") as output_file:	
                      json.dump(json_list, output_file)
        print("Split {} successfully completed".format(split))


if __name__ == '__main__':
    main()
