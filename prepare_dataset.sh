#!/usr/bin/env	bash
#SBATCH	-p	k80

#python36 prepare_dataset/prepare_small.py --images_dir ../VOC2012/JPEGImages/train --output_dir ../VOC2012/SmallImages/train
python36 prepare_dataset/prepare_samples.py --images_dir ../VOC2012/JPEGImages/train --output_dir ../VOC2012/ResizedImages/train
python36 prepare_dataset/prepare_samples.py --images_dir ../VOC2012/JPEGImages/val --output_dir ../VOC2012/ResizedImages/val
