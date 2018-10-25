#!/usr/bin/env	bash
#SBATCH	-p	k6000
#SBATCH --mem   4000            # memory pool per process

python36 -u train.py --train_dir=../VOC2012/ResizedImages/train/ --test_dir=../VOC2012/ResizedImages/val/ --train_summary_dir=checkpoints/train/ --test_summary_dir=checkpoints/test/ --ckpt_dir=checkpoints/ 
