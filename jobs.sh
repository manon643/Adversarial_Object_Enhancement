#!/usr/bin/env	bash
#SBATCH	-p	k6000
#SBATCH --mem   4000            # memory pool per process

python36 -u train_module.py --train_dir=../VOC2012/SmallImages/train/ --test_dir=../VOC2012/VerySmallImages/val/ --summary_dir=checkpoints/ --run "new" 
