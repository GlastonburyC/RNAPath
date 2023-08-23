#!/bin/bash

# SLURM options
#SBATCH --time=16:00:00 #define runtime
#SBATCH --mem=32G        #define amount of ram
#SBATCH --partition=gpu 
#SBATCH --mem=64G 
#SBATCH --gpus-per-node=2

python3 run.py --model ../../checkpoint_ts.pth --gpu_id 0,1 --save_folder /hpc/dhl_ec/VirtualSlides/cglastonbury/PathProfiler/tissue-segmentation/masks/ --slide_dir /hpc/dhl_ec/VirtualSlides/HE/_tif/_tifs/ --mask_magnification 2.5 --mpp_level_0 0.25
 
