import os
import argparse
import datetime
from PIL import Image
import numpy as np
import openslide
from PIL import ImageDraw
from preprocessing.preprocessing_utils import slide_to_scaled_pil_image, save_hdf5
from dataset import Whole_Slide_Mask
from torch.utils.data import DataLoader
import h5py
import time
import yaml
import pandas as pd
import glob
import math


parser = argparse.ArgumentParser()
# Index of current job
parser.add_argument('-index', type=str, default=0, help='index of actual job')     
# Number of tasks
parser.add_argument('-num_tasks', type=str, default=1, help='number of tasks')
args = parser.parse_args()

def get_args_parser():
    return parser

config_file = open("config.yaml", "r")
doc = yaml.load(config_file, Loader=yaml.FullLoader)

def tile_slides(chunk, doc):
    print('Jobs starting...')
    # === DATA === #
    slides = chunk

    # See config.yaml for information about these parameters
    tile_size = doc['patching_args']['tile_size']
    patches_dir = doc['patching_args']['patches_dir']
    save_thumbnails =  doc['patching_args']['save_thumbnails']
    thumbnails_dir =  doc['patching_args']['thumbnails_dir']

    masks_dir = doc['args']['masks_dir']

    print(f'Tile size: {tile_size}')
    print(f'Masks dir: {masks_dir}')
    print(f'Patches output dir: {patches_dir}')

    # Coordinates directory
    patches_dir = os.path.join(patches_dir, f'{tile_size}')
    os.makedirs(patches_dir, exist_ok=True)
    
    # Thumbnails directory
    if save_thumbnails:
        thumbnails_dir = os.path.join(thumbnails_dir, f'{tile_size}')
        os.makedirs(thumbnails_dir, exist_ok=True)

    # Scale factor for the thumbnails
    SCALE = 8

    # Iteration over list of slidenames
    for slidename in slides:
        start = datetime.datetime.now()

        # Slide name without extension
        slidename_noext = slidename.split('/')[-1].rsplit('.',1)[0]
        
        print(f'Tiling: slide {slidename_noext} {slides.index(slidename) + 1}/{len(slides)}', flush=True)

        # Open the slide
        slide = openslide.open_slide(slidename)

        # Read the mask
        mask = Image.open(os.path.join(masks_dir, slidename_noext + '.jpg'))
        mask = np.array(mask)

        # Downscaled version of the slide 
        downscaled_img, _ = slide_to_scaled_pil_image(slide, SCALE_FACTOR=SCALE)
        
        # Draw object
        draw = ImageDraw.Draw(downscaled_img)
        
        if os.path.exists(os.path.join(patches_dir,  slidename_noext + f'.coords.h5')):
                print(f'{slidename_noext} - already processed')
                continue
        
        # Whole Slide Bag Dataset
        slide_dataset = Whole_Slide_Mask(slide, tile_size = tile_size, mask = mask, overlap = 0.65)
        # Tiles Loader
        tiles_loader = DataLoader(dataset= slide_dataset, batch_size=800, num_workers=2, shuffle=False)

        coords = np.zeros((0,2))
        for (col, row), percentage  in (tiles_loader):
            # Indexes of patches containing having more than 50% of detected tissue
            tissue_indexes = (percentage[:] > 0.5).nonzero(as_tuple=False)

            # For each of the patches detected as tissue, save the couple of coordinates into a h5 file
            for t_idx in list(tissue_indexes):

                coords = np.concatenate((coords, np.array([int(col[t_idx]), int(row[t_idx])]).reshape(1,2)))

                if save_thumbnails:
                    s = (int(col[t_idx]/SCALE), int(row[t_idx]/SCALE))
                    draw.rectangle(((s[0], s[1]), (s[0] + tile_size/SCALE, s[1] + tile_size/SCALE)), fill=None, outline='green', width=1)
        
        h5_path = os.path.join(patches_dir , slidename_noext + f'.coords.h5')
        with h5py.File(h5_path, 'w') as fi:
            fi.create_dataset('coords', data=coords)


    end = datetime.datetime.now()
    print(f'Time required for slide {slidename_noext}: {end - start}', flush=True)
        
    if save_thumbnails:
        downscaled_img.save(os.path.join(thumbnails_dir,  slidename_noext + '_patches.png'))



if __name__ == '__main__':
    t = time.time()

    slides_dir = doc['args']['slides_dir']
    file_list_txt = doc['args']['file_list_txt']
    ext = doc['args']['ext']

    # read slides to process: if the txt filed is None, it will process all the slides in slides_dir
    if file_list_txt is None:
        slides = glob.glob(os.path.join(slides_dir, f'*.{ext}'))
    else:
        slides = pd.read_csv(file_list_txt, sep = ' ', header=None)[0].tolist()

    # Splitting slides into different chunks if multiple jobs are launched (see sbatch)
    # Each job has an index and will process the chunk corresponding to the index itself
    # The chunks size clearly depends on the number of jobs. If just a job is run, all the slides
    # will be assigned to that job

    os.makedirs(doc['args']['masks_dir'], exist_ok=True)
    num_tasks = int(args.num_tasks)
    i = int(args.index)
    print('Number of files:', len(slides), flush=True)
    files_per_job = math.ceil(len(slides)/num_tasks)
    chunks = [slides[x:x+ files_per_job] for x in range(0, len(slides), files_per_job )]
    
    if i < len(chunks):
        chunk = chunks[i]
        print(f'Chunk {i}: {len(chunk)} slides', flush= True)
    else:
        chunk = []

    tile_slides(chunk, doc)
    print('Tissue segmentation done (%.2f)s' % (time.time() - t))
