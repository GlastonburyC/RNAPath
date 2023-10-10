import os
import torch
import argparse
import datetime
import glob
import openslide
import math
from preprocessing_utils import  get_coords_h5
from model import FeaturesExtraction_Vit
from dataset import Tiles_Bag
from torch.utils.data import DataLoader
import h5py
import random
from torch.nn.functional import cosine_similarity
import yaml
import pandas as pd


parser = argparse.ArgumentParser('features-extraction', add_help=False)
# Job index
parser.add_argument('-index', type=str, default=0, help='index of actual job')  
# Number of jobs
parser.add_argument('-num_tasks', type=str, default=1, help='number of tasks')
# get args
args = parser.parse_args()


# Features of white patches (to discard them through cosine similarity)
file = h5py.File(f'./white_patch.h5', 'r')
WHITE_FEATURES = file['features'][:]
WHITE_FEATURES = torch.from_numpy(WHITE_FEATURES)


def extract_features(chunk, doc):
    checkpoint_path = doc['args']['checkpoint_path']
    output_dir = doc['args']['output_dir']
    coords_dir = doc['args']['coords_dir']
    batch_size = doc['args']['batch_size']
    tile_size = doc['args']['tile_size']


    print('Jobs starting...', flush=True)
    # === DATA === #
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # === FEATURES EXTRACTION MODEL === #
    features_extraction_class = FeaturesExtraction_Vit(arch='vit_small', pretrained_weights=checkpoint_path)
    features_extraction_class.model.eval()
    features_extraction_class.model.to(DEVICE)

    # Create output dir if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output dir will contain two subfolders: one for h5 files, one for pt files
    features_dir_h5 = os.path.join(output_dir, 'h5_files')
    features_dir_pt = os.path.join(output_dir, 'pt_files')

    # Create subfolders if they don't exist
    os.makedirs(features_dir_h5, exist_ok=True)
    os.makedirs(features_dir_pt, exist_ok=True)
    


    # Iteration over list of slidenames
    for slidename in chunk:
        
        print(f'Slide {chunk.index(slidename) + 1}/{len(chunk)} - {slidename}', flush = True)
        
        # get slide name without extension ('/path/to/slides/dir/slidename.ext') 
        slidename_noext = slidename.split('/')[-1].rsplit('.', 1)[0]

        # Open slide
        slide = openslide.open_slide(slidename)

        # Output features paths (for pt and h5)
        pt_path = os.path.join(features_dir_pt , slidename_noext + f'.pt')
        h5_path = os.path.join(features_dir_h5, slidename_noext + f'.h5')


        if os.path.exists(pt_path):
            print(f'Slide {slidename_noext} -  already processed')
            continue

        start = datetime.datetime.now()
        # Open coordinates file (assuming the file is named 'slidename_no_ext.coords.h5')
        coords_file = os.path.join(coords_dir, slidename_noext + f'.coords.h5')

        # get tiles coordinates
        coords = get_coords_h5(coords_file)

        # dataset of tiles and dataloader
        dataset = Tiles_Bag(slide=slide, tile_size = tile_size, h5=coords_file)
        loader = DataLoader(dataset = dataset, batch_size = batch_size, num_workers=4, shuffle=False)
        
        # empty tensors to store patch features (f) and coordinates (c)
        f = torch.empty(0, 384)
        c = torch.empty(0,2)
        

        # load tile (PIL image) and coordinates
        for tile, coords in loader:
            col, row = coords
            # extract features
            features = features_extraction_class.extractFeatures(tile, device=DEVICE)
            # just consider patches having a cosine similarity < 0.70 with the features of a white patch
            tissue_patches = torch.where(cosine_similarity(WHITE_FEATURES, torch.from_numpy(features)) <= 0.70)[0].tolist()
            
            features = features[tissue_patches]
            col = col[tissue_patches]
            row = row[tissue_patches]

            # concatenate coordinates and features
            c = torch.cat((c, torch.concat((col.unsqueeze(0), row.unsqueeze(0)), dim=0).T))
            f = torch.cat((f, torch.from_numpy(features)))


        # All the patches have been processed: store features and coordinates   
        features = f
        coords = c

        # H5 file will contain two fields: 'coords' -> (K,2) array and 'features' -> (K, 384) array where K= number of patches
        with h5py.File(h5_path, 'w') as fi:
            fi.create_dataset('coords', data=coords.numpy())
            fi.create_dataset('features', data=features.numpy())

        
        torch.save(features, os.path.join(features_dir_pt , slidename_noext + f'.pt'))


        end = datetime.datetime.now()

        print(f'Time required: {end - start}; shape: {features.shape}', flush=True)                            

if __name__ == "__main__":
    #parser = argparse.ArgumentParser('features-extraction', parents=[get_args_parser()])
    #args = parser.parse_args()

    # Read slides dir / path to txt file  and WSI extension from config
    config_file = open("config.yaml", "r")
    doc = yaml.load(config_file, Loader=yaml.FullLoader)
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

    extract_features(args, chunk, doc)
