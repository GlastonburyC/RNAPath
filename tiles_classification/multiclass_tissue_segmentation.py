from wsi_utils import get_features_h5
from sklearn.neighbors import KNeighborsClassifier
import os
import numpy as np
import argparse
import openslide
from PIL import ImageDraw
from wsi_utils import slide_to_scaled_pil_image
import yaml
import pandas as pd



parser = argparse.ArgumentParser()
parser.add_argument("--tissue_name", type=str, default=None)
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--features_dir", type=str, default=None)
parser.add_argument("--slides_dir", type=str, default=None)
args = parser.parse_args()

tissue = args.tissue_name
SCALE_FACTOR = 8
TILE_SIZE = 128

# Open clusters yaml and load tissue classes with their corresponding colors for visualization
f = open("clusters.yaml", "r")
doc = yaml.load(f, Loader=yaml.FullLoader)
classes = doc[tissue]['classes']
colors = doc[tissue]['colors']

# Load h5 file containing labelled clusters (output of define_clusters.py) for kNN
labels, features = get_features_h5(f'./{tissue}_clusters.h5')

# Load kNN model
model = KNeighborsClassifier(n_neighbors=200, weights= 'distance')
model.fit(features, labels)


def slide_segmentation(features, coords, slidename, downscaled_img, tissue, output_dir):

    draw = ImageDraw.Draw(downscaled_img, "RGBA")

    classes_df = []

    # for each patch
    for patch in range(features.shape[0]):
        # get patch features
        patch_features = features[patch].reshape(1, -1)
        # kNN classification
        label = model.predict(patch_features)

        s = int(coords[patch][0] / SCALE_FACTOR), int(coords[patch][1] / SCALE_FACTOR)
        s = np.array([s[0], s[1], s[0] + int(TILE_SIZE/SCALE_FACTOR), s[1] + int(TILE_SIZE/SCALE_FACTOR)])
        
        # color patch by class
        draw.rectangle(((s[0], s[1]), (s[0] + int(TILE_SIZE/SCALE_FACTOR), s[1] + int(TILE_SIZE/SCALE_FACTOR))), fill=tuple(colors[classes[int(label)]]), outline = tuple(colors[classes[int(label)]]))
        classes_df.append([coords[patch][0], coords[patch][1], int(label)])


    pd.DataFrame(classes_df, columns=['coord_x', 'coord_y', 'label']).to_csv(os.path.join(output_dir, tissue, slidename + '.csv'), index=False)    
    downscaled_img.save(os.path.join(output_dir, tissue, slidename + '.jpeg'))

# Slide names are retrieved using 
df = pd.read_csv('../resources/slides_dataset.csv')

# List of slide names to iterate
SLIDES = df[df.tissue == tissue].slide_id.tolist()

SLIDES = SLIDES[:2]

# Create output directory
os.makedirs(os.path.join(args.output_dir, tissue), exist_ok=True)

print(f'Tissue: {tissue}', f'number of slides: {len(SLIDES)}')


for slidename in SLIDES:
    
    # Features h5 file
    h5_file = f'{args.features_dir}/h5_files/{slidename}.h5'

    print('Slide name: ', slidename, flush=True)

    # Get features and coordinates from h5
    coords_tile, features_tile = get_features_h5(h5_file)
    coords_tile, features_tile = np.array(coords_tile), np.array(features_tile)

    # Open slide
    slide = openslide.open_slide(f'{args.slides_dir}/{tissue}/{slidename}.svs')
    # Get downscaled version for visualization
    downscaled_img, _ = slide_to_scaled_pil_image(slide, SCALE_FACTOR=8)
    # Multiclass segmentation by tiles classification
    slide_segmentation(features_tile, coords_tile, slidename, downscaled_img, tissue, args.output_dir)
