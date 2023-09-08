from wsi_utils import get_features_h5
from sklearn.neighbors import KNeighborsClassifier
import os
import numpy as np
import openslide
from wsi_utils import slide_to_scaled_pil_image
import yaml
from PIL import Image
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--tissue_name", type=str, default=None)
parser.add_argument("--checkpoint_path", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--features_dir", type=str, default=None)
parser.add_argument("--slide_name", type=str, default=None)
args = parser.parse_args()

tissue = args.tissue_name
SCALE_FACTOR = 8
TILE_SIZE = 128

f = open("clusters.yaml", "r")
doc = yaml.load(f, Loader=yaml.FullLoader)
classes = doc[tissue]['classes']
colors = doc[tissue]['colors']

labels, features = get_features_h5(f'./{tissue}_clusters.h5')
model = KNeighborsClassifier(n_neighbors=200, weights= 'distance')
model.fit(features, labels)

def slide_fine_grained_segmentation(downscaled_img, slidename, tissue):

    # Array to store pixel-level class assignments (in case of multiple patches overlapping the same pixel, majority
    # voting will be used)
    res = np.zeros((downscaled_img.size[1], downscaled_img.size[0], len(classes)))

    # Mask for background
    mask = np.zeros((downscaled_img.size[1], downscaled_img.size[0])) != 0

    # iterate over patch sets
    for patch_set in [0, 32, 64, 96]:
        f = h5py.File(f'{args.features_dir}/{slidename}_{patch_set}.h5')
        coords = f['coords'][:]
        features = f['features'][:]

        # iterate over patches in the current patch set
        for patch in range(features.shape[0]):
            patch_features = features[patch].reshape(1, -1)
            y, x = coords[patch].astype(int)
            x = int(x/SCALE_FACTOR)
            y = int(y/SCALE_FACTOR)

            # classify patch
            label = model.predict(patch_features)

            # increase class counter
            res[x:x+ int(TILE_SIZE/SCALE_FACTOR), y:y+ int(TILE_SIZE/SCALE_FACTOR), label] += 1

            # set the patch pixels to tissue
            mask[x:x+ int(TILE_SIZE/SCALE_FACTOR), y:y+ int(TILE_SIZE/SCALE_FACTOR)] = True

    # Class assignment by majority voting on overlapping patches
    res = np.argmax(res, axis = 2)
    
    # put background to -2; background pixels will not be modified in the output image
    res[mask == False] = -2

    # output image
    img_out = np.zeros((downscaled_img.size[1], downscaled_img.size[0], 3))
    # raw WSI
    img_base = np.array(downscaled_img.convert('RGB'))

    # overlaying colors to patches for each class
    for c in range(0, len(classes)):
        img_out[res == c] = img_base[res == c] * 0.5 + np.array(colors[classes[c]][:3]) * 0.5

    img_out[res == -2] = img_base[res == -2]

    # save image
    Image.fromarray(img_out.astype(np.uint8)).save(f'./{slidename}_fine_grained_segmentation.pdf', format='pdf')


if __name__ == 'main':

    slidename = args.slide_name

    print(f'Tissue: {tissue}; slidename: {slidename}')
    slide = openslide.open_slide(f'{args.slides_dir}/{tissue}/{slidename}.svs')
    print('Slide name: ', slidename, flush=True)
    downscaled_img, _ = slide_to_scaled_pil_image(slide, SCALE_FACTOR=8)
    patch_number = slide_fine_grained_segmentation(downscaled_img, slidename, tissue)
