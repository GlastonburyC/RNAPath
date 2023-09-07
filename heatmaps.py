import torch
import h5py
import numpy as np
import openslide
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import time
from wsi_utils import slide_to_scaled_pil_image
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--gene_name", type=str, default=None)
parser.add_argument("--slide_id", type=str, default=None)
parser.add_argument("--tissue_name", type=str, default=None)
parser.add_argument("--tissue_code", type=str, default=None)
parser.add_argument("--save_dir", type=str, default='./heatmaps', help='path to save dir')
parser.add_argument("--features_dir", type=str, default='./features', help='path to features dir')
parser.add_argument("--patch_logits_dir", type=str, default='./patch_logits')
parser.add_argument("--results_dir", type=str, default='./results')
parser.add_argument("--slides_dir", type=str, default='./slides')
parser.add_argument("--multiple_patch_sets", type=bool, action='store_true', default=False)

args = parser.parse_args()

# Tissue name from GTEx
tissue = args.tissue_name
# Tissue code from RNAPath
tissue_code = args.tissue_code
# List of genes profiled for the actual tissue (the list has to match the list of genes stored in the patch logits files)
genes_list = pd.read_csv(os.path.join(args.results_dir, tissue_code, 'report_val.txt'), sep=' ')
genes_list = genes_list[genes_list.r_score > 0.5].gene_desc.tolist()


def plot_gene(gene_name, slide_id, tissue_name):
    print(slide_id, flush=True)
    SCALE_FACTOR = 8
    # Index of gene
    gene_idx = genes_list.index(gene_name)
    # path to slide svs
    slide_path = f'{args.slides_dir}/{tissue_name}/{slide_id}.svs'
    # open slide WSI
    slide = openslide.open_slide(slide_path)
    # get downscaled version of the slide
    downscaled_img, _ = slide_to_scaled_pil_image(slide, SCALE_FACTOR=SCALE_FACTOR)
    slide.close()
    # matrix to store logits
    logits = np.zeros((downscaled_img.size[1], downscaled_img.size[0]))
    # mask to store tissue locations
    mask = np.zeros((downscaled_img.size[1], downscaled_img.size[0]))

    if args.multiple_patch_sets:
    # for each coordinate set
        for patch_set in [0, 32, 64, 96]:
            # open patch-level regressed exprssion values
            with h5py.File(os.path.join(args.features_dir, 'h5_files', f'{slide_id}_{patch_set}.h5'), 'r') as f:
                coords = f['coords'][:]
                patch_logits = torch.load(f'{args.patch_logits_dir}/{tissue_name}/{slide_id}_{patch_set}.pt', map_location='cpu').to(torch.float)
            
            # patch-level expression logits of the current gene
            pl = patch_logits[:, gene_idx]

            # Iteration over patches for the current coordinate set
            # Sum to each spatial location the local expression value and count the number of times that location is summed;
            # This will allow to correctly divide each final pixel value by the number of patches covering it.

            for coord, p in zip(coords, pl):
                y, x = map(int, coord)
                x //= SCALE_FACTOR
                y //= SCALE_FACTOR
                xspan, yspan = (slice(x, x+128//SCALE_FACTOR), slice(y, y+128//SCALE_FACTOR))
                logits[xspan, yspan] += p.numpy().item()
                mask[xspan, yspan] += 1
    
    else:

        # if a single patch set is used, the patch-level logits are loaded and added to the downscaled array
        with h5py.File(os.path.join(args.features_dir, 'h5_files', f'{slide_id}.h5'), 'r') as f:
            coords = f['coords'][:]
            patch_logits = torch.load(f'{args.patch_logits_dir}/{tissue_name}/{slide_id}.pt', map_location='cpu').to(torch.float)
        pl = patch_logits[:, gene_idx]

        for coord, p in zip(coords, pl):
            y, x = map(int, coord)
            x //= SCALE_FACTOR
            y //= SCALE_FACTOR
            xspan, yspan = (slice(x, x+128//SCALE_FACTOR), slice(y, y+128//SCALE_FACTOR))
            logits[xspan, yspan] += p.numpy().item()
            mask[xspan, yspan] += 1
            
    # average logits across each spatial location
    logits /= (mask + 1e-9)

    arr_min = np.min(logits[mask >= 1])
    arr_max = np.max(logits[mask >= 1])

    # normalize logits array
    arr = (logits - arr_min) / (arr_max - arr_min)

    # Convert the logits array into heatmap
    cmap = plt.get_cmap('coolwarm')
    rgba = cmap(arr, bytes=True)

    # Convert the 3D array to a PIL image
    img = Image.fromarray(rgba, 'RGBA')
    mask = mask > 0
    alpha = 170 * mask.astype(np.uint8)
    img.putalpha(Image.fromarray(alpha, 'L'))
    # Overlay heatmap to raw image
    overlaid_img = Image.alpha_composite(downscaled_img.convert('RGBA'), img).convert('RGB')
    return overlaid_img

    
    
if __name__ == "__main__":
    start = time.process_time()
    print('Computing heatmap...')
    overlaid_img = plot_gene(args.gene_name, args.slide_id, tissue_name=tissue)
    overlaid_img.save(f'./{args.save_dir}/{args.gene_name}_{args.slide_id}.jpeg')
    end = time.process_time()
    print('Done!')
    elapsed_time = (end - start)
    print(f'Elapsed time: {end - start} s')


