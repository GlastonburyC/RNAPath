import pandas as pd
import argparse
import yaml
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--tissue_name", type=str, default=None)
parser.add_argument("--segmentation_dir", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
args = parser.parse_args()

# Load classes from yaml
f = open("../clusters.yaml", "r")
doc = yaml.load(f, Loader=yaml.FullLoader)
classes = doc[args.tissue_name]['classes']

# output column names correspond to the classes defined for the tissue
column_names = classes.copy()

# Get slides from csv
slides_df = pd.read_csv('../resources/slides_dataset.csv')
slides_df = slides_df[slides_df.tissue == args.tissue_name]
slides = slides_df.slide_id.tolist()

# Output dataframe
out = pd.DataFrame()
out['slide_id'] = slides

# Iteration over classes
for index, target in enumerate(classes):
    print(f'{target}', flush=True)
    val = []
    # for each WSI, we compute the classes proportions in the sample dividing by the sample size
    for slide_id in slides:
        c = pd.read_csv(f'{args.segmentation_dir}/{args.tissue_name}/{slide_id}.csv')
        proportion = (c.label == index).sum() / (c.shape[0] + 1e-8)
        val.append(proportion)
    
    out[target] = val

# output compositional idps
out.to_csv(f'./{args.output_dir}/{args.tissue_name}_compositional.csv', index=False)