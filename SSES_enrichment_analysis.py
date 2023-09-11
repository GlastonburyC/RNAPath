import pandas as pd
import yaml
import torch
import numpy as np
import argparse
import h5py
import pandas as pd
import h5py
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--tissue_name", type=str, default=None)
parser.add_argument("--tissue_code", type=str, default=None)
parser.add_argument("--logits_dir", type=str, default=None)
parser.add_argument("--segmentation_dir", type=str, default=None)
parser.add_argument("--features_dir", type=str, default=None)
args = parser.parse_args()    

# File containig image derived phenotypes (tissue substructures or localised pathologies) of the annotated tissues
f = open("./resources/clusters.yaml", "r")
doc = yaml.load(f, Loader=yaml.FullLoader)
classes = doc[args.tissue_name]['clusters']

# Read tissue slides
slides = pd.read_csv('./resources/slides_dataset.csv')
slides = slides[slides.tissue == args.tissue_name]['slide_id']
slides = slides.tolist()

# Genes for enrichment; by default, enrichment scores are computed just for genes
# with validation r score > 0.5
genes = pd.read_csv(f'./results/{args.tissue_code}/report_val.txt', sep=' ')
genes = genes[genes['r_score'] > 0.5]

gene_ids = genes.gene_id.tolist()

# dataframe used to accumulate expression values in the target region
target_df = pd.DataFrame(np.zeros((len(gene_ids), len(classes))))
# dataframe used to accumulate bulk expression
total_df = pd.DataFrame(np.zeros((len(gene_ids), len(classes))))

for slidename in slides:
    if (slides.index(slidename) +1) % 20 == 0:
        print(f'{slidename}, {slides.index(slidename) + 1}/{len(slides)}', flush=True)

    # load patch logits
    patch_logits = torch.load(f'{args.logits_dir}/{args.tissue_name}/{slidename}.pt', map_location='cpu')
    patch_logits_df = pd.DataFrame(patch_logits).astype(np.float32)

    # load tile classes df
    tile_classes_df = pd.read_csv(f'{args.segmentation_dir}/{args.tissue_name}/{slidename}.csv')

    # read coordinates
    f = h5py.File(f'{args.features_dir}/h5_files/{slidename}.h5')
    coords = f['coords'][:].tolist()

    # for each image derived phenotype
    for phenotype_idx, idp in enumerate(classes):

        # get tiles having, as class, the current phenotype
        arr = np.array(tile_classes_df[tile_classes_df['label'] == phenotype_idx][['coord_x', 'coord_y']])
        arr = arr.tolist()

        # get the indexes of the tiles in the tissue substructure/pathology
        indexes = [i for i, x in enumerate(coords) if x in arr]

        if indexes != []:
            # store the average expression of tiles in the tissue substructure/pathology
            target_df[phenotype_idx] += patch_logits_df.loc[indexes].mean(0)
        
        # store the bulk expression of the sample
        total_df[phenotype_idx] += (patch_logits_df.mean(0) +1e-7)


# the enrichment analysis output is given by the ratio between local (inside the tissue substructure/pathology) and global expression
out = target_df/total_df
out.columns = classes
out.insert(0, 'gene_id', genes.gene_id.tolist())
out.insert(1, 'gene_name', genes.gene_desc.tolist())
out.to_csv(f'./SSES_enrichments/{args.tissue_name}_enrichment.csv', index=False)
