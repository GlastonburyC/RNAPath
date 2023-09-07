import os
import torch
from HE2RNA_GAMIL_all_genes.models.model_RNAPath import RNAPath
from utils.eval_utils import initiate_model
import argparse
import os
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--tissue_name", type=str, default=None)
parser.add_argument("--tissue_code", type=str, default=None)
parser.add_argument("--features_dir", type=str, default='./features', help='path to features dir')
parser.add_argument("--output_dir", type=str, default='./patch_logits')
parser.add_argument("--results_dir", type=str, default='./results')

parser.add_argument("--ckpt_path", type=str, default=None)
parser.add_argument("--multiple_patch_sets", type=bool, action='store_true', default=False)

args = parser.parse_args()

# Set n classes = n genes for the current tissue
args.n_classes = pd.read_csv(f'./resources/genes_{args.tissue_code}.txt', header=None).shape[0]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Select genes; by default, we make inference on genes with validation r score > 0.5
genes = pd.read_csv(os.path.join(args.results_dir, args.tissue_code, 'report_val.txt'), sep=' ')
genes = genes[genes['r_score'] > 0.5].index.tolist()

print(f'Number of genes with r > 0.5: {len(genes)}')

# Create folder to store patch logits of the specified tissue if it doesn't exist
os.makedirs(os.path.join(args.output_dir, args.tissue_name), exist_ok=True)


def compute_patch_logits():
    slide_data = pd.read_csv('resources/slides_dataset.csv')
    model = initiate_model(args, args.ckpt_path)
    model.eval()

    slides = slide_data[slide_data.tissue == args.tissue_name].slide_id.tolist()

    for idx, slide_id in enumerate(slides):

        print(f'{slide_id} {idx+1}/{len(slides)}')

        # if you have multiple patch sets per slide

        if args.multiple_patch_sets:
             
            for patch_set in [0, 32, 64, 96]:
                with torch.no_grad():
                        features_path = os.path.join(args.features_dir, 'pt_files', f'{slide_id}_{patch_set}.pt')
                        save_path = os.path.join(args.output_dir, args.tissue_name, f'{slide_id}_{patch_set}.pt')
                        features = torch.load(features_path)
                        # inference
                        logits, patch_logits = model(features.cuda(), return_patch_expression=True)
                        # saving patch logits (K, N_Genes)
                        torch.save(patch_logits.type(torch.float16)[:, genes], save_path)

        # if you just have a patch set per slide

        else:
            with torch.no_grad():
                    features_path = os.path.join(args.features_dir, 'pt_files', f'{slide_id}.pt')
                    save_path = os.path.join(args.output_dir, args.tissue_name, f'{slide_id}.pt')
                    features = torch.load(features_path)
                    # inference
                    logits, patch_logits = model(features.cuda(), return_patch_expression=True)
                    # saving patch logits (K, N_Genes)
                    torch.save(patch_logits.type(torch.float16)[:, genes], save_path)
                     


compute_patch_logits()