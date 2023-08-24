from __future__ import print_function

import argparse
import os

# internal imports
from utils.utils import *
from utils.core_utils import train
from datasets.dataset_generic import Generic_MIL_Dataset

# pytorch imports
import torch
import numpy as np
import pandas as pd
from utils.file_utils import save_pkl



def main(args):
    # create results directory if necessary
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)


    # seed for reproducibility
    seed_torch(args.seed)
        
    # Return splits from the csv file (the split dir is passed as argument, the number after the underscore
    # is equal to the number of the current fold (i)
    i = 0
    train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
    datasets = (train_dataset, val_dataset, test_dataset)
    # Train with the current split
    val_median_r, val_error, test_median_r, test_error  = train(datasets, i, args)

    results = {'val_median_r': val_median_r, 'val_error': val_error, 'test_median_r': test_median_r, 'test_error': test_error}
    
    #write results to pkl
    filename = os.path.join(args.results_dir, 'results.pkl'.format(i))
    save_pkl(filename, results)


# Training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')

parser.add_argument('--data_root_dir', type=str, default= "",  help='tiles representations main directory')
parser.add_argument('--max_epochs', type=int, default=200, help='maximum number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=1e-4, help='starting learning rate (default: 0.0001)')
parser.add_argument('--label_frac', type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--reg', type=float, default=1e-3, help='weight decay (default: 1e-5)')
parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--results_dir', default='./results', help='results directory (default: ./results)')
parser.add_argument('--split_dir', type=str, default=None, help='manually specify the set of splits to use, instead of infering from the task and label_frac argument (default: None)')
parser.add_argument('--log_data', action='store_true', default=False, help='log data using tensorboard')
parser.add_argument('--early_stopping', action='store_true', default=True, help='enable early stopping')
parser.add_argument('--bag_dropout', type=bool, default=True, help='apply dropout on bag instances')
parser.add_argument('--opt', type=str, choices = ['adam', 'sgd'], default='adam', help='optimizer')
parser.add_argument('--exp_code', type=str, help='experiment code for saving results')
parser.add_argument('--lr_scheduler', type=str, choices= ['plateau', 'constant'], default='plateau', help='learning rate scheduler')
parser.add_argument('--genes', default=[], help='list of genes for the current tissue')
parser.add_argument('--tissue_code', type=str, default=None, help='code to identify the GTEx tissue')


args = parser.parse_args()

# Manually specifying the list of genes
args.genes = pd.read_csv(f'./resources/gene_set_{args.tissue_code}.txt', sep=' ')['gene_id'].tolist()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

settings = {'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'seed': args.seed,
            'opt': args.opt}


print('\nLoad Dataset')

# Number of classes equals the number of genes to regress
args.n_classes=len(args.genes)

# dataset creation
dataset = Generic_MIL_Dataset(csv_path = 'resources/HE2RNA_dataset.csv',
                        data_dir= os.path.join(args.data_root_dir, ''),
                        shuffle = False, 
                        seed = args.seed, 
                        rna_seq_csv= './RNA-SEQ-Analysis/rnaseq_complete.csv',
                        print_info = True,
                        label_dict = {'gen_exp': 0},
                        genes = args.genes,
                        bag_dropout=args.bag_dropout,
                        ignore=[])


# Results directory

args.results_dir = os.path.join(args.results_dir, str(args.exp_code))

if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

# Splits directory
if args.split_dir is None:
    args.split_dir = os.path.join('splits', f'RNAPath_{args.tissue_code}')
else:
    args.split_dir = os.path.join('splits', args.split_dir)

print('split_dir: ', args.split_dir)
assert os.path.isdir(args.split_dir)

settings.update({'split_dir': args.split_dir})


with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    results = main(args)

    print("finished!")
    print("end script")


