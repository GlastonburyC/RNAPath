import numpy as np

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_mil import MIL_fc, MIL_fc_mc
from HE2RNA_GAMIL_all_genes.models.model_RNAPath import RNAPath
import pdb
import os
import pandas as pd
from utils.utils import *
#from utils.core_utils import Accuracy_Logger
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

def initiate_model(args, ckpt_path):
    print('Init Model')    
    model_dict = {'n_classes': args.n_classes}
    
    model = RNAPath(**model_dict)

    #print_network(model)
    if torch.cuda.is_available() is False:
        ckpt = torch.load(ckpt_path, map_location='cpu')
    else:
        ckpt = torch.load(ckpt_path)
    print('checkpoint loaded ...')
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})

    print('loading state dict ...')
    model.load_state_dict(ckpt_clean, strict=True)
    print('relocating ...')

    model.relocate()
    model.eval()
    return model

