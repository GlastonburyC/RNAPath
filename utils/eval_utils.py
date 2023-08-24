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

def eval(dataset, args, ckpt_path):
    # Model initialization (weights uploading from checkpoint)
    model = initiate_model(args, ckpt_path)
    
    print('Init Loaders')
    # dataloader
    loader = get_simple_loader(dataset)
    # summary -> real test function
    patient_results, test_error, auc, df, _ = summary(model, loader, args)
    print('test_error: ', test_error)
    print('auc: ', auc)
    return model, patient_results, test_error, auc, df

def summary(model, loader, args):
    print('Summary args: ', args)
    ## Modification to orginal code: check if we are evaluating the model on the test set
    if args.split == 'test':
        test = True
    else:
        test = False

    if test:
        slideFeatures = {}
    acc_logger = Accuracy_Logger(n_classes=args.n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), args.n_classes))
    all_labels = np.zeros(len(loader))
    all_preds = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}
    for batch_idx, (data, label) in enumerate(loader):
        data, label = data.to(device), label.to(device)
        # ID of the actual slide
        slide_id = slide_ids.iloc[batch_idx]
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, results_dict = model(data)

        if test:
            slide_level_embeddings = results_dict['features']
            # Slide-level embedding: contains the SLIDE features (512-dimensional)
            slideFeatures[slide_id] = slide_level_embeddings.to('cpu')

        
        acc_logger.log(Y_hat, label)
        
        probs = Y_prob.cpu().numpy()

        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        all_preds[batch_idx] = Y_hat.item()
        
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        
        error = calculate_error(Y_hat, label)
        test_error += error

    del data
    test_error /= len(loader)

    aucs = []
    if len(np.unique(all_labels)) == 1:
        auc_score = -1

    else: 
        if args.n_classes == 2:
            auc_score = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
            for class_idx in range(args.n_classes):
                if class_idx in all_labels:
                    fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                    aucs.append(auc(fpr, tpr))
                else:
                    aucs.append(float('nan'))
            if args.micro_average:
                binary_labels = label_binarize(all_labels, classes=[i for i in range(args.n_classes)])
                fpr, tpr, _ = roc_curve(binary_labels.ravel(), all_probs.ravel())
                auc_score = auc(fpr, tpr)
            else:
                auc_score = np.nanmean(np.array(aucs))

    results_dict = {'slide_id': slide_ids, 'Y': all_labels, 'Y_hat': all_preds}
    for c in range(args.n_classes):
        results_dict.update({'p_{}'.format(c): all_probs[:,c]})
    df = pd.DataFrame(results_dict)
    # Store data (serialize)
    with open('./' + args.save_dir + '/' + 'slideFeatures.pickle', 'wb') as handle:
        pickle.dump(slideFeatures, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return patient_results, test_error, auc_score, df, acc_logger
