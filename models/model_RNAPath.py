import torch
import torch.nn as nn
from utils.utils import initialize_weights
import numpy as np



class RNAPath(nn.Module):
    def __init__(self, n_classes=0):
        super(RNAPath, self).__init__()
        
        """
            Model initialisation requires:
            - number of classes (number of genes (N))
            We train N linear regressors, one for each gene.
            
        """
        self.n_classes = n_classes
        self.dropout = torch.nn.Dropout(p=0.1)
        classifiers = [nn.Linear(384, 1) for i in range(self.n_classes)]
        self.classifiers = nn.ModuleList(classifiers)


        initialize_weights(self)

    def relocate(self):
        """
            Move model to device
        """
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifiers = self.classifiers.to(device)
        

    def forward(self, h, genes_list = None, return_patch_expression = False):

        """"
            gene_list: list of indexes of genes; the full list of genes is [0, #genes)
            return_patch_expression: whether to return patch-level scores; patch logits are reported as a matrix
            of shape (K, N), where K is the number of patches and N the number of genes
            
            Due to memory limitation, we just train on a subset of genes, therefore we split the genes in batches 
            and we provide a list of indexes at each iteration; during inference, this is not required, therefore we 
            regress the expression of all the genes of interest at the same time.
        
        """

        if genes_list is None:
            genes_list = list(range(self.n_classes))
        
        # apply drouput to the input patches representation matrix
        h = self.dropout(h)

        # compute patch-level expression
        patch_logits = torch.stack([self.classifiers[i](h).squeeze() for i in genes_list], dim=1)
        patch_logits = nn.ReLU()(patch_logits)

        # merge patch-level predicted RNASeq to slide-level RNASeq through mean pooling
        logits = torch.mean(patch_logits, dim=0)

        if return_patch_expression:
            return logits, patch_logits.reshape(-1, len(genes_list))
            

        return logits



