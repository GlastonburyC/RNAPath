from html.entities import html5
from torch.utils.data import Dataset
import numpy as np
from wsi_preprocessing.preprocessing_utils import get_coords_h5
import math
import glob
import os
import torch
import torchstain
import torchvision.transforms as transforms
from PIL import Image

class Whole_Slide_Mask(Dataset):

    def __init__(self, slide, tile_size, mask, overlap = 0.0):
        
        # slide : openslide WSI
        # tile_size : dimension of each squared patch (e.g. 256 or 512)
        # transform : transform function for the tiles classification model
        # mask: black and white mask output by the segmentation script
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        self.normalizer.HERef = torch.tensor([[0.5542, 0.1667], [0.7380, 0.8497], [0.3757, 0.4837]])
        self.normalizer.maxCRef = torch.tensor([2.8576, 1.9241])
        self.T = transforms.ToTensor()
        
        self.slide = slide
        self.dimensions = slide.dimensions


        self.cols = np.arange(0, int(self.dimensions[0]), int(tile_size*(1-overlap)))
        self.rows = np.arange(0, int(self.dimensions[1]), int(tile_size*(1-overlap)))
        self.number_of_tiles = len(self.cols) * len(self.rows)
        self.tile_size = tile_size
        self.mask = mask
        
        # Scale factor between original WSI dimensions and segmentation masks 
        # Higher the magnification factor set for the segmentation, lower the scale difference between original and downscaled
        # WSI will be. Clearly, this will impact on the time required by the processing.
        self.scale_factor = round(self.dimensions[0] / self.mask.shape[1])

    
    def __len__(self):
        return self.number_of_tiles
    
    def __getitem__(self, idx):

        # Get row and col from the index
        row = self.rows[math.floor(idx/len(self.cols))]
        col = self.cols[idx % len(self.cols)]

        downscaled_row = row // self.scale_factor
        downscaled_col = col // self.scale_factor
        # Get the corresponding patch in the segmentation mask
        tile_mask = self.mask[downscaled_row: downscaled_row + self.tile_size // self.scale_factor, downscaled_col: downscaled_col + self.tile_size // self.scale_factor]
        # Percentage of white pixels in the patch = percentage of tissue
        tissue_percentage = (tile_mask == 255).sum() / (tile_mask.shape[0]*tile_mask.shape[1])

        return (col, row), tissue_percentage


class Tiles_Bag(Dataset):

    def __init__(self, slide, tile_size, h5):
        
        # slide : openslide WSI
        # tile_size : dimension of each squared patch (e.g. 128 or 256 or 512)
        # transform : transform function for the tiles classification model

        self.slide = slide
        self.tile_size = tile_size
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.coords = get_coords_h5(h5)
        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        self.normalizer.HERef = torch.tensor([[0.5542, 0.1667], [0.7380, 0.8497], [0.3757, 0.4837]])
        self.normalizer.maxCRef = torch.tensor([2.8576, 1.9241])
        self.T = transforms.ToTensor()

    
    def __len__(self):
        return self.coords.shape[0]
    
    def __getitem__(self, idx):

        col, row = self.coords[idx]
        img = self.slide.read_region((col, row), 0, (self.tile_size, self.tile_size)).convert('RGB')
        try:
            norm, H, E = self.normalizer.normalize(I=self.T(img)*255, stains=True)
            img = Image.fromarray(norm.numpy().astype(np.uint8))
            tile = self.transform(img)
        except:     
            tile = self.transform(img)
        return tile, (col, row)
