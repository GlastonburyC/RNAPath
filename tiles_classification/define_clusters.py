import glob
from PIL import Image
from wsi_preprocessing.model import FeaturesExtraction_Vit, FeaturesExtraction_IMAGENET
import random
import argparse
from wsi_preprocessing.preprocessing_utils import save_hdf5
import torch
import numpy as np
import tqdm
import yaml
import torchvision.transforms as transforms
import torchstain

random.seed(24)

DEVICE =  torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--tissue_name", type=str, default=None)
parser.add_argument("--checkpoint_path", type=str, default=None)
args = parser.parse_args()


# Load classes and colors from yaml file
f = open("clusters.yaml", "r")
doc = yaml.load(f, Loader=yaml.FullLoader)
classes = doc[args.tissue_name]['classes']
colors = doc[args.tissue_name]['colors']
print(f'classes: {classes}')


images_clusters = []

h5_file_path = './' + args.tissue_name + '_clusters.h5'

# Load images from each class into a list
for c in classes:    
    images = glob.glob(f'/group/glastonbury/gtex-proto-class-labels/{args.folder_name}/**/' + c + '_128/*.jpg')
    if len(images) > 1000:
        random.shuffle(images)
        images_clusters.append(images[:1000])
    else:
        images_clusters.append(images)


l = sum([len(images_clusters[i]) for i in range(len(images_clusters))])
print(f'Total number of images: {l}')

for c in classes:
    print(f'{c}: {len(images_clusters[classes.index(c)])} images')

# Load features extraction model
model = FeaturesExtraction_Vit(arch='vit_small', pretrained_weights="/home/f.cisternino/WSIproj/dino/results/ALL/checkpoint.pth")
model.model.eval()
model.model.to(DEVICE)

# Transform function
transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()])
T = transforms.ToTensor()

# Stain normalizer
normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
normalizer.HERef = torch.tensor([[0.5542, 0.1667], [0.7380, 0.8497], [0.3757, 0.4837]])
normalizer.maxCRef = torch.tensor([2.8576, 1.9241])

mode = 'w'

# Extract features from each patch and create a h5 file with patch features and class;
# this file will be used to classify tiles by k-Nearest Neighbors

for idx, c in enumerate(classes):
    print(idx, c)
    for tilename in tqdm.tqdm(images_clusters[idx]):
        # Open tile image        
        img = Image.open(tilename).convert('RGB')

        # apply stain normalization and transform
        try:
            norm, H, E = normalizer.normalize(I=T(img)*255, stains=True)
            img = Image.fromarray(norm.numpy().astype(np.uint8))
            tile = transform(img)
        except:
            tile = transform(img)

        # extract features
        features = model.extractFeatures(tile.unsqueeze(0), device = DEVICE)    
        asset_dict = {'features': features.reshape(1, -1), 'cluster':  np.array([idx])}
        save_hdf5(h5_file_path, asset_dict, attr_dict= None, mode=mode)
           
        mode = 'a'

print(f'Clusters creation for {args.tissue_name} completed!')