# Self-supervised learning for characterising histomorphological diversity and spatial RNA expression prediction across 23 human tissue types
Francesco Cisternino, Sara Ometto, Soumick Chatterjee, Edoardo Giacopuzzi, Adam P. Levine, Craig A. Glastonbury


# WSI Preprocessing
## 1. Segmentation
Segmentation allows to separate the tissue from background in WSIs. The output are binary masks (stored as .jpeg).
```
python preprocessing/segmentation_patching/segmentation.py
```
* Paramters configuration in preprocessing/segmentation_patching/config.yaml

![Supplementary01](https://github.com/GlastonburyC/RNAPath/assets/115783390/e8effb2a-3f4a-44c6-9f2a-44ec05d709c2)


## 2. Tiling
We divide the tissue region of the WSI into small squared tiles (e.g. 128x128); this allows both to process the WSI through GPU and to obtain local (tile-level) results.
```
python preprocessing/segmentation_patching/tiling.py
```
* Paramters configuration in preprocessing/segmentation_patching/config.yaml
* The output of patching is a .h5 file containing a 2D array of shape (K, 2) - where K is the number of tiles - with the upper left corner coordinates of each tile.

<img width="907" alt="image" src="https://github.com/GlastonburyC/RNAPath/assets/115783390/3d2d3dfc-57b5-4e3f-9dd5-524773386d23">

