# Self-supervised learning for characterising histomorphological diversity and spatial RNA expression prediction across 23 human tissue types
Francesco Cisternino, Sara Ometto, Soumick Chatterjee, Edoardo Giacopuzzi, Adam P. Levine, Craig A. Glastonbury


# WSI Preprocessing
1. Segmentation
Segmentation allows to separate the tissue from background in WSIs. The output are binary masks (stored as .jpeg).
```
python preprocessing/segmentation/segmentation.py
```
⋅⋅* Paramters configuration in preprocessing/segmentation/config.yaml

![Supplementary01](https://github.com/GlastonburyC/RNAPath/assets/115783390/e8effb2a-3f4a-44c6-9f2a-44ec05d709c2)
