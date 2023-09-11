# Self-supervised learning for characterising histomorphological diversity and spatial RNA expression prediction across 23 human tissue types
_Francesco Cisternino, Sara Ometto, Soumick Chatterjee, Edoardo Giacopuzzi, Adam P. Levine, Craig A. Glastonbury_

[Check out the biorxiv preprint](https://www.biorxiv.org/content/10.1101/2023.08.22.554251v1).

# WSI Preprocessing
## 1. Segmentation
Segmentation allows to separate the tissue from background in WSIs. The output are binary masks (stored as .jpeg).
```
python preprocessing/segmentation_patching/segmentation.py
```
* Parameters configuration in preprocessing/segmentation_patching/config.yaml

![image](https://github.com/GlastonburyC/RNAPath/assets/115783390/aff88069-de35-4fcd-99a6-515985272cae)



## 2. Tiling
The tissue region of WSI, identified by segmentation, is divided into small squared tiles (or patches) (e.g. 128x128); this allows both to process the WSI through GPU and to obtain local (tile-level) results.
```
cd ./preprocessing/segmentation_patching
python tiling.py
```
* Parameters configuration in preprocessing/segmentation_patching/config.yaml
* The weights of the ViT-S trained on 1.7M tiles from 23 GTEx tissues is available at (link to big files folder).
* The output of patching for each slide is a .h5 file containing a 2D array of shape (K, 2) - where K is the number of tiles - with the upper left corner coordinates of each tile.

<img width="907" alt="image" src="imgs/263020413-3d2d3dfc-57b5-4e3f-9dd5-524773386d23.png">


## 3. Features extraction

Tile images are turned into features vectors capturing their morphological content. To do this, we use a vision transformer (ViT-S) trained on 1.7 M histology patches using a self-supervised approach.
```
cd ./preprocessing/features_extraction
python extract_features.py
```
* Paramters configuration in preprocessing/features_extraction/config.yaml
* The output of features extraction for each slide is a .pt file containing a 2D tensor of shape (K, 384), where K is the number of tiles and 384 the number of features.
* During features extraction, white patches that could have been included in the tissue mask are filtered out; tipically this happens if there are very small holes in the tissue sample. For this reason, we also store a .h5 file for each slide containing both the features and the final list of coordinates.

# RNAPath

<img width="477" alt="image" src="https://github.com/GlastonburyC/RNAPath/assets/115783390/2c49a2d2-df71-44af-9cb4-15f7bb6d4c8a">


## 1. Training

RNAPath training requires patch features to represent WSIs, train/validation/test splits, a txt indicating the list of genes to be profiled (example in ./resources/gene_set_example.txt) and a csv file with the genes TPMs (link to big files folder).
In this study, 4 different partially overlapping representations of the same slide are computed: to the original patch set, 3 other sets have been added by shifting the original one of 32x32, 64x64 and 96x96 pixels. During each training interation, a single representation (patch set) is randomly selected.
The training script requires some arguments to be set:
* *--exp_code* : experiment code to identify logs and results of the actual run
* *--tissue_code*: alphanumeric code to indentify the tissue of interest
* *--data_root_dir*: main directory of patch features
* *--split_dir*: directory of splits (if not specified, it will be ./splits/RNAPath_{tissue_code}) (default: None)
* *--results_dir*: directory where results will be stored (default: './results')
* *--max_epochs*: maximum number of epochs (default: 200)
* *--lr*: starting learning rate (default: 1e-4)
* *--lr_scheduler*: learning rate scheduler; choices are 'constant' and 'plateau'; in the latter case, the lr drops after 10 epochs without validation loss improvement (default: 'plateau')
* *--label_frac*: fraction of training labels (default: 1.0)
* *--reg*: weight decay (default: 1e-3)
* *--seed*: random seed for reproducible experiment (default: 1)
* *--log_data*: log data using tensorboiard (default: False)
* *--early_stopping*: whether to enable early stopping; training stops after 20 epochs without validation loss improvements (default: True)
* *--bag_droput*: whether to apply droput to bag instances (default: True)
* *--opt*: optimizer; possible choices are 'adam' and 'sgd' (default: 'adam')

```
python train.py --exp_code test_0 --tissue_code HEA --data_root_dir /path/to/features/dir/
```

During training, training and validation loss values will be logged and a results folder will be created (inside results_dir) and named as the experiment code; in this folder, the gene-level r-scores for both validation and test set and the weights checkpoint file will be stored.

## 2. Inference and visualization

### 2.1 Inference

At inference, trained models are used to infer patch-level expression. Patch logits are stored as .pt files and can be used to plot heatmaps of the genes of interest.
The inference scripts requires the following arguments:

* *--tissue_name*: name of the tissue (e.g. Heart, Colon, Skin, EsophagusMucosa for GTEx)
* *--tissue_code*: alphanumeric code to indentify the tissue of interest
* *--features_dir*: main directory of patch features
* *--output_dir*: directory where patch level expression values (patch logits) will be stored
* *--results_dir*: trainig results directory
* *--ckpt_path*: path to RNAPath model checkpoint
* *--multiple_patch_sets*: if multiple partially overlapping patch sets are used for the same slide (default: False)
```
python inference.py --tissue_name Heart --tissue_code HEA --features_dir /path/to/features/dir/ --output_dir /path/to/patch_logits/dir/ --results_dir /path/to/results/dir/ --ckpt_path /path/to/rnapath/checkpoint.pt --multiple_patch_sets
```

### 2.2 Visualization

The predicted localization of gene activity can be visually represented by plotting patch logits over the histology sample. Heatmaps are stored in the jpeg format.
The following arguments are needed:

* *--gene_name*: gene name (description)
* *--slide_id*: full ID of the histology sample
* *--tissue_name*: name of the tissue (e.g. Heart, Colon, Skin, EsophagusMucosa for GTEx)
* *--tissue_code*: alphanumeric code to indentify the tissue of interest
* *--save_dir*: directory where heatmaps images will be stored
* *--features_dir*: main directory of patch features
* *--patch_logits_dir*: directory where patch logits are stored
* *--results_dir*: trainig results directory
* *--slides_dir*: raw whole slide images directory
* *--multiple_patch_sets*: if multiple partially overlapping patch sets are used for the same slide (default: False)

```
python heatmaps.py --gene_name CD19 --slide_id SLIDE_ID --tissue_name EsophagusMucosa --tissue_code EMUC --save_dir /path/to/save/dir --features_dir /path/to/features/dir --patch_logits_dir /path/to/patch_logits/dir/ --results_dir /path/to/results/dir/ --slides_dir /path/to/wsi/dir/ --multiple_patch_sets
```
<center>
<img width="706" alt="image" src="https://github.com/GlastonburyC/RNAPath/assets/115783390/a92aa060-da76-4a8a-b807-767f038cfcf8">
</center>


# Tissue multiclass segmentation by tiles clustering

The following scripts foresee this file organization for the WSI; please, apply changes to the code in case of different structure.
```
SLIDES_DIRECTORY/
    ├── Tissue1
    │   ├── slide_1.svs
    │   ├── slide_2.svs
    │   └── ...
    ├── Tissue2
    │   ├── slide_3.svs
    │   ├── slide_4.svs
    │   └── ...
```

To segment tissues into substructures or localised pathologies by patch-level classification using a k-Nearest Neighbors model, two steps are required:
1. Definition of instances and labels for the k-NN; instances are patch-level features, while classes are defined into a yaml file. The instances used to fit the k-NN have been hand-labelled. The following script creates a h5 file containing patch features and their corresponding labels; this file will be then used to fit the k-NN model for tiles classification.
```
python tiles_classification/define_clusters_kNN.py --tissue_name Heart --checkpoint_path /path/to/features_extraction/checkpoint.pth
```
2. Multi-class segmentation of H&E tissue samples by tiles classification. The script loads the previously defined h5 file, fits a k-NN model using the features and labels stored in the .h5 and classify all the patches of the WSIs. Segmentation masks and csv files containing the class of each patch (identified by the upper left corner coordinates) are output.
As arguments, the tissue name, the output directory, the patch features main directory and the slides directory are required.
```
cd ./tiles_classification
python multiclass_tissue_segmentation.py --tissue_name Heart --output_dir /path/to/output/dir/ --features_dir /path/to/features/dir/ --slides_dir /path/to/slides/dir/
```
A script for fine-grained segmentation is also provided; in this case, the 4 partially overlapping patch sets are used for the same slide and, in the regions where multiple patch sets overlap, classes are assigned by majority voting. 

```
cd ./tiles_classification
python fine_grained_multiclass_tissue_segmentation.py --tissue_name Heart --output_dir /path/to/output/dir/ --features_dir /path/to/features/dir/ --slide_name /slide/name
```

<img width="510" alt="image" src="https://github.com/GlastonburyC/RNAPath/assets/115783390/5ea6a74e-2888-4922-984b-6db43980da07" align="center">

# Image derived phenotypes

Image phenotypes (e.g. amount of mucosa in colon samples, aumont of adipocytes, etc.) are derived using the patch classes output by the multiclass tissue segmentation script; indeed, these phenotypes reflect the relative amount of each target class in a sample.
The following script can be used to compute such phenotypes as proportions (with respect to the sample size). This will make the compositional phenotypes comparable across samples.
```
cd ./image_derived_phenotypes
python compute_IDPs.py --tissue_name EsophagusMucosa --segmentation_dir /path/to/segmentation/dir/ --output_dir /path/to/idps/dir/
```
The script outputs a csv file for each tissue, in the following format:


| Slide ID | IDP_0 | IDP_1 | IDP_2 | IDP_3 | IDP_4 | IDP_5 | IDP_6 |
|----------|----------|----------|----------|----------|----------|----------|----------|
| SLIDE_001|20.9%|25.1%|44.3%|1.0%|2.5%|1.2%|4.9%|
| SLIDE_002|39.6%|23.1%|33.0%|1.5%|1.2%|2.8%|3.5%|

Compositional phenotypes are easy to interpretate, but they are not the proper choice for statistical analysis, given the closure problem. For this reason, they can be turned into pivot coordinates (a special case of isometric logratio coordinates):

```
cd ./image_derived_phenotypes
python compute_pivot_coordinates.py --tissue_name EsophagusMucosa --idps_dir /path/to/idps/dir/
```

# SSES - Substructure-Specific Enrichment Analysis

SSES combines results from RNAPath (local RNASeq prediction) with tiles classification into tissue substructures or localised pathologies. The output of this analyis is a set of enrichment scores, one per each couple (gene, substructure) indicating the ratio between the predicted expression inside the substructure/pathology and the bulk (sample-level) predicted expression; the bigger this value, the more the expression of the gene will be focused in that substructure/pathology.
The following arguments are required


```
python enrichment.py --tissue_name EsophagusMucosa --tissue_code EMUC --patch_logits_dir /path/to/patch/logits --segmentation_dir /path/to/segmentation/results/ --features_dir /path/to/features
```
* *--tissue_name*
* *--tissue_code*
* *--patch_logits_dir*: directory where patch level expression values (logits) have been stored
* *--segmentation:dir*: directory containing segmentation results
* *--features_dir*: features_directory

The script outputs a csv file with as many row as the number of genes and as many columns as the number of identified classes (substructures/pathologies).
  

# Differential expression analysis

Differential expression analysis of image derived phenotypes has been performed by linear models fitting. To run the analysis, some parameters are required:

* *--tissue_name*
* *--gtex_expression_bed_file*: bed file with normalized expression levels for the selected tissue
* *--gtex_subject_phenotypes_file*: file containing subject phenotypes
* *--gtex_covariates_file*: covariates file (provided by GTEx)
* *--idps_format*: phenotypes format; possible choices are 'binary' for binary phenotypes (e.g. presence/absence of calcification in arteries), 'compositional' and 'pivot', whose difference has been described in the previous section.

```
cd ./differential_expression_analysis
python differential_expression_analysis_IDPs.py --tissue_name EsophagusMucosa --gtex_normalized_expression_bed_file /path/to/gtex/expression/bed --gtex_subject_phenotypes_file /path/to/subject/phenotypes --gtex_covariates_file /path/to/gtex/covariates --idps_format pivot
```


  
# GWAS

The genome-wide association analysis was conducted using [nf-pipeline-regenie](https://github.com/HTGenomeAnalysisUnit/nf-pipeline-regenie) (v1.8.1). A config file defining the computations environment and a config file for the project (example in ./gwas/gtex.conf) are needed to launch the pipeline. A complete description on how to use it and template files can be found in the linked repository.

Regional plots, Manhattan plots and quantile-quantile plots were generated with [GWASLab](https://github.com/Cloufield/gwaslab) (v3.4.21).

<img width="907" alt="image" src="imgs/20230908-122647.png">


# Interaction eQTLs

Interaction eQTLs have been analyzed using [tensorqtl](https://github.com/broadinstitute/tensorqtl), a gpu-enabled QTL mapper; the required input are the genoptypes, the gene expression, the interaction term (in this case, the image derived phenotypes, one per donor) and the coviariates.

<img width="661" alt="image" src="https://github.com/GlastonburyC/RNAPath/assets/115783390/314c11f0-3b19-4f88-af47-10022f950967">

# Supplementary Material and Data

| Data | Link |
|----------|----------|
| Features Extraction|DINO checkpoint||
| RNAPath|Results and checkpoints||
| Image Derived Phenotypes|Derived substructures||
| Image Derived Phenotypes|Differential expression analysis summary stats||
| Image Derived Phenotypes|Differential expression analysis - Genes enrichment||
| Image Derived Phenotypes|GWAS summary stats||
| Image Derived Phenotypes|ieQTLs summary stats||


_[WIP - We are still transfering code from our internal gitlab to this github repo but we wanted to make this public asap.]_
