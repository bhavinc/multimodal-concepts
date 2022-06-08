# mutlimodal-concepts

Code to reproduce the results presented in the SVRHM Workshop (NeurIPS 2021) paper ['Multimodal neural networks better explain multivoxel patterns in the hippocampus'](https://openreview.net/forum?id=6dymbuga7nL).


## Introduction

<!-- <p align='center'><img src="./images/Flow_conceptcellproject.png" width="600")></p> -->



## Reproducing the results
To reproduce the results shown in this article, please download the preprocessed fMRI dataset from [figshare](link).
The raw fMRI dataset can be directly downlaoded from [KamitaniLab](https://github.com/KamitaniLab/GenericObjectDecoding)'s github page.

### Setup 

The requirements/dependencies are mentioned in `requirements.txt`.

`get_model_features.py` To get all the representations of all the models. It will generate a dict with all the representations. 

`kamitani_utils.py` demonstrates how to use the beta files (obtained after GLM) and the latent representations of the images (obtained from models). You can adapt this file as per your requirements. The beta files can be found on the figshare [here](link)

`main_analysis.py` can be used to replicate/reproduce all the results shown in the paper. The file creates 3 plots -- a normalized version, a non-normalized version, and a modality-specific version. It takes configuration from the `config.yaml` file. Please update that file as desired before running `main_analysis.py`. 

`config.yaml` shows all the configuration for a particular run. Please edit this file depending on the layers you wish to look at. 









