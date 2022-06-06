

import re
import os
from os.path import join as opj

import pickle
import numpy as np
import nibabel as nib
from glob import glob
from loguru import logger



DESKIAN_ATLAS = ""
DESTRIEUX_ATLAS = ""
TRAIN_BETAS_DIR = ""
LANGUAGE_MODELS_REPS = ""
VISUAL_MODELS_REPS = "/mnt/HD2/bhavin/all_layerwise_reps_new"

TRAIN_BETAS_DIR = ""
TEST_BETAS_DIR = f"/media/bhavin/My Passport2/BigGANProject/datasink/glm_test_mni305/sub-01"

def _get_mask(region_idx,atlas):

    # get the atlas
    if atlas == 'destrieux':
        aparc_file = nib.load(DESKIAN_ATLAS)
    else:
        aparc_file = nib.load(DESKIAN_ATLAS)
    aparc_data = aparc_file.get_fdata()

    # get the mask
    if region_idx is not None:

        if isinstance(region_idx,int):
            region_mask = aparc_data == region_idx
        elif isinstance(region_idx,tuple):
            region_mask = np.isin(aparc_data,region_idx)

    else:
        region_mask = aparc_data != np.nan    #basically whole brain

    return region_mask



def get_region_vectors(sub_idx,region_idx,atlas='default',use_train=True,use_test=True):

    '''
    Function to get beta values in a specific regions of the brain.

    Args : 
        sub_idx     : index of the subject (from [1,2,3,4,5])
        region_idx  : (int/tuple/None) index of the region as per the atlas used. None would give the values in the whole brain.
        atlas       : atlas used to define the region_idx. Can be 'default' or 'destrieux'
        use_train   : include train betas
        use_test    : include test betas

        NOTE : if both use_train and use_test are given to be True, the first 150 will we train and the next 50 will be test. This order has to be preserved for all the subsequent analysis.
    '''

    region_mask = _get_mask(region_idx,atlas)

    train_beta_dir = opj(TRAIN_BETAS_DIR,f"sub-0{sub_idx}")
    test_beta_dir = opj(TEST_BETAS_DIR,f"sub-0{sub_idx}")

    # get all names for beta files
    beta_fnames = []
    if use_train==True:
        beta_fnames = sorted([f for f in os.listdir(train_beta_dir) if f[:4]=='beta'])
        beta_fnames = [opj(train_beta_dir,f) for f in beta_fnames[2:152]]

    if use_test==True:
        test_beta_fnames = sorted([f for f in glob(opj(test_beta_dir,'*.nii')) if re.search(r'\d+_\d+',f)])
        assert len(test_beta_fnames)==50

        for x in test_beta_fnames:
            beta_fnames.append(x)


    # get the betas
    beta_data = []
    for fname in beta_fnames:
        beta_file = nib.load(fname) 
        orig_data = beta_file.get_fdata()
        assert beta_file.shape == region_mask.shape

        region_data = np.nan*np.ones(orig_data.shape,dtype=orig_data.dtype)
        region_data[region_mask] = orig_data[region_mask]

        beta_data.append(region_data)

    if use_train == True and use_test == True  : assert len(beta_data) == 200
    if use_train == True and use_test == False : assert len(beta_data) == 150 
    if use_train == False and use_test == True : assert len(beta_data) == 50 

    # reshape and remove nans
    betas = np.stack(beta_data,axis=0)
    n,h,w,c = betas.shape
    sub_data = betas.reshape(n,h*w*c) 

    nans = np.isnan(sub_data[0])
    sub_data = sub_data[:,np.logical_not(nans)]
    logger.debug(f'Shape of data for region {region_idx} is {sub_data.shape}')

    return sub_data if sub_data.shape[1] != 0 else None



def get_model_vectors(model_name,layer_to_use,use_test=False):

    '''
    Function to get new_model_vectors. 
    The function expects a dictionary of the form dict[image_id] = { layer_name : layer_feature_obtained } for each model

    Args : 
        model_name : name of the model
        layer_to_use : layer in the dict whose features are required
        use_test : use features of test images    
    '''


    #handle language models separately
    if model_name in ['BERT','GPT2','CLIP-L']:
        vector_mat =  np.load(f"{LANGUAGE_MODELS_REPS}/{model_name}_200_vecs.npy")
        return vector_mat[150:,150:]


    if use_test == True:

        valid_beta_fnames = [f'beta_00{valid_num:02d}.nii' for valid_num in range(3,53)] #valid_num = 3...52 

        vectors_path = opj(VISUAL_MODELS_REPS,f"all_{model_name}_representations_fmri_test.p")
        with open(vectors_path,'rb') as f:
            representation_dict = pickle.load(f)

        representation_dict = {k_category:v[layer_to_use] for k_category,v in representation_dict.items()}
 
        vectorlist = []
        for valid_betaname in valid_beta_fnames:
            for f in os.listdir(TEST_BETAS_DIR):   # use beta names to ensure the ordering of the features for the RDM
                if valid_betaname in f:
                    x = f.split('_')
                    for key,vector in representation_dict.items():
                        if x[0] in key:
                            vectorlist.append(vector.flatten())


        return np.row_stack(vectorlist)



def get_interesting_regions(use_atlas):

    '''
    Get all the regions that can be interesting based on the atlas. 
    use_atlas can be 'default' or 'destrieux' 
    '''
    
    
    
    if use_atlas == 'default' :
        interesting_regions_mapping = {

            (1007) : 'left_fusiform',
            (2007) : 'right_fusiform',
            (1007,2007):'fusiform',


            (1011,1021) : 'left visual region',
            (2011,2021) : 'right visual region',
            (1011,2011,1021,2021):'visual_region',


            (17):'left hippocampus',
            (53):'right hippocampus',
            (17,53):'hippocampus',


            (1016):'left parahippocampal',
            (2016):'right parahippocampal',
            (1016,2016):'parahippocampal',

        }

    else : 
        raise ValueError("Wrong name for the atlas. Supported values are 'default'")
    return interesting_regions_mapping

