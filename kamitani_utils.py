from glob import glob
import itertools
from loguru import logger
import nibabel as nib
import nilearn
from nilearn.image import load_img,get_data
import numpy as np
import os
from os.path import join as opj
import pickle
import re
import scipy
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform
import sklearn
from sklearn.model_selection import PredefinedSplit
from sklearn.linear_model import LinearRegression


DESKIAN_ATLAS = ""
DESTRIEUX_ATLAS = ""
TRAIN_BETAS_DIR = ""
TEST_BETAS_DIR = ""
LANGUAGE_MODELS_REPS = ""
VISUAL_MODELS_REPS = ""


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
        beta_file = load_img(fname) 
        orig_data = beta_file.get_data()
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



def _get_vector_size(model_name):

    if model_name == 'clip' or model_name == 'audioclip':
        vectorshape = 1024
    elif model_name == 'tsmresnet50vat':
        vectorshape = 256
    else :
        vectorshape = 2048
    return vectorshape


def get_model_vectors(model_name,use_train=True,use_test=True): 

    '''
    Get the RDMs for the models        
    '''

    #handle language models separately
    if model_name in ['BERT','GPT2','CLIP-L']:
        
        vector_mat =  np.load(f"{LANGUAGE_MODELS_REPS}/{model_name}_200_vecs.npy")
        out_vec = vector_mat[:150,:150] if use_train==True and use_test==False else vector_mat[150:,150:]
        out_vec = vector_mat if use_train==True and use_test==True else out_vec
        return out_vec

    else :
    
        num_images = 150 if use_train == True and use_test == False else 50
        num_images = 200 if use_train == True and use_test == True else num_images

        indx = 0
        vectorlist = []
        vector_mat = np.zeros((num_images,_get_vector_size(model_name)))

        # add train reps
        if use_train == True:
        
            training_ids = ('1518878', '1639765', '1645776', '1664990', '1704323', '1726692', '1768244', '1770393', '1772222', '1784675', '1787835',
                            '1833805', '1855672', '1877134', '1944390', '1963571', '1970164', '2005790', '2054036', '2055803', '2068974', '2084071',
                            '2090827', '2131653', '2226429', '2233338', '2236241', '2317335', '2346627', '2374451', '2391049', '2432983', '2439033',
                            '2445715', '2472293', '2480855', '2481823', '2503517', '2508213', '2692877', '2766534', '2769748', '2799175', '2800213',
                            '2802215', '2808440', '2814860', '2841315', '2843158', '2882647', '2885462', '2943871', '2974003', '2998003', '3038480',
                            '3063599', '3079230', '3085013', '3085219', '3187595', '3209910', '3255030', '3261776', '3335030', '3345487', '3359137',
                            '3394916', '3397947', '3400231', '3425413', '3436182', '3445777', '3467796', '3472535', '3483823', '3494278', '3496296',
                            '3512147', '3541923', '3543603', '3544143', '3602883', '3607659', '3609235', '3612010', '3623556', '3642806', '3646296',
                            '3649909', '3665924', '3721384', '3743279', '3746005', '3760671', '3790512', '3792782', '3793489', '3815615', '3837869',
                            '3886762', '3918737', '3924679', '3950228', '3982430', '4009552', '4044716', '4070727', '4086273', '4090263', '4113406',
                            '4123740', '4146614', '4154565', '4168199', '4180888', '4197391', '4225987', '4233124', '4254680', '4255586', '4272054',
                            '4273569', '4284002', '4313503', '4320973', '4373894', '4376876', '4398044', '4401680', '4409515', '4409806', '4412416',
                            '4419073', '4442312', '4442441', '4477387', '4482393', '4497801', '4555897', '4587559', '4591713', '4612026', '7734017',
                            '7734744', '7756951', '7758680', '11978233', '12582231', '12596148', '13111881')


            train_path = opj(VISUAL_MODELS_REPS,f"{model_name}_representations_fmri_kamitani_train_avgpool.p")
            with open(train_path,'rb') as ff:
                rep_dict = pickle.load(ff)

        
            filelist = []
            for _id in training_ids:
                category_reps = []
                for key in rep_dict.keys():
                    if _id in key:
                        category_reps.append(rep_dict[key].flatten())
                assert len(category_reps) == 8

                category_reps = np.array(category_reps).mean(0)
                if indx == 0:
                    logger.debug(f'Shape of avg representation for id {_id} is : {category_reps.shape}')

                vectorlist.append(category_reps)
                filelist.append(_id)
                vector_mat[indx,:] = category_reps
                indx += 1

        # add test reps
        if use_test == True:
            folderpath = f'{TEST_BETAS_DIR}/sub-01'  # just to get the names and order
            fnames = [f'beta_00{valid_num:02d}.nii' for valid_num in range(3,53)] #valid_num = 3...52 

            rep_file = opj(VISUAL_MODELS_REPS,f"{model_name}_representations_fmri_test_avgpool.p")
            with open(rep_file,'rb') as ff: 
                rep_dict = pickle.load(ff)

            filelist = []
            for j in fnames:
                for f in os.listdir(folderpath):
                    if j in f:
                        x = f.split('_')
                        for key in rep_dict.keys():
                            if x[0] in key:
                                filelist.append(key)
                                vectorlist.append(rep_dict[key]) 
                                vector_mat[indx,:] = rep_dict[key].flatten()
                                indx += 1
    return  vector_mat




def get_interesting_regions(use_atlas):

    '''
    Get all the regions that can be interesting based on the atlas. 
    use_atlas can be 'default' or 'destrieux' 
    '''

    if use_atlas == 'default' :
        interesting_regions_mapping = {
            # 1011 : 'ctx-lh-lateraloccipital',
            # 2011 : 'ctx-rh-lateraloccipital',
            # (1011,2011):'lateraloccipital',

            (1007,2007):'fusiform',

            # 11143: 'ctx-lh_Pole_occipital',
            # 12143: 'ctx-rh_Pole_occipital',

            # 11145: 'ctx-lh_S_calcarine',
            # 12145: 'ctx-rh_S_calcarine',

            # 1021 : 'ctx-lh-pericalcarine',
            # 2021 : 'ctx-rh-pericalcarine',
            # (1021,2021):'pericalcarine',

            (1011,2011,1021,2021):'visual_region',

            # 1028 : 'ctx-lh-superiorfrontal',
            # 2028 : 'ctx-rh-superiorfrontal',
            
            # 17 : 'left-hippocampus',
            # 53 : 'right-hippocampus',
            (17,53):'hippocampus',

            # 1006 : 'ctx-lh-entorhinal',
            # 2006 : 'ctx-rh-entorhinal',
            # (1006,2006):'entorhinal',

            # 18 : 'Left amygdala',
            # 54 : 'Right amygdala',
            # (18,54):'amygdala',

            # 1016 : 'ctx-lh-parahippocampal',
            # 2016 : 'ctx-rh-parahippocampal',
            (1016,2016):'parahippocampal',

            # (17,53,1006,2006,1016,2016):'MTL_combined_without_amygdala',
            # (17,53,1006,2006,18,54,1016,2016):'MTL_combined',

            # 1119 : 'ctx-lh-G_occipit-temp_med-Parahippocampal_part',      # doesn't exist
            # 2119 : 'ctx-rh-G_occipit-temp_med-Parahippocampal_part',      # doesn't exist

            # 11123: 'ctx_lh_G_oc-temp_med-Parahip',
            # 12123: 'ctx_rh_G_oc-temp_med-Parahip',
        }


    elif use_atlas == 'destrieux':


        interesting_regions_mapping = {

            # 1011 : 'ctx-lh-lateraloccipital',
            # 2011 : 'ctx-rh-lateraloccipital',

            # 11143: 'ctx-lh_Pole_occipital',
            # 12143: 'ctx-rh_Pole_occipital',
            # (11143,12143):'Pole_occipital',

            # 11145: 'ctx-lh_S_calcarine',
            # 12145: 'ctx-rh_S_calcarine',
            # (11145,12145):'S_calcarine',

            (11143,12143,11145,12145):'visual_region',

            # 1021 : 'ctx-lh-pericalcarine',
            # 2021 : 'ctx-rh-pericalcarine',

            # 1028 : 'ctx-lh-superiorfrontal',
            # 2028 : 'ctx-rh-superiorfrontal',
            
            # 17 : 'left-hippocampus',
            # 53 : 'right-hippocampus',
            (17,53):'hippocampus',

            # 1006 : 'ctx-lh-entorhinal',
            # 2006 : 'ctx-rh-entorhinal',

            # 18 : 'Left amygdala',
            # 54 : 'Right amygdala',
            # (18,54):'amygdala',

            # 1016 : 'ctx-lh-parahippocampal',
            # 2016 : 'ctx-rh-parahippocampal',

            # 1119 : 'ctx-lh-G_occipit-temp_med-Parahippocampal_part',      # doesn't exist
            # 2119 : 'ctx-rh-G_occipit-temp_med-Parahippocampal_part',      # doesn't exist

            # 11123: 'ctx_lh_G_oc-temp_med-Parahip',
            # 12123: 'ctx_rh_G_oc-temp_med-Parahip',
            (11123,12123):'ctx_G_oc-temp_med-Parahip',

            (17,53,18,54,11123,12123):'MTL_combined',
        }
    else : 
        raise ValueError("Wrong name for the atlas. Supported values are 'default' or 'destrieux'")
    return interesting_regions_mapping


