#%%
####################################################
#    Imports
####################################################

from loguru import logger

import os
from os.path import join as opj

import numpy as np
import matplotlib.pyplot as plt

plt.style.use("default")

import scipy
from scipy.spatial import distance
from scipy.stats import spearmanr, pearsonr , ttest_ind 
from scipy.spatial.distance import pdist, squareform


from general_utils import *
from kamitani_utils import *

####################################################
#                    Global Args
####################################################

class Args:
    def __init__(self):

        # data to use
        self.USE_ATLAS = "default"
        self.USE_TRAIN_BETAS = False
        self.USE_TEST_BETAS = True

        # creating the rdms
        self.DISTANCE = "correlation"  # distance used to get the rdms
        self.ZERO_CENTER_BRAIN_RDMS = False

        # between rdms
        self.CORR_FUNC = pearsonr
        self.STATISTICAL_TEST = ttest_ind

        self.MODELS = [
            "clip",
            "BiT",
            "virtex",
            "virtexv2",
            "icmlm_attfc",
            "icmlm_tfm",
            "tsmresnet50",
            "audioclip",
            "resnet",
            "madry",
            "madryli4",
            "madryl23",
            "geirhos",
            "geirhos_sinin",
            "geirhos_sininfin",
            "BERT",
            "GPT2",
            "CLIP-L",
        ]

    @property
    def SAVE_DIR(self):
        return f"./plots_for_paper/{'test' if self.USE_TEST_BETAS else 'train'}_atlas_{self.USE_ATLAS}_{'zerocentered' if args.ZERO_CENTER_BRAIN_RDMS else ''}_distance_{self.DISTANCE}_corrfunc_{self.CORR_FUNC.__name__}"

    @property
    def print_params(self):
        for x in vars(self):
            print("{:<20}: {}".format(x, getattr(args, x)))


args = Args()
args.print_params

POSSIBLE_TASKS = [
    "normal_correlations",
    "select_top30_voxels",
    "select_voxels_based_on_hippocampus",
    "select_from_train_for_test",
]
MAIN_TASK = "select_top30_voxels"

assert MAIN_TASK in POSSIBLE_TASKS


#####################################################
#                Helper functions
#####################################################

def _get_sort_idx(mat):

    # we want to find voxels that respond maximally to the images
    # hence, let's first find voxels that have highest betas
    var_list = [mat[:, vox].max() for vox in range(mat.shape[1])]
    sort_idx = np.argsort(var_list)

    return sort_idx


def select_topk_voxels(sampledata, num_voxels, sort_idx=None):
    """
    Select the top n voxels based on the sort idx provided
    if sort_idx is None, then the max values will be calculated using max values 
    Remember that we are technically changing the order of the voxels. There will be effects because of this. 
    """
    if num_voxels == None:
        return sampledata

    sort_idx = _get_sort_idx(sampledata) if sort_idx is None else sort_idx

    subsample = []
    for c in sort_idx[::-1]:
        if len(subsample) == num_voxels:
            # print(c, "--", sampledata[:, c].max())
            break

        if np.all(sampledata[:, c] != 0.0):
            subsample.append(sampledata[:, c])

    subsample = np.stack(subsample, axis=1)

    return subsample




#get regions, subjects and their data
REGIONS = get_interesting_regions(args.USE_ATLAS)  
SUBJECTS = range(1,6)

data_dict = {_region : {sub_idx : get_region_vectors(sub_idx,_region,atlas=args.USE_ATLAS,use_train=args.USE_TRAIN_BETAS,use_test=args.USE_TEST_BETAS)  for sub_idx in SUBJECTS} for _region in REGIONS}



#####################################################
#            Task-specific processing
#####################################################
if MAIN_TASK == 'normal_correlations':
    smaller_data_dict = data_dict


if MAIN_TASK == 'select_top30_voxels':

    sort_idx_dict = {_region:{sub_idx:_get_sort_idx(data_dict[_region][sub_idx]) for sub_idx in SUBJECTS} for _region in REGIONS}
    num_voxel_dict = { r : 30 for r in REGIONS }
    smaller_data_dict = {region : {sub: select_topk_voxels(data_dict[region][sub],num_voxel_dict[region],sort_idx=sort_idx_dict[region][sub]) for sub in SUBJECTS } for region in REGIONS}

if MAIN_TASK == 'select_from_train_for_test':

    test_data_dict = {_region : {sub_idx : get_region_vectors(sub_idx,_region,atlas=args.USE_ATLAS,use_train=True,use_test=False)  for sub_idx in SUBJECTS} for _region in REGIONS}
    sort_idx_dict = {_region:{sub_idx:_get_sort_idx(test_data_dict[_region][sub_idx]) for sub_idx in SUBJECTS} for _region in REGIONS}

    # NOTE that the above is using train data to build the sort_idx_dict
    smaller_data_dict = {region : {sub: select_topk_voxels(data_dict[region][sub],num_voxel_dict[region],sort_idx=sort_idx_dict[region][sub]) for sub in SUBJECTS } for region in REGIONS}


if MAIN_TASK == 'select_voxels_based_on_hippocampus':

    def get_threshold_dict(num_voxels_of_hippocampus):

        threshold_dict = {s:0. for s in SUBJECTS}
        for s in SUBJECTS:
            var_list = [data_dict[(17,53)][s][:,vox].max() for vox in range(data_dict[(17,53)][s].shape[1]) ]
            threshold_dict[s] = sorted(var_list)[-(num_voxels_of_hippocampus+1)]
        return threshold_dict

    def select_thresholded_voxels(sampledata,threshold=0.):
        '''
        Select the top voxels based on the threshold. 
        '''
        var_list = [sampledata[:,vox].max() for vox in range(sampledata.shape[1])]
        subsample = []

        for c,max_val in enumerate(var_list):
            if max_val > threshold:
                subsample.append(sampledata[:,c])

        subsample = np.stack(subsample,axis=1)
        logger.debug(f'Using threshold found :{subsample.shape}')
        return subsample

    threshold_dict = get_threshold_dict(num_voxels_of_hippocampus=30)
    smaller_data_dict = {region : {sub : select_thresholded_voxels(smaller_data_dict[region][sub],threshold=threshold_dict[sub]) for sub in SUBJECTS} for region in REGIONS}


# calculate the upper_ceilings
upper_ceiling_dict = {region: calculate_noise_ceilings([ get_rdm(smaller_data_dict[region][sub],distance=args.DISTANCE) for sub in SUBJECTS],corr_func=args.CORR_FUNC) for region in REGIONS}




# GET CORR DATA DICT
corr_data_dict = {}
for _model in args.MODELS:

    corr_data_dict[_model] = {k:{sub:None for sub in SUBJECTS} for k in REGIONS}
    for _region in REGIONS:
        for _sub in SUBJECTS:
            model_rdm = get_rdm(get_model_vectors(_model,use_train=args.USE_TRAIN_BETAS,use_test=args.USE_TEST_BETAS),distance=args.DISTANCE)
            brain_vectors = smaller_data_dict[_region][_sub]
            corr_data_dict[_model][_region][_sub] = args.CORR_FUNC(model_rdm,get_rdm(brain_vectors,distance=args.DISTANCE)) if brain_vectors is not None else np.nan


#####################################################
#                   Plotting
#####################################################

def get_plot_title(reason):

    base_str = f"{'zerocentered ;' if args.ZERO_CENTER_BRAIN_RDMS else ''}  Distance:{args.DISTANCE} ; corrfunc : {args.CORR_FUNC.__name__}"

    if reason == 'noise_ceiling':
        return (f"{'Default' if args.USE_ATLAS=='default' else 'destrieux'} Atlas"
               f"; {'train' if args.USE_TRAIN_BETAS else 'test'} Betas;" 
               f"{base_str}")
    
    if reason == 'select_from_train_for_test':
        return f"{reason} : {base_str} \n selecting top voxels from test data on train data"

    if reason == 'select_voxels_based_on_hippocampus':
        return f"{reason} : {base_str} \n thresholded using max value of the 30th hippocampus voxel"

    if reason == 'normal_correlations' or reason == 'select_top30_voxels':
        return f"{reason} : {base_str}"

      
def _change_flag(mat):
    try:
        get_rdm(mat,distance=args.DISTANCE)
    except ValueError:
        flag = False
        return flag
    else:
        flag = True
        return flag

      
def plot_noise_ceilings(save_name=None,sort_idx_dict=None):

    num_voxels = np.arange(2,2000,1)

    fp = {'fontsize':16}
    plt.figure(figsize=(10,6))

    plt.suptitle(get_plot_title('noise_ceiling'),**fp)

    for counter,region in enumerate(REGIONS):

        ys,errs = [],[]
        rand_ys,rand_errs = [],[]

        for n in num_voxels:

            min_num = min([data_dict[region][x].shape[1] for x in SUBJECTS])
            if n > min_num:
                break

            five_rdms , rand_five_rdms = [],[]
            for subject in SUBJECTS:
                if sort_idx_dict is None:
                    sort_idx = _get_sort_idx(data_dict[region][subject]) 
                else:
                    sort_idx = sort_idx_dict[region][subject]  # because for other tasks we want to use the predefined sort_idx_dict 
                    

                # using only those voxels
                subsample = data_dict[region][subject][:,sort_idx[-n:]] 
                if args.ZERO_CENTER_BRAIN_RDMS == True:
                    subsample = zero_center(subsample)
                main_rdm = squareform(get_rdm(subsample,distance=args.DISTANCE))
                five_rdms.append(squareform(main_rdm))

                
                # while loop here since some voxels are just zero...
                flag = False
                while flag == False:
                    if args.ZERO_CENTER_BRAIN_RDMS == True:
                        subsample = zero_center(subsample)                    
                    rand_subsample = data_dict[region][subject][:,np.random.choice(sort_idx,n)]
                    flag = _change_flag(rand_subsample)
                rand_five_rdms.append(get_rdm(rand_subsample,distance=args.DISTANCE))

                assert subsample.shape == rand_subsample.shape
            
            m,sem = np.array(calculate_noise_ceilings(five_rdms,corr_func=args.CORR_FUNC)).mean() , scipy.stats.sem(np.array(calculate_noise_ceilings(five_rdms,corr_func=args.CORR_FUNC)))
            ys.append(m)
            errs.append(sem)

            rand_m,rand_sem =  np.array(calculate_noise_ceilings(rand_five_rdms,corr_func=args.CORR_FUNC)).mean() , scipy.stats.sem(np.array(calculate_noise_ceilings(rand_five_rdms,corr_func=args.CORR_FUNC)))
            rand_ys.append(rand_m)
            rand_errs.append(rand_sem)

        xs = list(num_voxels)[:len(ys)] 
        
        plt.subplot(2,3,counter+1)
        plt.title(f"{REGIONS[region]}",**fp)    
        plt.xlabel('#Voxels chosen',**fp)
        plt.ylabel('Noise ceilings',**fp) if counter==1 or counter ==4 or counter == 7 else None

        plt.plot(xs,ys,marker='.',label='x voxels',alpha=0.9)
        plt.fill_between(xs, [ys[i]-errs[i] for i in range(len(ys))],[ys[i]+errs[i] for i in range(len(ys))],color='gray',alpha=0.3)


        plt.plot(xs,rand_ys,marker='.',label='random',alpha=0.3)
        plt.fill_between(xs, [rand_ys[i]-rand_errs[i] for i in range(len(rand_ys))],[rand_ys[i]+ rand_errs[i] for i in range(len(rand_ys))],color='gray',alpha=0.3)

    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1,1))

    if save_name is not None:
        plt.savefig(f'{save_name}.pdf',bbox_inches='tight')






############# SOME COLORS #################
cm1 , cnum1 = plt.get_cmap('Blues') , 5.
cm2 , cnum2 = plt.get_cmap('Reds')  , 8.
cm3 , cnum3 = plt.get_cmap('Greens'), 3.

model_plot_dict = {
    'clip'          : ('CLIP',                1.5, cm3(1/cnum1)),
    # 'virtex'        : ('VirTex',              1.5, cm3(2/cnum1)),
    'virtexv2'      : ('VirTex',              1.5, cm3(2/cnum1)),
    'icmlm_attfc'   : ('ICMLM-attfc',         1.5, cm3(3/cnum1)), 
    'icmlm_tfm'     : ('ICMLM-tfm',           1.5, cm3(4/cnum1)), 
    'tsmresnet50'   : ('TSMResNet50-visual',  1.5, cm3(5/cnum1)),
    # 'audioclip'     : ('AudioCLIP',           1.5,cm3(7/cnum1)),


    'BiT'              : ('BiT-M',       3.0, cm2(1/cnum2)),
    'resnet'           : ('ResNet50',    3.0, cm2(2/cnum2)),
    'madry'            : ('AR-Linf8',    3.0, cm2(3/cnum2)),
    'madryli4'         : ('AR-Linf4',    3.0, cm2(4/cnum2)),
    'madryl23'         : ('AR-L2',       3.0, cm2(5/cnum2)),
    'geirhos'          : ('SIN',         3.0, cm2(6/cnum2)),
    'geirhos_sinin'    : ('SIN-IN',      3.0, cm2(7/cnum2)),
    'geirhos_sininfin' : ('SIN-IN+FIN',  3.0, cm2(8/cnum2)),


    'BERT'   : ('BERT',     4.5, cm1(1/cnum3)),
    'GPT2'   : ('GPT2',     4.5, cm1(2/cnum3)),
    'CLIP-L' : ('CLIP-L',   4.5, cm1(3/cnum3)),


    'M'     : ('M', 6.0, cm3(0.25*cnum3)),
    'V'     : ('V', 6.0, cm2(0.1*cnum2)),
    'L'     : ('L', 6.0, cm1(0.33*cnum1)),

}



def plot_mainfig(save_name=None,normalized_version=False):

    fp = {'fontsize':16}
    plt.figure(figsize=(10,9))
    plt.suptitle(get_plot_title(MAIN_TASK),fontsize=10)


    counter = 1
    for ind,region in enumerate(REGIONS):

        upper_ceilings = upper_ceiling_dict[region]
        
        plt.subplot(3,3,counter)

        # if MAIN_TASK == 'normal_correlations' or MAIN_TASK=='select_top30_voxels':
        plt.title(f"{REGIONS[region]} : {smaller_data_dict[region][1].shape[1] } voxels",**fp)
        # else:
            # plt.title(f"{REGIONS[region]} ",**fp)
        
        plt.xticks([])

        if normalized_version == True:
            plt.axhline(1,linestyle='dashed')
            factors = [1.0/x for x in upper_ceilings]
            plt.ylim((-0.05,1.4))
            plt.yticks([0.0,1.0,1.4],[0.0,1.0,1.4],**fp)

        else:
            mnoise,smnoise=np.mean(upper_ceilings),scipy.stats.sem(upper_ceilings)
            plt.fill_between(np.arange(31),mnoise-smnoise,mnoise+smnoise,color='gray',alpha=0.4)
            factors = [1.0 for x in upper_ceilings]
            plt.yticks(**fp)

        combined_values = {'M':[],'V':[],"L":[]}

        for mind,model in enumerate(model_plot_dict):

            if model in ['M','V','L']:
                continue

            val = [corr_data_dict[model][region][s][0]*factors[s-1] for s in SUBJECTS]
            datum = np.array(val)
            mean = datum.mean()
            sem = scipy.stats.sem(datum)
            
            if model in ['clip','virtex','virtexv2','icmlm_attfc','icmlm_tfm','tsmresnet50','audioclip']:
                combined_values['M'].append(mean)
            elif model in ['BERT','GPT2','CLIP-L']:
                combined_values['L'].append(mean)
            else:
                combined_values['V'].append(mean)

            label,addendum,color=model_plot_dict[model]
            plt.bar(mind+addendum,mean,width=1.,label=label,color=color,edgecolor='black',alpha=0.7)
            _,c,_ = plt.errorbar(mind+addendum,mean,yerr=sem,lolims=True,color='black',capsize=0)
            for capline in c:
                capline.set_marker('_')
            

        xticks_for_models = []
        for xind,x in enumerate(['M','V','L']):

            label,addendum,color=model_plot_dict[x]

            plt.bar(1.3*(mind+xind+addendum-5.5),np.array(combined_values[x]).mean(),width=1.3,color=color,edgecolor='black',alpha=0.7,linestyle='--')
            xticks_for_models.append(1.3*(mind+xind+addendum-5.5))
            _,c,_ = plt.errorbar(1.3*(mind+xind+addendum-5.5),np.array(combined_values[x]).mean(),yerr=scipy.stats.sem(np.array(combined_values[x])),lolims=True,color='black',capsize=0)
            for capline in c:
                capline.set_marker('_')
            
        plt.xticks(xticks_for_models,["M","V","L"],rotation=25)

        counter += 1

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.legend(bbox_to_anchor=(1.,1.),fontsize=12,ncol=2)

    if save_name is not None:
        plt.savefig(f"{save_name}.pdf",bbox_inches='tight')



 
def plot_summaryfig(save_name=None,normalized_version=True):

    fp = {'fontsize':16}
    plt.figure(figsize=(6,4))
    plt.suptitle(get_plot_title(MAIN_TASK),fontsize=10)

    plt.axhline(1.,linestyle='dashed')
    plt.ylabel('Normalized Correlation',fontsize=14)

    xticks_for_models = []
    labels_for_models = []

    asterisk_counter = 0
    heights = [0.4,0.1,0.3,0.2,0.1,0.2]

    for ind,region in enumerate(REGIONS):

        upper_ceilings = upper_ceiling_dict[region]
        labels_for_models.append(REGIONS[region])
        xticks_for_models.append(5*ind+1.5)

        plt.xticks([])
        plt.yticks(**fp)


        factors = [1.0/x for x in upper_ceilings]
        combined_values = {'M':[],'V':[],"L":[]}

        for model in model_plot_dict:

            if model in ['M','V','L']:
                continue

            val = [corr_data_dict[model][region][s][0]*factors[s-1] for s in SUBJECTS]
            datum = np.array(val)
            mean = datum.mean()
            sem = scipy.stats.sem(datum)
            
            if model in ['clip','virtex','virtexv2','icmlm_attfc','icmlm_tfm','tsmresnet50','audioclip']:
                combined_values['M'].append(mean)
            elif model in ['BERT','GPT2','CLIP-L']:
                combined_values['L'].append(mean)
            else:
                combined_values['V'].append(mean)


        for xind,x in enumerate(['M','V','L']):
            label,addendum,color=model_plot_dict[x]

            plt.bar(((5*ind)+xind+addendum-5.5),np.array(combined_values[x]).mean(),width=1.,color=color,edgecolor='black',alpha=0.7,linestyle='--')
            _,c,_ = plt.errorbar(1.*(5*ind+xind+addendum-5.5),np.array(combined_values[x]).mean(),yerr=scipy.stats.sem(np.array(combined_values[x])),lolims=True,color='black',capsize=0)
            for capline in c:
                capline.set_marker('_')
            

            _,p1 = args.STATISTICAL_TEST(combined_values['M'],combined_values['L'],equal_var=False)
            if p1 < 0.05:
                logger.debug(f'Statistically siginificant differences in region {region}: \t \t ML ')
            _,p1 = args.STATISTICAL_TEST(combined_values['M'],combined_values['V'],equal_var=False)
            if p1 < 0.05:
                logger.debug(f'Statistically siginificant differences in region {region}: \t \t MV ')
            _,p1 = args.STATISTICAL_TEST(combined_values['V'],combined_values['L'],equal_var=False)
            if p1 < 0.05:
                logger.debug(f'Statistically siginificant differences in region {region}: \t \t VL ')


            _ ,p = args.STATISTICAL_TEST(combined_values[x],combined_values['V'],equal_var=False)
            if p < 0.05:
                label,addendum,color=model_plot_dict[x]
                if xind < 1:
                    x1,x2 = ((5*ind)+xind+addendum-5.5) , ((5*ind)+xind+addendum-5.5) + 1.
                else:
                    x1,x2 = ((5*ind)+xind+addendum-5.5) -1. , ((5*ind)+xind+addendum-5.5)


                h = heights[asterisk_counter]
                y,col = np.array(combined_values[x]).mean()+scipy.stats.sem(np.array(combined_values[x])),'k'
                plt.text((x1+x2)*.5,y+h,"*",ha='center',va='bottom',color=col)
                plt.annotate('',xy=(x1,y+h-0.04),xytext=(x2,y+h-0.04),arrowprops={'connectionstyle':'bar','arrowstyle':'-','shrinkA':20,'shrinkB':20})
                asterisk_counter += 1

    plt.xticks(xticks_for_models,labels_for_models,rotation=15,fontsize=12)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.legend(bbox_to_anchor=(1.,1.),fontsize=12,ncol=2)

    if save_name is not None:
        plt.savefig(f"{save_name}_fancyfig.pdf",bbox_inches='tight')




plot_noise_ceilings(save_name=f"{args.SAVE_DIR}_noiseceilings")
plot_mainfig(save_name=f"{args.SAVE_DIR}_normalized_selectedvoxels",normalized_version=True)
plot_mainfig(save_name=f"{args.SAVE_DIR}_unnormalized_selectedvoxels",normalized_version=False)
plot_summaryfig(save_name=f"{args.SAVE_DIR}_normalized_selectedvoxels",normalized_version=True)
