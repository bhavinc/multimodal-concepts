#%%
####################################################
#    Imports
####################################################

import time
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('default')

import scipy
from scipy.spatial import distance
from scipy.stats import spearmanr,pearsonr,ttest_ind
from scipy.spatial.distance import squareform


from general_utils import *
from kamitani_utils import *
from omegaconf import OmegaConf


args = OmegaConf.load('analysis_config.yaml')
assert args.USE_TRAIN_BETAS == False  #we stopped doing that analysis now

# get global vars early on
if args.CORR_FUNC == 'pearsonr':
    CORR_FUNC = pearsonr
if args.CORR_FUNC == 'spearmanr':
    CORR_FUNC = spearmanr    

if args.STATISTICAL_TEST == 'ttest_ind':
    STATISTICAL_TEST = ttest_ind

if os.path.exists(args.SAVE_DIR) == False:
    os.mkdir(args.SAVE_DIR)

cm1 , cnum1 = plt.get_cmap('Blues') , 6.
cm2 , cnum2 = plt.get_cmap('Reds')  , 8.
cm3 , cnum3 = plt.get_cmap('Greens'), 3.

model_plot_dict = {
    'clip'          : ('CLIP',                1.5, cm3(1/cnum1)),
    'virtexv2'      : ('VirTex',           1.5, cm3(3/cnum1)),
    'icmlm-attfc'   : ('ICMLM-attfc',         1.5, cm3(4/cnum1)), 
    'icmlm-tfm'     : ('ICMLM-tfm',           1.5, cm3(5/cnum1)), 
    'tsmresnet50'   : ('TSMResNet50-visual',  1.5, cm3(6/cnum1)),


    'BiT'              : ('BiT-M',       3.0, cm2(1/cnum2)),
    'resnet'           : ('ResNet50',    3.0, cm2(2/cnum2)),
    'madryli8'            : ('AR-Linf8',    3.0, cm2(3/cnum2)),
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
    'L'     : ('L', 6.0, cm1(0.3*cnum1)),

}



def _get_sort_idx(mat):

    # we want to find voxels that respond maximally to the images
    # hence, let's first find voxels that have highest betas
    var_list=[mat[:,vox].max() for vox in range(mat.shape[1])]  
    sort_idx = np.argsort(var_list) 

    return sort_idx



def voxel_selection(data_dict,number_based=False,threshold_based=False,**kwargs):

    '''
        Voxel selection based on 2 criteria -- either number of voxels (number_based) or threshold on the beta values (threshold_based)
    '''

    if number_based:

        def select_topk_voxels(sampledata,num_voxels,sort_idx=None):
            '''
            Select the top n voxels based on the sort idx provided
            if sort_idx is None, then the max values will be calculated using max values 

            Remember that we are technically changing the order of the voxels. There will be effects because of this. 

            # TODO : what if there are high negative beta values? (Or do we not care about those? -- we dont since it means that the voxel shows anti-expt behaviour?)
            '''


            if num_voxels == None:
                return sampledata
            
            sort_idx = _get_sort_idx(sampledata) if sort_idx is None else sort_idx
            
            subsample = []
            for c in sort_idx[::-1]:        
                if len(subsample) == num_voxels:
                    # print (c,'--',sampledata[:,c].max())
                    break

                if np.all(sampledata[:,c] != 0.):
                    subsample.append(sampledata[:,c])

            subsample = np.stack(subsample,axis=1)
            return subsample




        num_voxel_dict = kwargs.pop('num_voxel_dict')
        sort_idx_dict = {_region:{sub_idx:_get_sort_idx(data_dict[_region][sub_idx]) for sub_idx in range(1,6)} for _region in data_dict.keys()}
        smaller_data_dict = {region : {sub: select_topk_voxels(data_dict[region][sub],num_voxels=num_voxel_dict[region],sort_idx=sort_idx_dict[region][sub]) for sub in range(1,6) } for region in data_dict.keys()}

        return smaller_data_dict




    if threshold_based:
        num_voxels_of_hippocampus = kwargs.pop('num_voxels_of_hippocampus')


        def get_threshold_dict(num_voxels_of_hippocampus):
            '''
            Get thresholds that you need to use as per the num_voxels_of_hippocampus. The thresholds are given per subject in a form of a dictionary.
            The idea is that we want to threshold the values of betas and use only those that are quite high. 
            '''

            threshold_dict = {s:0. for s in range(1,6)}
            
            for s in range(1,6):
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

            print ('Using threshold found : ',subsample.shape)

            return subsample


        threshold_dict = get_threshold_dict(num_voxels_of_hippocampus)
        smaller_data_dict = {region : {sub : select_thresholded_voxels(smaller_data_dict[region][sub],threshold=threshold_dict[sub]) for sub in range(1,6)} for region in data_dict.keys() }

        return smaller_data_dict



def select_voxels(data_dict):

    if args.MAIN_TASK == 'normal_correlations':
        return data_dict #don't change anything

    if args.MAIN_TASK == 'select_top30_voxels':
        # print ("HERE")
        num_voxel_dict = {r:30 for r in data_dict.keys()}
        smaller_data_dict = voxel_selection(data_dict,number_based=True,num_voxel_dict=num_voxel_dict)

        return smaller_data_dict

    if args.MAIN_TASK == 'select_voxels_based_on_hippocampus':
        num_voxels_of_hippocampus = 30
        smaller_data_dict = voxel_selection(data_dict,threshold_based=True,num_voxels_of_hippocampus=num_voxels_of_hippocampus)
        return smaller_data_dict


    if args.MAIN_TASK == 'max_of_each_region':

        num_voxel_dict = {
            (1007)                   : 235,
            (2007)                   : 355,
            (1007, 2007)             : 629,
            (1011, 1021)             : 888,
            (2011, 2021)             : 343,
            (1011, 2011, 1021, 2021) : 1367,
            (17)                     : 8,
            (53)                     : 17,
            (17, 53)                 : 29,
            (1016)                   : 56,
            (2016)                   : 108,
            (1016, 2016)             : 89,
        }

        smaller_data_dict = voxel_selection(data_dict,number_based=True,num_voxel_dict=num_voxel_dict)
        return smaller_data_dict



def get_upper_ceilings(smaller_data_dict):
    '''
        Calculate the upper_ceilings
    '''
    upper_ceiling_dict = {region: calculate_noise_ceilings([ get_rdm(smaller_data_dict[region][sub],distance=args.DISTANCE) for sub in range(1,6)  ],corr_func=CORR_FUNC) for region in smaller_data_dict.keys() }
    
    return upper_ceiling_dict



def get_model_rdms():
    tstart = time.time()
    model_data_dict = {model_1[0]:get_rdm(get_model_vectors(model_1[0],layer_to_use=model_1[1],use_test=args.USE_TEST_BETAS),distance=args.DISTANCE) for model_1 in args.MODELS}
    tend = time.time()
    print ("Time taken to get all model RDMS : ",tend-tstart)
    return model_data_dict



def get_correlations(smaller_data_dict):

    '''
        Main function that takes data dict and provides correlations with model rdms.
        smaller_data_dict : data dict of selected voxels (based on number or thresholds or No selection) 
    '''
    
    model_rdms_dict = get_model_rdms()


    corr_data_dict = {}
    for model_1 in args.MODELS:

        corr_data_dict[model_1[0]] = {k:{sub:None for sub in range(1,6)} for k in smaller_data_dict.keys()}
        for _region in smaller_data_dict.keys():
            for sub_idx in range(1,6):
            
                model_rdm = model_rdms_dict[model_1[0]]
                brain_vectors = smaller_data_dict[_region][sub_idx]
                corr_data_dict[model_1[0]][_region][sub_idx] = CORR_FUNC(model_rdm,get_rdm(brain_vectors,distance=args.DISTANCE)) if brain_vectors is not None else np.nan
    
    return corr_data_dict




interesting_regions_mapping = get_interesting_regions(args.USE_ATLAS)
data_dict = {_region : {sub_idx : get_region_vectors(sub_idx,_region,atlas=args.USE_ATLAS,use_train=args.USE_TRAIN_BETAS,use_test=args.USE_TEST_BETAS)  for sub_idx in range(1,6)} for _region in interesting_regions_mapping.keys()}
smaller_data_dict = select_voxels(data_dict)

upper_ceiling_dict = get_upper_ceilings(smaller_data_dict)
corr_data_dict = get_correlations(smaller_data_dict)




def plot_noise_ceilings(save_name=None,sort_idx_dict=None):


    def _change_flag(mat):
        try:
            get_rdm(mat,distance=args.DISTANCE)
        except ValueError:
            flag = False
            return flag
        else:
            flag = True
            return flag


    num_voxels = np.arange(2,2000,1)

    fp = {'fontsize':16}

    counter = 1
    plt.style.use('default')
    plt.figure(figsize=(18,12))
    plt.suptitle(f"{'Default' if args.USE_ATLAS=='default' else 'destrieux'} Atlas; {'train' if args.USE_TRAIN_BETAS else 'test'} Betas; Distance:{args.DISTANCE} ; corrfunc : {args.CORR_FUNC}",**fp)

    for region in smaller_data_dict.keys():

        ys,errs = [],[]
        rand_ys,rand_errs = [],[]

        for n in num_voxels:
            if n > data_dict[region][1].shape[1]:
                break


            five_rdms , rand_five_rdms = [],[]
            for subject in range(1,6):
                if sort_idx_dict is None:
                    sort_idx = _get_sort_idx(data_dict[region][subject]) 
                else:
                    sort_idx = sort_idx_dict[region][subject]
                    

                # using only those voxels
                subsample = data_dict[region][subject][:,sort_idx[-n:]] 
                # if args.ZERO_CENTER_BRAIN_RDMS == True:
                    # subsample = zero_center(subsample)
                main_rdm = squareform(get_rdm(subsample,distance=args.DISTANCE))
                five_rdms.append(squareform(main_rdm))

                
                flag = False
                while flag == False:
                    # if args.ZERO_CENTER_BRAIN_RDMS == True:
                        # subsample = zero_center(subsample)                    
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


        print (f"Region {region} : At {np.argmax(np.array(ys))} point, max noise ceiling of {np.max(np.array(ys))} is found ")

        xs = list(num_voxels)[:len(ys)] 
        
        plt.subplot(4,3,counter)
        plt.title(f"{interesting_regions_mapping[region]}",**fp)    
        plt.xlabel('#Voxels chosen',**fp)
        plt.ylabel('Noise ceilings',**fp) if counter==1 or counter ==4 or counter == 7 else None


        plt.plot(xs,ys,marker='.',label='x voxels',alpha=0.9)
        plt.fill_between(xs, [ys[i]-errs[i] for i in range(len(ys))],[ys[i]+errs[i] for i in range(len(ys))],color='gray',alpha=0.4)


        plt.plot(xs,rand_ys,marker='.',label='random',alpha=0.3)
        plt.fill_between(xs, [rand_ys[i]-rand_errs[i] for i in range(len(rand_ys))],[rand_ys[i]+ rand_errs[i] for i in range(len(rand_ys))],color='gray',alpha=0.4)

        counter += 1
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1,1))

    if save_name is not None:
        plt.savefig(f'{save_name}.png',bbox_inches='tight')





def plot_fancyfig(save_name=None):


    fp = {'fontsize':16}
    plt.figure(figsize=(20,30))

    title_init = f"{args.MAIN_TASK} :"

    if args.MAIN_TASK == 'select_from_train_for_test':
        plt.suptitle(f"{title_init} Distance: {args.DISTANCE}; corrfunc {args.CORR_FUNC} \n selecting top voxels from test data on train data",fontsize=10)
    if args.MAIN_TASK == 'select_voxels_based_on_hippocampus':
        plt.suptitle(f"{title_init} Distance: {args.DISTANCE}; corrfunc {args.CORR_FUNC} \n thresholded using max value of the 30th hippocampus voxel",fontsize=10)
    if args.MAIN_TASK == 'normal_correlations' or args.MAIN_TASK=='select_top30_voxels':
        plt.suptitle(f"{title_init} Distance: {args.DISTANCE}; corrfunc {args.CORR_FUNC}",fontsize=10)


    plt.axhline(1.,linestyle='dashed')
    plt.ylabel('Normalized Correlation',fontsize=14)

    xticks_for_models = []
    labels_for_models = []

    heights = [0.4,0.1,0.3,0.2,0.1,0.2]

    beh = 0
    for ind,region in enumerate(smaller_data_dict.keys()):

        upper_ceilings = upper_ceiling_dict[region]
        labels_for_models.append(interesting_regions_mapping[region])
        xticks_for_models.append(5*ind+1.5)
        
        plt.xticks([])
        plt.yticks(**fp)


        factors = [1.0/x for x in upper_ceilings]

        combined_values = {'M':[],'V':[],"L":[]}

        for mind,model in enumerate(model_plot_dict.keys()):

            if model in ['M','V','L']:
                continue


            val = [corr_data_dict[model][region][s][0]*factors[s-1] for s in range(1,6)]
            datum = np.array(val)
            mean = datum.mean()
            # sem = scipy.stats.sem(datum)
            
            if model in ['clip','virtex','virtexv2','icmlm_attfc','icmlm_tfm','tsmresnet50','audioclip']:
                combined_values['M'].append(mean)
            elif model in ['BERT','GPT2','CLIP-L']:
                combined_values['L'].append(mean)
            else:
                combined_values['V'].append(mean)

        
        _,p1 = ttest_ind(combined_values['M'],combined_values['L'],equal_var=False)
        if p1 < 0.05:
            print ('ML \t \t',region)
        _,p1 = ttest_ind(combined_values['M'],combined_values['V'],equal_var=False)
        if p1 < 0.05:
            print ('MV \t \t',region,'\t \t',p1)
        _,p1 = ttest_ind(combined_values['V'],combined_values['L'],equal_var=False)
        if p1 < 0.05:
            print ('VL \t \t',region)

        print ("COMBINED_VALUES")
        print (combined_values)
        for xind,x in enumerate(['M','V','L']):
            label,addendum,color=model_plot_dict[x]

            plt.bar(((5*ind)+xind+addendum-5.5),np.array(combined_values[x]).mean(),width=1.,color=color,edgecolor='black',alpha=0.7,linestyle='--')
            _,c,_ = plt.errorbar(1.*(5*ind+xind+addendum-5.5),np.array(combined_values[x]).mean(),yerr=scipy.stats.sem(np.array(combined_values[x])),lolims=True,color='black',capsize=0)
            for capline in c:
                capline.set_marker('_')
            
            _ ,p = ttest_ind(combined_values[x],combined_values['V'],equal_var=False)

            if p < 0.05:
                label,addendum,color=model_plot_dict[x]
                print (region,x)

                if xind < 1:
                    x1,x2 = ((5*ind)+xind+addendum-5.5) , ((5*ind)+xind+addendum-5.5) + 1.
                else:
                    x1,x2 = ((5*ind)+xind+addendum-5.5) -1. , ((5*ind)+xind+addendum-5.5)


                print (beh)
                h = heights[beh]
                y,col = np.array(combined_values[x]).mean()+scipy.stats.sem(np.array(combined_values[x])),'k'
                plt.text((x1+x2)*.5,y+h,"*",ha='center',va='bottom',color=col)
                plt.annotate('',xy=(x1,y+h-0.04),xytext=(x2,y+h-0.04),arrowprops={'connectionstyle':'bar','arrowstyle':'-','shrinkA':20,'shrinkB':20})
                beh += 1
        
    plt.xticks(xticks_for_models,labels_for_models,rotation=15,fontsize=12)



    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.legend(bbox_to_anchor=(1.,1.),fontsize=12,ncol=2)

    if save_name is not None:
        plt.savefig(f"{save_name}_fancyfig.png",bbox_inches='tight')







def plot_mainfig(save_name=None,normalized_version=False):


    fp = {'fontsize':16}
    plt.figure(figsize=(9,12))

    title_init = f"{args.MAIN_TASK} :"

    if args.MAIN_TASK == 'select_from_train_for_test':
        plt.suptitle(f"{title_init} Distance: {args.DISTANCE}; corrfunc {args.CORR_FUNC} \n selecting top voxels from test data on train data",fontsize=10)
    if args.MAIN_TASK == 'select_voxels_based_on_hippocampus':
        plt.suptitle(f"{title_init} Distance: {args.DISTANCE}; corrfunc {args.CORR_FUNC} \n thresholded using max value of the 30th hippocampus voxel",fontsize=10)
    if args.MAIN_TASK == 'normal_correlations' or args.MAIN_TASK=='select_top30_voxels':
        plt.suptitle(f"{title_init} Distance: {args.DISTANCE}; corrfunc {args.CORR_FUNC}",fontsize=10)

    counter = 1
    for ind,region in enumerate(smaller_data_dict.keys()):

        upper_ceilings = upper_ceiling_dict[region]
        
        ax1 = plt.subplot(4,3,counter)
        plt.title(f"{interesting_regions_mapping[region]} : {smaller_data_dict[region][1].shape[1] } voxels")
        
        plt.xticks([])
        plt.yticks(**fp)


        if normalized_version == True:
            plt.axhline(1,linestyle='dashed')
            factors = [1.0/x for x in upper_ceilings]
        else:
            mnoise,smnoise=np.mean(upper_ceilings),scipy.stats.sem(upper_ceilings)
            if region == (17,53):
                print (f"Region : {region} mean {mnoise} {smnoise} ")
            plt.fill_between(np.arange(21),mnoise-smnoise,mnoise+smnoise,color='gray',alpha=0.4)
            factors = [1.0 for x in upper_ceilings]




        for mind,model in enumerate(model_plot_dict.keys()):

            if model in ['M','V','L']:
                continue


            val = [corr_data_dict[model][region][s][0]*factors[s-1] for s in range(1,6)]
            datum = np.array(val)
            mean = datum.mean()
            sem = scipy.stats.sem(datum)
            
            label,addendum,color=model_plot_dict[model]

            plt.bar(mind+addendum,mean,width=1.,label=label,color=color,edgecolor='black',alpha=0.7)
            _,c,_ = plt.errorbar(mind+addendum,mean,yerr=sem,lolims=True,color='black',capsize=0)
            for capline in c:
                capline.set_marker('_')
            

            val_resnet = [corr_data_dict['resnet'][region][s][0]*factors[s-1] for s in range(1,6)]

            if model == 'resnet':
                continue

            _ ,p = scipy.stats.wilcoxon(val,val_resnet)
            
            if p < 0.05:
                print (region,model)

                x1 = (mind+addendum)

                y,h,col = mean+sem, 0.7,'k'
                plt.text(x1,y+h,"*",ha='center',va='bottom',color=col)


        counter += 1



    plt.tight_layout()
    plt.subplots_adjust(top=0.8)

    if save_name is not None:
        plt.savefig(f"{save_name}.png",bbox_inches='tight')



if args.MAIN_TASK == 'select_top30_voxels':
    plot_mainfig(save_name=f"{args.SAVE_DIR}_normalized_selectedvoxels",normalized_version=True)
    plot_mainfig(save_name=f"{args.SAVE_DIR}_unnormalized_selectedvoxels",normalized_version=False)
    plot_fancyfig(save_name=f"{args.SAVE_DIR}_normalized_selectedvoxels",normalized_version=True)

if args.MAIN_TASK == 'max_of_each_region':
    plot_mainfig(save_name=f"{args.SAVE_DIR}_normalized_maxforeachregionvoxels",normalized_version=True)
    plot_mainfig(save_name=f"{args.SAVE_DIR}_unnormalized_maxforeachregionvoxels",normalized_version=False)
    plot_fancyfig(save_name=f"{args.SAVE_DIR}_normalized_maxforeachregionvoxels",normalized_version=True)

if args.MAIN_TASK == 'normal_correlations':
    plot_mainfig(save_name=f"{args.SAVE_DIR}_normalized_allvoxels",normalized_version=True)
    plot_mainfig(save_name=f"{args.SAVE_DIR}_unnormalized_allvoxels",normalized_version=False)
    plot_fancyfig(save_name=f"{args.SAVE_DIR}_normalized_allvoxels",normalized_version=True)

if args.MAIN_TASK == 'select_voxels_based_on_hippocampus':
    plot_mainfig(save_name=f"{args.SAVE_DIR}_normalized_thresholded_using_30hippocampus_voxels",normalized_version=True)
    plot_mainfig(save_name=f"{args.SAVE_DIR}_unnormalized_thresholded_using_30hippocampus_voxels",normalized_version=False)
    plot_fancyfig(save_name=f"{args.SAVE_DIR}_normalized_thresholded_using_30hippocampus_voxels",normalized_version=True)

if args.MAIN_TASK == 'select_from_train_for_test':
    plot_mainfig(save_name=f"{args.SAVE_DIR}_normalized_SelectingVoxelsFromTestForTrain",normalized_version=True)
    plot_mainfig(save_name=f"{args.SAVE_DIR}_unnormalized_SelectingVoxelsFromTestForTrain",normalized_version=False)
    plot_fancyfig(save_name=f"{args.SAVE_DIR}_normalized_SelectingVoxelsFromTestForTrain",normalized_version=True)



