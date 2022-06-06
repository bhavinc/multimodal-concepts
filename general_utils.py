
import numpy as np

import scipy
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform

import sklearn
from sklearn.linear_model import LinearRegression


def get_rdm(vectors, distance='correlation'):
    """
    Computes the rdm of the given vectors using the specified distance.
    """
    rdm = pdist(vectors, distance)
    if not np.any(np.isnan(rdm)):
        return pdist(vectors, distance)
    else:
        raise ValueError('Found some NaNs in the RDM')

def calculate_noise_ceilings(all_rdms,corr_func=pearsonr):
    """
    Calcuilate the lower noise ceilings for each subject.
    """
    total_nums = len(all_rdms)
    
    all_corrs = []
    for each_sub in range(1,total_nums+1):
        sub_rdm = all_rdms[each_sub-1]
        other_rdms = [all_rdms[x-1] for x in range(1,total_nums+1) if x!=each_sub]
        assert len(other_rdms)== (total_nums - 1) 
      
        all_corrs.append(np.array([corr_func(x,sub_rdm)[0] for x in other_rdms]).mean())
    
    assert len(all_corrs) == total_nums
    upper_ceiling = np.array(all_corrs)

    # mean = upper_ceiling.mean()
    # sem = scipy.stats.sem(upper_ceiling)
    # return mean,sem

    return upper_ceiling
