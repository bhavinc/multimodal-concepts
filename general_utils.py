
import numpy as np

import scipy
from scipy.stats import pearsonr
from scipy.spatial.distance import pdist, squareform

import sklearn
from sklearn.linear_model import LinearRegression


def pcorr(x1,x2,y):
    """
    Computes the (signed square-root of) proportion of variance (r^2) in y that is uniquely explained
    by x1, uniquely explained by x2, or simultaneously explained by x1 and x2
    (intersection). These signed r values are returned respectively in
    unique_x1_r, unique_x2_r and inter_r.
    In terms of a venn diagram with 3 circles representing x1, x2 and y, the 3
    outputs represent the 3 separate regions of overlap with y.
    """
    x1x2   = np.concatenate([x1, x2], axis=1)
    lm1    = LinearRegression().fit(x1, y)
    lm2    = LinearRegression().fit(x2, y)
    lmboth = LinearRegression().fit(x1x2, y)

    x1_r2    = lm1.score(x1,y)
    x2_r2    = lm2.score(x2,y)
    union_r2 = lmboth.score(x1x2,y)
    inter_r2 = max(x1_r2 + x2_r2 - union_r2, 0)
    
    unique_x1_r2 = x1_r2 - inter_r2
    unique_x2_r2 = x2_r2 - inter_r2

    inter_r     = np.sqrt(inter_r2)
    unique_x1_r = np.sqrt(np.maximum(0, unique_x1_r2)) * np.sign(lmboth.coef_[0])
    unique_x2_r = np.sqrt(np.maximum(0, unique_x2_r2)) * np.sign(lmboth.coef_[1])

    # if np.count_nonzero(np.sign(lmboth.coef_[0]) != np.sign(lmboth.coef_[1])) != 0:
    #     print("Unequal sign")
    #     raise Exception("Unequal sign")
    
    return unique_x1_r, unique_x2_r, inter_r

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

    '''
    Calcuilate the lower noise ceilings for each subject.
    '''

    all_corrs = []
    
    for each_sub in range(1,6):

        sub_rdm = all_rdms[each_sub-1]
        other_rdms = [all_rdms[x-1] for x in range(1,6) if x!=each_sub]
        assert len(other_rdms)==4
        
        
        all_corrs.append(np.array([corr_func(x,sub_rdm)[0] for x in other_rdms]).mean())
    
    assert len(all_corrs) == 5
    upper_ceiling = np.array(all_corrs)

    # mean = upper_ceiling.mean()
    # sem = scipy.stats.sem(upper_ceiling)
    # return mean,sem

    return upper_ceiling
