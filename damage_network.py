"""
Functions for damaging neural networks
written in Python 3.8.3
@ Addison Schwamb
"""

import numpy as np
import time

def remove_neurons(JT, pct_rmv, inhibitory):
    tic = time.time()
    JT_dim = np.shape(JT)
    if inhibitory:
        remove = JT<0
    else:
        remove = JT>0
    num_to_keep = ((1-pct_rmv)*(remove.sum())).astype(int)
    keep_indices = np.random.randint(1,JT_dim[0],(2,num_to_keep))
    num_kept = 0

    for i in range(num_to_keep):
        if remove[keep_indices[0][i]][keep_indices[1][i]]:
            remove[keep_indices[0][i]][keep_indices[1][i]] = False
            num_kept += 1
        elif np.count_nonzero(remove[keep_indices[0][i]])>0:
            j = (keep_indices[1][i]+1)%(JT_dim[0])
            while not remove[keep_indices[0][i]][j]:
                j = (j+1)%(JT_dim[0])
            remove[keep_indices[0][i]][j] = False
            num_kept += 1
        elif np.count_nonzero(remove,axis=0)[keep_indices[1][i]]>0:
            j = (keep_indices[0][i]+1)%(JT_dim[1])
            while not remove[j][keep_indices[1][i]]:
                j = (j+1)%(JT_dim[1])
            remove[j][keep_indices[1][i]] = False
            num_kept += 1
        else:
            keep_indices[0][i] = np.random.randint(1,JT_dim[0])
            keep_indices[1][i] = np.random.randint(1,JT_dim[1])
            i -= 1
    JT[remove] = 0
    toc = time.time()
    
    print('time out: ', (toc-tic)/60)
    print(np.size(JT))
    print(num_to_keep)
    print(num_kept)
    return JT