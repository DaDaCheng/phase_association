
import numpy as np

from scipy.optimize import quadratic_assignment, linear_sum_assignment
from scipy.spatial import distance_matrix
def greed_assignment(A,B):
    num=A.shape[0]
    dM=distance_matrix(A,B)
    indexA=np.arange(num,dtype=int)
    indexB=np.zeros(num,dtype=int)
    dmax=dM.max()-dM.min()+1
    for i in range(num):
        d=np.argmin(dM)
        dA,dB=d//num,d%num
        indexB[dB]=indexA[dA]
        dM[dA,:]+=dmax
        dM[:,dB]+=dmax
    return indexB



def LAP(A,B):
    num=A.shape[0]
    dM=distance_matrix(B,A)
    row_ind, col_ind = linear_sum_assignment(dM)
    return col_ind
