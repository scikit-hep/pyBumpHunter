import numpy as np
from scipy.special import erfc, erfcinv



# Function used to fit the BumpHunter test statistic distribution
def bh_stat(x, pM, m, A):
    '''
    x : min p-value
    pM : median of p
    m : Number of test (window)
    A : Global scale
    '''
    xe = np.exp(-x)
    res = erfcinv(2*pM) * (2*erfcinv(2*xe) - erfcinv(2*pM))
    res = m * np.exp(res)
    res2 = (1 - 0.5*erfc(erfcinv(2*xe) - erfcinv(2*pM)))**(m-1)
    return A * (res * res2) * xe

