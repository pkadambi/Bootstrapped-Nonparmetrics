
'''
Experiment 1:
    - Matching the entropy of labels from SLS and from


'''
import numpy as np
from nonparametrics import *
from asymptotic import *
n_dims = 8

a = np.random.rand(2500, n_dims)
b = np.hstack([np.random.rand(2500, 1) + 0.5, np.random.rand(2500, n_dims - 1)])

print('Point Estimates')
print(dp_div(a, b, method='1nn'))
# print(dp_div(a, b, method='mst'))

print('Asymp Estimate')
print(estimate_asmptotic_value(a, b))