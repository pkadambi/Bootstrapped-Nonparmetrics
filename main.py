
'''
Experiment 1:
'''
import numpy as np
from nonparametrics import *
from asymptotic import *
np.random.seed(7890)

n_dims = 8
n_samps_per_class=2000
a = np.random.rand(n_samps_per_class, n_dims)
b = np.hstack([np.random.rand(n_samps_per_class, 1) + 0.5, np.random.rand(n_samps_per_class, n_dims - 1)])

'''
Estimate pointwise valuSe
'''
point_estimate = dp_div(a, b, method='1nn')[0]
print('Point Estimates')
print(point_estimate)

'''
Estimate asymptotic value
'''
# print(estimate_asmptotic_value(a, b, nruns=5))

asymp_estimate,  powerlaw_constants, values, subsamp_sizes,  mciters = \
    estimate_asmptotic_value(a, b, nruns=1, n_mc_iters=100, debug=True)
# asymp_estimate,  powerlaw_constants, values, subsamp_sizes,  mciters = \
#     estimate_asmptotic_value(a, b, nruns=10, n_mc_iters=10, debug=True)
print('Asymp Estimate')
print(asymp_estimate)

import matplotlib.pyplot as plt
plt.figure()
plt.title('Example:8-d uniform dataset')
plt.scatter(subsamp_sizes, np.mean(values, axis=0), label='Dp average of monte-carlo estimates')
plt.plot([0, max(subsamp_sizes)],[0.5, 0.5], '-g', label='GroundTruth')
plt.plot([0, max(subsamp_sizes)],[asymp_estimate, asymp_estimate], '-b', label='Asymp Estimate')
plt.plot([0, max(subsamp_sizes)],[point_estimate, point_estimate], '--r', label='Point Estimate')
plt.legend()
plt.xlabel('Sample Size')
plt.ylabel('Divergence')
plt.grid()

plt.figure()
plt.title('Number of MC iterations at Sample Size')
plt.xlabel('SampleSize')
plt.ylabel('# of MC Iterations (# estimates of Dp)')
plt.grid()
plt.scatter(subsamp_sizes, mciters)
plt.show()
