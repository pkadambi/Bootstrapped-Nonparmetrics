# Bootstrapped Nonparametrics

The purpose of this repo is to provide a set of functions to calculate a bootstrapped asymptotic estimate the Dp divergence, a non-parametric f-divergence described here: https://arxiv.org/pdf/1408.1182.pdf


**Example**
Run `python main.py` for a quick example.

This example shows an 8-d uniform dataset with ground truth Dp=0.5. 


**Relevant Functions**

The first relevant functions is `dp_div` found in `nonparametrics.py`, which calculates a point estimate of the divergence with either 1-nn or MST. 
1nn is 10x faster or more.

The second relevant function is `estimate_asmptotic_value` in `asymptotic.py`, which calculates the bootstrapped asymptotic estimate.

- The most relevant parameter is `n_mc_iters` which sets the number of monte-carlo iterations to be performed at each value of subsample size. 
- (Subsample sizes are automatically chosen from [D+1, ... , N/2] for feature dimension D and number of examples per class N)
- The `n_runs` parameter controls the number of nested boostraps (not recommended to change from default=1)

Refer to the paper here: https://ieeexplore.ieee.org/document/8335474
