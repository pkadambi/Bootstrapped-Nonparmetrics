import tqdm
from scipy.stats import powerlaw
from scipy.optimize import curve_fit
from nonparametrics import *
import numpy as np

def generate_sample_sizes(nsamp_sizes, min_sampsize, max_sampsize, samp_spacing='logunif'):
    '''

    :param nsamp_sizes: the number of subsample sizes (an even number)
    :param min_sampsize: minimum subsample size
    :param max_sampsize: should be no more than
    :param samp_spacing:
    :return:
    '''

    if 'log' in samp_spacing and 'unif' in samp_spacing:
        numspacing = int(np.floor(nsamp_sizes/2))
        logspaced = np.logspace(np.log10(min_sampsize), np.log10(max_sampsize), numspacing)
        linspaced = np.linspace(min_sampsize, max_sampsize, numspacing)
        subsamp_sizes = np.concatenate([logspaced, linspaced])
        subsamp_sizes = np.sort(np.unique(subsamp_sizes))

    elif 'log' in samp_spacing:
        subsamp_sizes = np.logspace(np.log10(min_sampsize), np.log10(max_sampsize), nsamp_sizes)

    elif 'unif' in samp_spacing:

        subsamp_sizes = np.linspace(min_sampsize, max_sampsize, nsamp_sizes)

    return subsamp_sizes.astype('int')

def get_MC_iters_per_sampsize(samp_sizes, data_size, nMCiters, maxMCiters=None):
    #TODO: incomplete function --> it should be possible to do fewer estimates at larger values of sample size since the variance is less
    '''
    Function checks to see if there are actually more permutations than the number of given MC iterations
    :param data_size: size of the dataset
    :param samp_sizes: 1-d array of sample sizes
    :param maxMCiters: the maximum number of mc estimates to be performed at any value of subsample size
    :return: number of monte carlo iterations to do at each value of the sample size
    '''

    mc_iter_sizes = nMCiters * np.ones_like(samp_sizes)

    # for subsamp_size in samp_sizes:
    available_permutations = np.ceil(data_size/samp_sizes)

    for ii, (subsamp_size, permutations) in zip(samp_sizes, available_permutations):
        if permutations<nMCiters:
            pass
    if maxMCiters is not None:
        mc_iter_sizes[mc_iter_sizes>maxMCiters] = mc_iter_sizes

    return mc_iter_sizes


def asymptotic_estimator(samp_sizes, div_estimates):
    '''
    :param samp_sizes: This list of sample sizes
    :param div_estimates: The list of dpdiv estimates
    :param quantity: The quantity being calculated must be either 'BER' or 'DIV' (divergence)
    :return: the coefficients of the power law (including the asymptotic value)
    '''

    # TODO: add weighted regression
    def _powlaw(x, a, b, asymp):
        return asymp + a * (x ** b)

    #constraints to ensure that asymptotic value is [0,1], exponent absolute value b>0

    constraints = ((0., 0., 1.), (np.inf, 1., 1.))

    result = curve_fit(_powlaw, samp_sizes, div_estimates, bounds=constraints)

    powlaw_constants = {'a': result[0], 'b': result[1], 'asymp': result[2]}
    asymp_value = result[2]
    return powlaw_constants, asymp_value


def display_subsample_size_warining():
        print("------------------------------------------------------------------------")
        print("WARNING: Maximum sub-sample size is larger than one half the dataset size!")
        print("For high values of sub-sample size overlap between MC draws is high")
        print("This reduces the effectiveness of the bootstrap sampling and ***re-introduces higher bias*** at larger subsample sizes!")
        print("Consider reducing `max_subsamp_size` if asymptotic estimate is not providing an improved estimate")
        print("------------------------------------------------------------------------")

def estimate_asmptotic_value(class0data, class1data, num_subsamp_sizes=100, nruns= 50, min_subsamp_size=None,
                             max_subsamp_size=None, graph_method='1nn', k=None, debug=False):
    #TODO: `k` for >1-nn, and for MST
    '''
    :param xdata: input dataset
    :param ydata: input labels
    :param min_subsamp_size: defaults to num_features+1
    :param max_subsamp_size: defaults to nsamples/2
    :param metric:
    :return: asymptotic value, or more depending if debug flag is enabled
    '''
    xdata = np.vstack([class0data, class1data]) #TODO: for unbalanced classes

    if max_subsamp_size is None:
        max_subsamp_size = int(np.floor(np.shape(class0data)[0]/2))
        print(f'Assigned maximum subsample size {max_subsamp_size}')

    if max_subsamp_size > np.floor(np.shape(class0data)[0]/2):
        display_subsample_size_warining()
        max_subsamp_size = int(np.floor(np.shape(class0data)[0]/2))

    if min_subsamp_size is None or min_subsamp_size<np.shape(xdata)[1]:
        min_subsamp_size = int(np.shape(xdata)[1])
        print(f'Assigned minimum subsample size {min_subsamp_size}')

    if num_subsamp_sizes > max_subsamp_size - min_subsamp_size:
        num_subsamp_sizes = max_subsamp_size - min_subsamp_size
        print('WARNING: Specified number of subsample sizes is larger than (max_subsamp_size - min_subsamp_size)')
        print(f'Set num_subsamp_sizes to:\t {num_subsamp_sizes}')


    values = np.zeros([nruns, num_subsamp_sizes])
    mc_iterations = int(np.ceil(np.shape(xdata)[0]/2))
    subsamp_sizes = generate_sample_sizes(num_subsamp_sizes, min_subsamp_size,
                                          max_subsamp_size, samp_spacing='logunif')
    for ii in tqdm.tqdm(range(nruns)):
        for jj, sampsize in enumerate(subsamp_sizes):
            rslts = np.zeros(mc_iterations)
            for kk in range(mc_iterations):
                inds = np.random.choice(np.arange(len(class0data)), sampsize, replace=False)
                rslts[kk] =  dp_div(class0data[inds, :], class1data[inds, :])[0]
            values[ii, jj] = np.mean(rslts)

    dpdivmeans = np.mean(values, axis=0)

    powerlaw_constants, asymp_estimate = asymptotic_estimator(subsamp_sizes, dpdivmeans)
    if debug:
        return asymp_estimate,  powerlaw_constants, values, subsamp_sizes,  mc_iterations
    else:
        return asymp_estimate