import numpy as np

def autocorrelation(x, mean):

    '''
    Calculates the autocorrelation function of a chain of observables.

    x: 1D array of observables
    mean: An independent mean of x.
    '''

    assert x.ndim == 1, f"x has to be 1D, got {x.ndims}"
    xc = x - mean
    ac = np.correlate(xc, xc, mode='full')
    return ac[len(ac)//2:]

def int_autocorr_time(ac, cutoff=0.05):

    '''
    Calculates the integrated autocorrelation time from the autocorrelation function

    ac: An array with the output of `autocorrelation`.]
    
    cutoff: A value to cutoff the evaluation of the autocorrelation fn,
        to avoid integrating statistical noise.
    '''
    
    acn = ac/ac[0]
    i = np.argmax(acn < cutoff)
    
    return 0.5 + np.sum(acn[:i])