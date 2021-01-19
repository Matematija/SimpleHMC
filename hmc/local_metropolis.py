import numpy as np
import os, sys

lib_path = os.path.abspath('..')

if lib_path not in sys.path:
    sys.path.append(lib_path)

from hmc.sampler import Sampler

class LocalMetropolisSampler(Sampler):

    '''
    A Local Metropolis Sampler. Update rule:
        1.) Choose a local degree of freedom x_i.
        2.) Update x_i with r ~ Uniform(-delta, delta)
        3.) Accept/reject with the Metropolis-Hastings rule.
    '''
    
    def __init__(self, logp, delta, dim, verbose=False):

        '''
        logp: Callable
            Function with signature logp(x, *args, **kwargs) that evaluates to the log-probability.
            It needs to be written in a batched form so as to accept 2D arrays
            of shape [batch_size, dim] and returns a 1D array of shape [batch_size].

        delta: float
            Defines the local update as x_i -> x_i  + Uniform(-delta, delta),
            for a selected local degree of freedom.

        dim: Int
            The target space dimensionality.

        verbose: Bool
            Determines if the sampler prints out messages during longer sampling runs or not.
        '''


        super().__init__(logp, dim, verbose)
        
        self.delta = float(delta)
        self.dim = int(dim)
        
    def propose(self, x):

        '''
        The local Metropolis proposal.

        x0: Array of shape [batch_size, dim]
            The initial position state vector.

        p0: Array of shape [batch_size, dim]
            The initial momentum state vector.

        **params:
            Additional keyword arguments to pass to grad_logp.

        Returns: An array of shape [batch_size, dim] with proposed states.
        '''
        
        bs = x.shape[0]
        x_ = x.copy()
        
        inds = np.random.randint(low=0, high=self.dim, size=bs)
        x_[range(bs), inds] += np.random.uniform(low=-self.delta, high=self.delta, size=bs)
        
        return x_
    
    def next_state(self, x, **params):

        '''
        Advances the chain one step.

        x: Array of shape [batch_size, dim]

        **params:
            Additional keyword arguments to pass to logp and grad_logp.
        '''
        
        x_ = x.copy()
        
        x_prop = self.propose(x)
        accepted = self.MH_accept(x, x_prop, **params)
        x_[accepted] = x_prop[accepted].copy()
        
        return x_, accepted

