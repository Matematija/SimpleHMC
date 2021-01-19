from time import time
import numpy as np

class Sampler():

    '''
    An abstract base sampler class for possible future extension.
    '''
    
    def __init__(self, logp, dim, verbose=False):

        '''
        logp: Callable
            Function with signature logp(x, *args, **kwargs) that evaluates to the log-probability.
            It needs to be written in a batched form so as to accept 2D arrays
            of shape [batch_size, dim] and returns a 1D array of shape [batch_size].

        dim: Int
            The target space dimensionality.

        verbose: Bool
            Determines if the sampler prints out messages during longer sampling runs or not.
        '''

        self.logp = logp
        self.dim = int(dim)
        self.verbose = bool(verbose)
        
    @property
    def logp(self):
        return self.__logp
    
    @logp.setter
    def logp(self, logp):
        assert callable(logp), f'"logp" has to be a callable, got {type(logp)}'
        self.__logp = logp
        
    @property
    def verbose(self):
        return self.__verbose
    
    @verbose.setter
    def verbose(self, verbose):
        self.__verbose = bool(verbose)
        
    def MH_accept(self, x, x_, **params):

        '''
        The Metropolis-Hastings accept/reject step.
        
        x: Array of shape [batch_size, dim]
            Current position/momentum state.

        x_: Array of shape [batch_size, dim]
            Proposed position/momentum states.

        Returns: Array of Bool with shape [batch_size] indicating if the state was accepted.
        '''

        logp_old = self.logp(x, **params)
        logp_new = self.logp(x_, **params)

        return logp_new - logp_old >= np.log(np.random.rand(x.shape[0]))
    
    def propose(self):
        raise NotImplementedError('Method "propose" not implemented for this sampler!')
        
    def next_state(self, x, **params):
        raise NotImplementedError('Method "next_state" not implemented for this sampler!')
        
    def sample(self, init, n_samples, warmup=0, sweep=1, **params):

        '''
        Samples the target probability distribution. Supports running parallel chains.

        init: Array of shape [n_chains, ndim]
            Determines the number of parallel Markov chains to be run and sets their initial states.

        n_samples: Int
            Sets the target numner of samples to be output.

        warmup: Int
            Sets the number of chain iterations to be run before any samples are collected.

        sweep: Int
            Sets the number of chain transitions to be performed in between recording samples. Should be set to
            a value high enough to avoid correlated samples but low enough to avoid long sampling times.

        **params:
            Additional keyword arguments to be passed to the logp and grad_logp functions.

        Returns: A 3D array with shape [n_chains, n_samples, dim] with all of the samples.
        '''
        
        x = np.atleast_2d(np.asarray_chkfinite(init))
        markov_chain = np.zeros(shape=(n_samples, *x.shape))

        if self.verbose:
            t0 = time()
            accept_counter = 0
            
        for _ in range(warmup):
            x, accepted = self.next_state(x, **params)
        
        for t in range(n_samples):
            
            for _ in range(sweep):
                x, accepted = self.next_state(x, **params)
            
            markov_chain[t] = x.copy()

            if self.verbose:
                accept_counter += np.mean(accepted)

                if time() - t0 > 15:
                    print('Current step: {} out of {}.'.format(t+1, n_samples))
                    t0 = time()

        if self.verbose:
            print('Average acceptance rate: {:6f}'.format(accept_counter/n_samples))

        return markov_chain.transpose([1,0,2]).squeeze()