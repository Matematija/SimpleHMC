import numpy as np
import os, sys

lib_path = os.path.abspath('..')

if lib_path not in sys.path:
    sys.path.append(lib_path)

from hmc.sampler import Sampler

class HMCSampler(Sampler):

    '''
    A Hamiltonian Monte Carlo sampler.
    '''

    def __init__(self, logp, grad_logp, dim, dt=0.1, n_leaps=10, m=1.0, verbose=False):

        '''
        logp: Callable
            Function with signature logp(x, **kwargs) that evaluates to the log-probability.
            It needs to be written in a batched form so as to accept 2D arrays
            of shape [batch_size, dim] and returns a 1D array of shape [batch_size].

        grad_logp: Callable
            Function with signature grad_logp(x, **kwargs) that evaluates to the log-probability.
            It needs to be written in a batched form so as to accept 2D arrays
            of shape [batch_size, dim] and returns a 2D array of shape [batch_size, dim] of
            input gradients along the 1st axis.

        dim: Int
            The target space dimensionality.

        dt: Float
            A time step to use with the leapfrog integrator when proposing new states.

        n_leaps: Int
            The number of leapfrog steps to take when proposing new states.

        m: float
            A (scalar) mass value to add to the kinetic energy term: p**2 / (2*m)

        verbose: Bool
            Determines if the sampler prints out messages during longer sampling runs or not.
        '''
        
        super().__init__(logp, dim, verbose)
        
        self.grad_logp = grad_logp
        self.dt = float(dt)
        self.n_leaps = int(n_leaps)
        self.m = float(m)
        
    @property
    def grad_logp(self):
        return self.__grad_logp
    
    @grad_logp.setter
    def grad_logp(self, grad_logp):
        assert callable(grad_logp), f'"grad_logp" has to be a callable, got {type(grad_logp)}'
        self.__grad_logp = grad_logp

    def propose(self, x0, p0, **params):

        '''
        The leapfrog integrator proposal.

        x0: Array of shape [batch_size, dim]
            The initial position state vector.

        p0: Array of shape [batch_size, dim]
            The initial momentum state vector.

        **params:
            Additional keyword arguments to pass to grad_logp.

        Returns: A 2-tuple of arrays of shape [batch_size, ndim] with proposed states.
        '''
        
        x, p = x0.copy(), p0.copy()

        p += (self.dt/2)*self.grad_logp(x, **params).reshape(-1, self.dim)

        for _ in range(1, self.n_leaps):
            x += self.dt*(p/self.m)
            p += self.dt*self.grad_logp(x, **params).reshape(-1, self.dim)

        x += self.dt*(p/self.m)
        p += (self.dt/2)*self.grad_logp(x, **params).reshape(-1, self.dim)

        return x, -p

    def MH_accept(self, x, p, x_, p_, **params):

        '''
        The Metropolis-Hastings accept/reject step.
        
        x, p: Arrays of shape [batch_size, dim]
            Current position/momentum state.

        x_, p_: Arrays of shape [batch_size, dim]
            Proposed position/momentum states.

        Returns: Array of Bool with shape [batch_size]
        '''

        H_old = np.sum(p**2, axis=-1)/(2*self.m) - self.logp(x, **params)
        H_new = np.sum(p_**2, axis=-1)/(2*self.m) - self.logp(x_, **params)

        return H_old - H_new >= np.log(np.random.rand(x.shape[0]))
    
    def next_state(self, x, **params):

        '''
        Advances the chain one step.

        x: Array of shape [batch_size, dim]

        **params:
            Additional keyword arguments to pass to logp and grad_logp.

        Returns: An array of shape [batch_size, dim] with proposed states.
        '''
        
        x_prop = x.copy()
        
        p = np.random.normal(scale=self.m, size=x.shape)
        x_, p_ = self.propose(x, p, **params)
        accepted = self.MH_accept(x, p, x_, p_, **params)
        
        x_prop[accepted] = x_[accepted].copy()
        
        return x_prop, accepted