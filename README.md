# SimpleHMC

A short-and-simple Python implementation of the vanilla Hamiltonian Monte Carlo sampling algorithm. Intended for ease-of-use in simple numerical experiments.

Written in pure Python and has NumPy as the only dependency (for now).

# A simple 1D example

Define a test probability distribution and its gradient:

```python
def logp(x):
    return np.squeeze(-(x**2)/2 + np.sin(10*x))

def grad_logp(x):
    return (-x + 10*np.cos(10*x)).reshape(-1,1)
```

Import the module and simply sample:

```python
from hmc import HMCSampler

sampler = HMCSampler(
    logp=logp,
    grad_logp=grad_logp,
    dim=1, dt=0.21, n_leaps=10, verbose=True)

init = np.random.uniform(-3, 3, size=(10,1))
n_samples = 2000
sweep = 10
warmup = 100

hmc_samples = hmc.sample(
    init=init,
    n_samples=n_samples,
    sweep=sweep,
    warmup=warmup)
```

The output is a NumPy 3D array with shape [n_chains, n_samples, dim] containing the samples.

# Notes

Note that the `init` variable implicitly set the number of chains to be run in parallel to 10 through its first dimension. Generally `init.shape` should be `(n_chains, n_dimensions)`.

All implemented methods support first-dimension batching so NumPy internals can handle some of the looping for you. It only requires a similarly batched implementation of `logp` and `grad_logp` (the log-probability and its gradient).

Most mehods have been annotated and commented within source files so documentation should be easily available.