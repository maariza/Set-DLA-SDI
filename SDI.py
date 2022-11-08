#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Fitting a lognormal parametrization to the histogram
from iminuit.cost import LeastSquares
from iminuit import Minuit


# In[ ]:


#For generate multiple nDLA
data = np.ones((256, 256))
seed = data.copy() - 1
countsF = np.zeros(bins.shape [0] - 1)

for i in range (50):
    "generate de DLA to reproduce"
    data= DLA(seed.copy(), data, int(1e3), maxiter, tolerance = 1e10, random_seed = i)
    "replicate de DLA with the mDLA"
    dla= DLA(seed.copy(), data, int(1e5), maxiter, tolerance = 1e6, random_seed  = i + 1)
    counts, edges, _ = ax[0].hist(dla_minus_seed.ravel(), bins=bins, density=True)
    countsF += counts

countsF /= 50


# In[ ]:


fig,ax=pplt.subplots(nrows=1, ncols=1, sharex=False, sharey=False)
countsF, edges, _  = ax[0].hist(dla_minus_seed.ravel(), bins=bins, density=True)
ax[0].format(xscale='log', xformatter='log')
mu, sigma = m.values
ax[0].plot(x, lognormal_pdf(x, *m.values))
print(m.values)


# In[ ]:


def lognormal_pdf(x, mu, sigma):
    
    return np.exp( -(np.log(x) - mu)**2 / (2 * sigma**2) ) / (x * np.sqrt(2 * np.pi * sigma**2))

y_err = np.ones_like(countsF) * 1e-1
#lsq = LeastSquares(x, countsF, y_err, lognormal_pdf)
m = Minuit(lsq, mu=1., sigma=0.96)
print(m)
m.migrad()

