import numpy as np
from numpy import log
import numba as nb

@nb.njit(fastmath=True,error_model='numpy')
def gammaln_nr(z):
    """Numerical Recipes 6.1"""
    coefs = np.array([
    57.1562356658629235, -59.5979603554754912,
    14.1360979747417471, -0.491913816097620199,
    .339946499848118887e-4, .465236289270485756e-4,
    -.983744753048795646e-4, .158088703224912494e-3,
    -.210264441724104883e-3, .217439618115212643e-3,
    -.164318106536763890e-3, .844182239838527433e-4,
    -.261908384015814087e-4, .368991826595316234e-5])

    out=np.empty(z.shape[0])

    for i in range(z.shape[0]):
      y = z[i]
      tmp = z[i] + 5.24218750000000000
      tmp = (z[i] + 0.5) * np.log(tmp) - tmp
      ser = 0.999999999999997092

      n = coefs.shape[0]
      for j in range(n):
          y = y + 1.
          ser = ser + coefs[j] / y

      out[i] = tmp + log(2.5066282746310005 * ser / z[i])
    return out

@nb.njit(fastmath=True,error_model='numpy')
def uniform_logpdf(theta,min_val,max_val):

    if(theta < max_val and theta > min_val):
        return np.log(1/(max_val - min_val))
    return np.log(0.)

@nb.njit(fastmath=True,error_model='numpy')
def nbinom_logpmf(x,n,p):
  coeff = gammaln_nr(n+x) - gammaln_nr(x+1) - gammaln_nr(n)
  return coeff + n*log(p) + x * np.log(-p + 1)
   
@nb.njit(fastmath=True,error_model='numpy')
def poisson_logpmf(k,mu):
  return k * log(mu) - gammaln_nr(k + 1) - mu

@nb.njit(fastmath=True,error_model='numpy')
def norm_logpmf(k,mu,sig):
   return - log(sig) -1/2 * log(2 * np.pi) -1/2 * ((k - mu)/sig)**2

@nb.njit(fastmath=True,error_model='numpy')
def beta_logpdf(k,alpha,beta): 
  return (alpha - 1) * log(k) + (beta - 1) * log(1-k) + gammaln_nr(np.array([alpha]) + np.array([beta])) - gammaln_nr(np.array([alpha])) - gammaln_nr(np.array([beta]))
