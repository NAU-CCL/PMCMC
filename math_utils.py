import numpy as np
from numpy import log
import numba as nb

'''Functions in this file are used to evaluate distributions. We need to implement the pdfs 
ourselves so they can be jit compiled.'''

@nb.njit(fastmath=True,error_model='numpy')
def gammaln_nr(z):
    """Numerical Recipes 6.1
    
    Args: 
      z: Value to evaluate in the log_gamma function. 
    
    Returns: 
      out: Value of log-gamma(z)
    """
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
  '''The log-pdf for the uniform distribution. 

  Args: 
    theta: The value at which to evaluate the distribution. 
    min_val: The minimum value of the support. 
    max_val: The maximum value of the support. 

  Returns: 
    Log-probability of theta
  
  '''
  if(theta < max_val and theta > min_val):
    return np.log(1/(max_val - min_val))

  return np.log(0.)

@nb.njit(fastmath=True,error_model='numpy')
def nbinom_logpmf(x,n,p):
  '''The log-pmf for the negative binomial distribution. 

  Args:
    x: The value at which to evaluate the distribution. 
    n: The number of successes until the experiment is stopped. 
    p: The probability of success on each trial.
  
  '''
  coeff = gammaln_nr(n+x) - gammaln_nr(x+1) - gammaln_nr(n)
  return coeff + n*log(p) + x * np.log(-p + 1)
   
@nb.njit(fastmath=True,error_model='numpy')
def poisson_logpmf(k,mu):
  '''The log-pmf for the poisson distribution. 

  Args: 
    k: Value at which to evaluate the distribution. 
    mu: Parameter of the poisson distribution. 

  Returns: 
    Log-probabilty of k. 

  '''
  return k * log(mu) - gammaln_nr(k + 1) - mu

@nb.njit(fastmath=True,error_model='numpy')
def norm_logpmf(k,mu,sig):
  '''
  log-pdf for the normal distribution. 

  Args: 
    k: Value at which to evaluate the distribution. 
    mu: Mean of the distribution. 
    sig: Standard deviation of the distribution. 

  Returns: 
    Log-probability of k. 
  '''
  return - log(sig) -1/2 * log(2 * np.pi) -1/2 * ((k - mu)/sig)**2

@nb.njit(fastmath=True,error_model='numpy')
def beta_logpdf(k,alpha,beta): 
  '''log-pdf for the beta distribution
  
  Args: 
    k: Value at which to evaluate the distribution. 
    alpha: parameter of the beta distribution. 
    beta: parameter of the beta distribution. 

  Returns: 
    log-probability of k. 
  
  '''
  return (alpha - 1) * log(k) + (beta - 1) * log(1-k) + gammaln_nr(np.array([alpha]) + np.array([beta])) - gammaln_nr(np.array([alpha])) - gammaln_nr(np.array([beta]))
