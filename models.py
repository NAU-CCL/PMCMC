import numpy as np
import numba as nb
from scipy.stats import norm

'''OU Implementations'''
@nb.njit
def OU_model(particles,observations,t,dt,theta,rng):
    particles[:,0,t] += -theta[0] * (particles[:,0,t] - theta[1]) * \
    dt + np.sqrt(2 * theta[0]) * theta[2] * rng.normal(size = (particles.shape[0],),scale = np.sqrt(dt))

    observations[:,0,t] = particles[:,0,t]
    return particles,observations

@nb.njit
def OU_Obs(data_point, particle_observations, theta):
    return 1/np.sqrt(2 * np.pi * theta[3]**2) * np.exp(-((data_point - particle_observations[:,0])**2)/(2 * theta[3] ** 2))

