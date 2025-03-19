import sys
from sys import argv

sys.path.append('../')

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import pandas as pd
import jsonpickle

from particle_filter import particlefilter
from pmcmc import particlemcmc
from math_utils import nbinom_logpmf,norm_logpmf,beta_logpdf,uniform_logpdf,poisson_logpmf

times = [70,80,90,100,110,120,130,140,150,160,170,180]

data = pd.read_csv('./AZ_FLU_HOSPITALIZATIONS.csv',index_col = False).to_numpy().T[1]
data = np.expand_dims(data[:times[int(argv[1])]],0)

'''Model definition for the PF, uses geometric brownian motion.'''
def SIRH_model(particles,observations,t,dt,model_params,rng):
    hosp,D,mu,sig = model_params

    gamma = 1/1000
    L = 0.
    lam = 1/35
    sig_state = 0.005

    A = np.exp(-lam * dt)
    M = mu * (np.exp(-lam * dt) - 1)
    C = sig * np.sqrt(1 - np.exp(-2 * lam * dt))

    for index in range(particles.shape[0]):

        new_S = ((L * particles[index,3,t]) * dt) 
        new_I = ((particles[index,4,t] * (particles[index,0,t] * particles[index,1,t])/np.sum(particles[index,:,t])) * dt)
        new_IH = ((D * gamma * particles[index,1,t]) * dt)
        new_HR = ((hosp * particles[index,2,t]) * dt)
        new_IR = ((D *(1 - gamma) * particles[index,1,t]) * dt)

        particles[index,0,t] = np.maximum(0.,particles[index,0,t] + new_S - new_I + sig_state/np.sum(particles[index,:,t]) * particles[index,0,t] * rng.normal(0,np.sqrt(dt)))
        particles[index,1,t] = np.maximum(0.,particles[index,1,t] + new_I - (new_IH + new_IR) + sig_state * particles[index,1,t] * rng.normal(0,np.sqrt(dt)))
        particles[index,2,t] = np.maximum(0.,particles[index,2,t] + new_IH - new_HR + sig_state * particles[index,2,t] * rng.normal(0,np.sqrt(dt)))
        particles[index,3,t] = np.maximum(0.,particles[index,3,t] + new_HR + new_IR - new_S + sig_state * particles[index,3,t] * rng.normal(0,np.sqrt(dt)))

        particles[index,4,t] = np.exp(A * np.log(particles[index,4,t]) - M + C * rng.standard_normal())

        observations[index,0,t] = particles[index,2,t]

    return particles,observations

'''Observation function for the real data. Here we use poisson as the overdispersion was empirically observed to be small.'''
def SIRH_Obs(data_point, particle_observations, model_params):
    weights = poisson_logpmf(k = data_point,mu = particle_observations[:,0] + 0.005)
    return weights

'''Initializes the distribution of particles. 7,329,000 is the approximate population of Arizona. '''
def SIRH_init(num_particles, model_dim, rng):
    particles_0 = np.zeros((num_particles,model_dim))
    particles_0[:,0] = 7_329_000
    I_init = rng.integers(10_000,200_000,size = (num_particles))
    particles_0[:,0] -= I_init
    particles_0[:,1] = I_init
    particles_0[:,2] = 8
    particles_0[:,3] = 0
    particles_0[:,4] = rng.uniform(0.0,0.4, size = (num_particles,))
    
    return particles_0

'''Uniform prior for all the parameters in PMCMC.'''
def sirh_prior(theta):
    return uniform_logpdf(theta[0],min_val= 0.,max_val= 0.5) + \
    uniform_logpdf(theta[1],min_val = 0.,max_val = 0.5) + \
    uniform_logpdf(theta[2],min_val = -2,max_val = -0.5) + \
    uniform_logpdf(theta[3],min_val = 0.,max_val = 5.)


'''Hyperparameters for the PMCMC'''
pmcmc_params = {'iterations':200_000,
                'init_params':np.array([0.3,0.1,-0.6,0.5]),
                'prior':sirh_prior,
                'init_cov':  np.diag([0.001,0.001,0.001,0.001]),
                'burn_in':1_000}

'''Hyperparameters for the particle filter'''
pf_params = {'num_particles':1000, 
                      'dt':0.1,
                      'model':SIRH_model,
                      'observation_model':SIRH_Obs,
                      'model_dim':5,
                      'particle_initializer':SIRH_init
                      }

'''Runs the pmcmc, outputs a dictionary of information about the run.'''
pmcmc_output = particlemcmc(
                  data = data,
                  pmcmc_params=pmcmc_params,
                  pf_params=pf_params,
                  adaptive=False,
                  rng = np.random.default_rng(0),
                  req_jit=True
                  )

'''Saves the results to an npz file.'''
np.savez_compressed(f'Results/PMCMC_Output_{int(argv[1])}.npz',
data = data,
accepted_params = pmcmc_output['accepted_params'],
Log_Likelihood = pmcmc_output['Log Likelihood'], 
MLE_particle_dist = pmcmc_output['MLE_particle_dist'],
MLE_Observation_dist = pmcmc_output['MLE_particle_observations']
)