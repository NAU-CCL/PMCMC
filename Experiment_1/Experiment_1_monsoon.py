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
from math_utils import nbinom_logpmf,norm_logpmf,beta_logpdf,uniform_logpdf

'''Simulated Data, setting up the SIRH model'''
# setting up the random number generator
rng = np.random.default_rng(int(argv[1]))

T = 250
dt = 0.1
t_vec = np.arange(0,T,dt)

'''IC'''
N = 1_000_000
I = 100

'''model params''' 
model_params = {'gamma':1/1000,'L':0.,'D':1/7,'hosp':1/10,'R':100,'sig_state':0.005}

'''OU params'''
OU_params = {'lam':1/35,'mean_ou':-1.3,'sig':0.4}

A = np.exp(-OU_params['lam'] * dt)
M = OU_params['mean_ou'] * (np.exp(-OU_params['lam'] * dt) - 1)
C = OU_params['sig'] * np.sqrt(1 - np.exp(-2 * OU_params['lam'] * dt))

betas = np.zeros_like(t_vec)
betas[0] = 0.4
for time_index in range(1,len(t_vec)):
    betas[time_index] = np.exp(A * np.log(betas[time_index - 1]) - M + C * rng.normal(0,1))

state = np.zeros((4,len(t_vec)))
state[:,0] = np.array([N - I,I,0,0])

for time_index in range(1,len(t_vec)):
   new_S = ((model_params['L'] * state[3,time_index-1]) * dt)
   new_I = ((betas[time_index - 1] * (state[0,time_index-1] * state[1,time_index-1])/np.sum(state[:,time_index-1])) * dt)
   new_IH = ((model_params['D'] * model_params['gamma'] * state[1,time_index-1]) * dt)
   new_HR = ((model_params['hosp'] * state[2,time_index-1]) * dt)
   new_IR = ((model_params['D'] *(1 - model_params['gamma']) * state[1,time_index-1]) * dt)

   state[0,time_index] = state[0,time_index-1] + new_S - new_I + model_params['sig_state']/np.sum(state[:,time_index-1])* state[0,time_index-1] * rng.normal(0,np.sqrt(dt))
   state[1,time_index] = state[1,time_index-1] + new_I - (new_IH + new_IR) + model_params['sig_state'] * state[1,time_index-1] *rng.normal(0,np.sqrt(dt))
   state[2,time_index] = state[2,time_index-1] + new_IH - new_HR + model_params['sig_state'] * state[2,time_index] * rng.normal(0,np.sqrt(dt))
   state[3,time_index] = state[3,time_index-1] + new_HR + new_IR - new_S + model_params['sig_state'] * state[3,time_index]  *rng.normal(0,np.sqrt(dt))

data = np.expand_dims(rng.negative_binomial(n = model_params['R'],p = model_params['R']/(model_params['R'] + state[2,::int(1/dt)] + 0.005)),0)


'''Particle Filter Code'''
def SIRH_model(particles,observations,t,dt,model_params,rng):
    '''SIRH model'''
    hosp,R,mu,sig,lam = model_params

    gamma = 1/1000
    L = 0.
    D = 1/7
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

def SIRH_Obs(data_point, particle_observations, model_params):
    r = 1/model_params[1]
    weights = nbinom_logpmf(x = data_point,p = r/(r + particle_observations[:,0] + 0.005),n = np.array([r]))
    return weights

def SIRH_init(num_particles, model_dim, rng):
    particles_0 = np.zeros((num_particles,model_dim))
    particles_0[:,0] = 1_000_000
    I_init = rng.integers(0,1000,size = (num_particles))
    particles_0[:,0] -= I_init
    particles_0[:,1] = I_init
    particles_0[:,4] = rng.uniform(0.,1., size = (num_particles,))
    
    return particles_0

def sirh_prior(theta):
    return uniform_logpdf(theta[0],min_val= 0.,max_val= 1.) + \
    uniform_logpdf(theta[1],min_val = 1/1000.,max_val = 1/5.) + \
    norm_logpmf(theta[2],mu = -0.8,sig = 0.4) + \
    beta_logpdf(theta[3],alpha = 3.,beta = 10.) + \
    beta_logpdf(theta[4],alpha = 1.5, beta = 10.)


pmcmc_params = {'iterations':100_000,
                'init_params':np.array([0.3,0.01,-0.5,0.5,0.2]),
                'prior':sirh_prior,
                'init_cov':  np.diag([0.001,0.001,0.001,0.001,0.001]),
                'burn_in':1_000}

pf_params = {'num_particles':1000, 
                      'dt':0.1,
                      'model':SIRH_model,
                      'observation_model':SIRH_Obs,
                      'model_dim':5,
                      'particle_initializer':SIRH_init
                      }
print("Starting PMCMC")
pmcmc_output = particlemcmc(
                  data = data,
                  pmcmc_params=pmcmc_params,
                  pf_params=pf_params,
                  adaptive=True,
                  rng = rng,
                  req_jit=True
                  )


np.savez_compressed(f'Results/PMCMC_Output_{int(argv[1])}.npz',
data = data,
state = state,
betas = betas,
model_params = np.array(list(model_params.values())),
OU_params = np.array(list(model_params.values())),
accepted_params = pmcmc_output['accepted_params'],
Log_Likelihood = pmcmc_output['Log Likelihood'], 
MLE_particle_dist = pmcmc_output['MLE_particle_dist'],
MLE_Observation_dist = pmcmc_output['MLE_particle_observations']
)


