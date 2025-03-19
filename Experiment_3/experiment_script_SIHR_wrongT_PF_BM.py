import sys
sys.path.append('../')

import numpy as np
import numba as nb

from particle_filter import particlefilter
from math_utils import nbinom_logpmf,norm_logpmf

from sys import argv


'''Read arguments from the parser''' 
decorrelation_time = int(argv[1])*7
deco_wrong = int(argv[2])*7
run=int(argv[3]) 

#if decorrelation_time == 0, then we set it to 1 day
if decorrelation_time == 0:
    decorrelation_time = 1
if deco_wrong == 0:
    deco_wrong = 1

'''Simulated Data, setting up the SIRH model'''
# setting up the random number generator
rng = np.random.default_rng(run)

'''model params''' 
model_params = {'gamma':1/100,'L':0.,'D':1/7,'hosp':1/7,'R':100,'sig_state':0.005}

'''OU params'''
OU_params = {'lam':1/decorrelation_time,'mean_ou':-1.3,'sig':0.3}

T = 250
dt = 0.1
t_vec = np.arange(0,T,dt)

'''IC'''
N = 1_000_000
I = 10

print(f"Expectation of OU Process: {np.exp(OU_params['mean_ou'] + (OU_params['sig']**2)/2)}")
print(f"Variance of OU Process: {np.exp(OU_params['sig']**2 - 1) * np.exp(2 * OU_params['mean_ou'] + OU_params['sig']**2)}")

'''Discretization of the log-OU process'''
A = np.exp(-OU_params['lam'] * dt)
M = OU_params['mean_ou'] * (np.exp(-OU_params['lam'] * dt) - 1)
C = OU_params['sig'] * np.sqrt(1 - np.exp(-2 * OU_params['lam'] * dt))

'''Generate a sample path of the log OU process to substitute into the model.'''
betas = np.zeros_like(t_vec)
betas[0] = 0.4
for time_index in range(1,len(t_vec)):
    betas[time_index] = np.exp(A * np.log(betas[time_index - 1]) - M + C * rng.normal(0,1))

'''Create an empty state vector to house the data. '''
state = np.zeros((4,len(t_vec)))
state[:,0] = np.array([N - I,I,0,0])


'''Simulate the system to obtain the data. Uses geometric brownian motion to stochastically simulate the system.'''
for time_index in range(1,len(t_vec)):
   new_S = ((model_params['L'] * state[3,time_index-1]) * dt)
   new_I = ((betas[time_index - 1] * (state[0,time_index-1] * state[1,time_index-1])/np.sum(state[:,time_index-1])) * dt)
   new_IH = ((model_params['D'] * model_params['gamma'] * state[1,time_index-1]) * dt)
   new_HR = ((model_params['hosp'] * state[2,time_index-1]) * dt)
   new_IR = ((model_params['D'] *(1 - model_params['gamma']) * state[1,time_index-1]) * dt)

   state[0,time_index] = np.maximum(0.,state[0,time_index-1] + new_S - new_I)
   state[1,time_index] = np.maximum(0.,state[1,time_index-1] + new_I - (new_IH + new_IR))
   state[2,time_index] = np.maximum(0.,state[2,time_index-1] + new_IH - new_HR)
   state[3,time_index] = np.maximum(0.,state[3,time_index-1] + new_HR + new_IR - new_S)

'''The data is the H compartment with negative binomial noise applied.'''
data = np.expand_dims(rng.negative_binomial(n = model_params['R'],p = model_params['R']/(model_params['R'] + state[2,::int(1/dt)] + 0.005)),0)

'''Model definition for the PF, uses geometric brownian motion to be in line with the test system.'''
def SIRH_model(particles,observations,t,dt,model_params,rng):
    D,gamma,L,hosp,R,sig_state,mu,sig,lam = model_params

    A = np.exp(-lam * dt)
    M = mu * (np.exp(-lam * dt) - 1)
    C = sig * np.sqrt(1 - np.exp(-2 * lam * dt))


    for index in range(particles.shape[0]):

        new_S = ((L * particles[index,3,t]) * dt) 
        new_I = ((particles[index,4,t] * (particles[index,0,t] * particles[index,1,t])/np.sum(particles[index,:,t])) * dt)
        new_IH = ((D * gamma * particles[index,1,t]) * dt)
        new_HR = ((hosp * particles[index,2,t]) * dt)
        new_IR = ((D *(1 - gamma) * particles[index,1,t]) * dt)

        particles[index,0,t] = particles[index,0,t] + new_S - new_I + sig_state/np.sum(particles[index,:,t]) * particles[index,0,t] *rng.normal(0,np.sqrt(dt))
        particles[index,1,t] = particles[index,1,t] + new_I - (new_IH + new_IR) + sig_state * particles[index,1,t] *rng.normal(0,np.sqrt(dt))
        particles[index,2,t] = particles[index,2,t] + new_IH - new_HR + sig_state * particles[index,2,t] *rng.normal(0,np.sqrt(dt))
        particles[index,3,t] = particles[index,3,t] + new_HR + new_IR - new_S + sig_state * particles[index,3,t] *rng.normal(0,np.sqrt(dt))
        particles[index,4,t] = np.exp(A * np.log(particles[index,4,t]) - M + C * rng.standard_normal())

        observations[index,0,t] = particles[index,2,t]

    return particles,observations

'''The observation function for the PF. Uses negative binomial noise. '''
def SIRH_Obs(data_point, particle_observations, model_params):
    r = 1/model_params[4]
    weights = nbinom_logpmf(x = data_point,p = r/(r + particle_observations[:,0] + 0.005),n = np.array([r]))
    return weights

'''Function to initialize the particle distribution to time zero. Only the S and I compartments are assumed to be nonzero to start.
Beta is initialized randomly between 0 and 1.'''
def SIRH_init(num_particles, model_dim, rng):
    particles_0 = np.zeros((num_particles,model_dim))
    particles_0[:,0] = 1_000_000
    I_init = rng.integers(0,200,size = (num_particles))
    particles_0[:,0] -= I_init
    particles_0[:,1] = I_init
    particles_0[:,4] = rng.uniform(0.,1., size = (num_particles,))
    

    return particles_0

'''Hyperparamter for PF'''
pf_params = {'num_particles':10_000, 
                      'dt':dt,
                      'model':SIRH_model,
                      'observation_model':SIRH_Obs,
                      'model_dim':5,
                      'particle_initializer':SIRH_init,
                      }

'''Where the PF is run, output is a dictionary containing the output of the PF. '''
output = particlefilter(data = data,
        model_params= (model_params['D'],
        model_params['gamma'],
        model_params['L'],
        model_params['hosp'],
        1/model_params['R'],
        model_params['sig_state'],
        OU_params['mean_ou'],
        OU_params['sig'],
        1/deco_wrong),
        pf_params = pf_params,
        rng = rng,
        req_jit=True
        )


'''Saves results to npz file.'''
np.savez_compressed(f'./Experiment_3/results/results_trueT_{decorrelation_time}_wrongT_{deco_wrong}_days_run_{run}.npz',particle_distribution = output['particle_distribution'],
particle_observations = output['particle_observations'],
log_weights = output['log_weights'],
Log_likelihood = output['Log_likelihood'],
data = data, 
state = state,
betas = betas
)

    


