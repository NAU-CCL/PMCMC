# %%
import sys
from sys import argv

sys.path.append('../')

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import pandas as pd

from particle_filter import particlefilter
from pmcmc import particlemcmc
from math_utils import nbinom_logpmf,norm_logpmf,beta_logpdf,uniform_logpdf,poisson_logpmf

from matplotlib.backends.backend_pdf import PdfPages

# %%
data = pd.read_csv('./AZ_FLU_HOSPITALIZATIONS.csv',index_col = False).to_numpy().T[1]
data = np.expand_dims(data[:200],0)


# %%
def SIRH_model(particles,observations,t,dt,model_params,rng):
    '''Definition of SEIR model as described in Calvetti's paper. Difference 
    is the use of Tau leaping to introduce stochasticity into the system and continuous log-normal OU process definition for beta.'''
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

def SIRH_Obs(data_point, particle_observations, model_params):
    weights = poisson_logpmf(k = data_point,mu = particle_observations[:,0] + 0.005)
    return weights

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


# %%

param_names = ['hosp','D','mean_ou','sig']

times = [70,80,90,100,110,120,130,140,150,160,170,180]

for run in range(len(times)):
    burn_in = 50_000
    output = np.load(f'Results/PMCMC_Output_{run}.npz')

    pf_params = {'num_particles':10_000, 
                      'dt':0.1,
                      'model':SIRH_model,
                      'observation_model':SIRH_Obs,
                      'model_dim':5,
                      'particle_initializer':SIRH_init,
                      'forecast_time':times[run]
                      }

    par = np.mean((output['accepted_params'][:,burn_in:]),axis = 1)
    
    pf_output = particlefilter(data = data,
        model_params= par,
        pf_params = pf_params,
        rng = np.random.default_rng(0),
        req_jit=True
        )

    t_vec = np.arange(0,200,1)

    with PdfPages(f'figures/figure_forecast_{times[run]}.pdf') as pdf:

        fig,axs = plt.subplots(nrows = 2,ncols = 4,figsize = (20,10))

        for i in range(3):
            axs[0,i].hist(output['accepted_params'][i,burn_in:],bins = 50)
            axs[0,i].set_title(f"{param_names[i]}, Mean: {np.round(np.mean(output['accepted_params'][i,burn_in:]),2)}, std: {np.round(np.std(output['accepted_params'][i,burn_in:]),2)}")

        axs[1,0].set_title(f"{param_names[3]}, Mean: {np.round(np.mean(output['accepted_params'][3,burn_in:]),2)}, std: {np.round(np.std(output['accepted_params'][3,burn_in:]),2)}")
        axs[1,0].hist(output['accepted_params'][3,burn_in:],bins = 50)

        axs[0,3].fill_between(t_vec,np.percentile(pf_output['particle_distribution'][:, 4, :].T, 12.5, axis=1),
                              np.percentile(pf_output['particle_distribution'][:, 4, :].T, 87.5, axis=1),
                              alpha=0.5, color='steelblue')

        #plot the mean of S compartment
        axs[1,1].plot(np.mean(pf_output['particle_distribution'][:, 0, :], axis=0), label='S')
        axs[1,1].legend()

        axs[1,2].set_title('Log Likelihood')
        axs[1,2].plot(output['Log_Likelihood'][burn_in:])

        axs[1,3].set_title('Real Data')
        axs[1,3].fill_between(t_vec,
                                      np.percentile(pf_output['particle_observations'][:, 0, :].T, 12.5, axis=1),
                                      np.percentile(pf_output['particle_observations'][:, 0, :].T, 87.5, axis=1),
                                      alpha=0.5, color='steelblue')
        
        axs[1,3].plot(data.T,'--',color = 'black')

        # Save the current figure to the PDF
        pdf.savefig(fig)  # Save the figure to the PDF
        plt.close(fig)    # Close the figure to free up memory



