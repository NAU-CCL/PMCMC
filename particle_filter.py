import numpy as np
import numba as nb

from numpy.typing import NDArray
from numpy.random import Generator 
from typing import Dict
from functools import partial

def particlefilter(data:NDArray[np.float64],model_params:NDArray[np.float64],pf_params:Dict,rng:Generator,req_jit = False):
    '''The public interface for the particle filter. The version of pf that will run is returned by the validator.
    
    Args:
        data: A (observation_dim,T) matrix of observations of the system. 
        model_params: Vector of system parameters, used in the observation density and transition density. 
        pf_params: A dictionary of parameters used in the particle filter. 
        rng: An instance of the NumPy Generator class. Used for random number generation. 
        req_jit: Whether to request the particle filter to be jit compiled in Numba. 

    Returns: 
        A dictionary of the output of the particle filter. Keys are particle_distribution, representing the particle cloud at each time step. 
        particle observations, similar as above but representing the observation realizations for each particle at each time step. Then log_weights and
        Log_likelihood are the importance weights and the likelihood used in pMCMC. 

        First 3 keys are (num_particles,T) and Log_Likelihood is (T,).
    '''



    p_filter = pf_validator(data,model_params,pf_params,rng,req_jit) #mild shape enforcement and checking of parameters.

    particles,particle_observations,weights,likelihood = p_filter(
    data,
    model_params,
    rng = rng,
    ) #Runs the particle filter
    
    return {'particle_distribution':particles,'particle_observations':particle_observations,'log_weights':weights,'Log_likelihood':likelihood}

def pf_validator(data:NDArray[np.float64],model_params:NDArray[np.float64],pf_params:Dict,rng:Generator,req_jit:bool = False):
    '''Checks if the arguments being passed to the particle filter are of the correct type and shape for the specified model.
    
    Takes the same arguments as the public particlefilter interface function. The return is the specific version of pf the user requested. 
    '''
    pf_params.setdefault('forecast_time', None)  # Set default value

    #First check the data shape is consistent with (m x T)
    if(len(data.shape) < 2):
        raise ValueError('Data shape must be m x T, where m represents the dimension of a single data point and T the length of the time series.')
    
    #Check the Generator is an instance of np.random.Generator
    if not isinstance(rng,Generator):
        raise AttributeError('The generator must be a NumPy Generator object as defined in np.random.')
    
    #Check mandatory keys for particle filtering specification.
    key_list = ['num_particles', 'dt','model','observation_model','model_dim','particle_initializer'] 
    for key in key_list:
        if not (key in pf_params.keys()):
            raise AttributeError(f'Required key {key} not found in pf_params.') 
        
    #Check if the provided functions are amenable to jit compilation
    jit_keys = ['observation_model','model','particle_initializer']
    if(req_jit):
        for key in jit_keys: 
            try: 
                pf_params[key] = nb.jit(pf_params[key], nopython=True)
            except: 
                raise 

    p_filter = nb.njit(filter_internal) if req_jit else filter_internal
    p_filter = partial(p_filter,
                       num_particles = pf_params['num_particles'],
                       dt = pf_params['dt'],
                       model = pf_params['model'],
                       observation_model = pf_params['observation_model'],
                       model_dim = pf_params['model_dim'],
                       particle_initializer = pf_params['particle_initializer'],
                       forecast_time = pf_params['forecast_time']
                       )

    return p_filter

def filter_internal(data:NDArray[np.float64],model_params:NDArray[np.float64],
                    num_particles,
                    dt,
                    rng,
                    model,
                    observation_model,
                    model_dim,
                    particle_initializer,
                    forecast_time):

    '''Initialize the particle distribution, observations and weights. 

    Args: 
        data: A (observation_dim,T) matrix of observations of the system. 
        model_params: Vector of system parameters, used in the observation density and transition density. 
        num_particles: How many particles to use to perform inference. 
        dt: Discretization step of a continuous time model, for discrete SSMs set to 1.  
        rng: An instance of the NumPy Generator class. Used for random number generation. 
        model: A python function describing the transition map for the model. Arguments are (particles,observations,t,dt,model_params,rng,num_particles)
        observation_model: A python function describing the observation density/measure. Arguments are (data_point, particle_observations, model_params)
        model_dim: dimension of the model 
        particle_initializer: Initializer function for the particles.
        forecast_time: Time point at which to discontinue weight computation and resampling

    Returns: 
        NumPy arrays representing the distribution of the particles, the observations, the weights, and the likelihood vector. 

        First 3 (num_particles,T), last (T,)
    '''

    particles = np.zeros((num_particles,model_dim,data.shape[1]),dtype = np.float64)
    particle_observations = np.zeros((num_particles,data.shape[0],data.shape[1]),dtype=np.float64)
    particles[:,:,0] = particle_initializer(num_particles,model_dim,rng)

    weights = np.zeros((num_particles,data.shape[1]),dtype = np.float64)
    likelihood = np.zeros((data.shape[1],),dtype=np.float64)

    for t,data_point in enumerate(data.T):

        '''Simulation/forecast step for all t > 0'''
        if(t > 0):
            particles[:,:,t] = particles[:,:,t-1]
            for _ in range(int(1/dt)):
                particles,particle_observations = model(particles, particle_observations, t, dt, model_params, rng)

        if forecast_time is None or t < forecast_time:
                '''Resampling and weight computation'''
                weights[:,t] = observation_model(data_point = data_point,
                                        particle_observations = particle_observations[:,:,t],
                                        model_params = model_params)

                jacob_sums = jacob(weights[:,t]) #Only performing this computation once to amortize the cost. 
                likelihood[t] = jacob_sums[-1] - np.log(num_particles) # Computes the Monte Carlo estimate of the likeihood. I.E. log(P(y_{1:T}))
                weights[:,t] -= jacob_sums[-1] #Normalization step
                indices = log_resampling(particles[:,:,t],particle_observations[:,:,t],weights[:,t],rng) #log_resampling returns a list of indices

    return particles,particle_observations,weights,likelihood

@nb.njit
def log_resampling(particles,particle_observations,weights,rng): 

    '''Systematic resampling algorithm in log domain, the njit decorator is important here as it gives a significant speedup. Time 
    complexity is O(n), as opposed to O(nlog(n)) in multinomial resampling. 
    
    Args:
        particles: A slice of the particle array at time t. 
        particle_observations: A slice of particle_observations at time t. 
        weights: A slice of the weights array at time t. 
        rng: The random number generator. 

    Returns: 
        The indices of the resampled particles. 

    '''

    indices = np.zeros(len(weights),dtype = np.int_) #initialize array to hold the indices
    cdf = jacob(weights)

    u = rng.uniform(0,1/len(weights)) #random number between 1 and 1/n, only drawn once vs the n draws in multinomial resampling
    i = 0
    for j in range(0,len(weights)): 
        r = np.log(u + 1/len(weights) * j)
        while r > cdf[i]: 
            i += 1
        indices[j] = i

    particles[:,:] = particles[indices,:] 
    particle_observations[:,:] = particle_observations[indices,:]

    return indices

@nb.njit
def jacob(δ:NDArray[np.float64])->NDArray[np.float64]:
    """The jacobian logarithm, used in log likelihood normalization and resampling processes
    δ will be an array of values. 
    
    Args: 
        δ: An array of values to sum

    Returns: 
        The vector of partial sums of δ.          
    
    """
    n = len(δ)
    Δ = np.zeros(n)
    Δ[0] = δ[0]
    for i in range(1,n):
        Δ[i] = max(δ[i],Δ[i-1]) + np.log(1 + np.exp(-1*np.abs(δ[i] - Δ[i-1])))
    return(Δ)
