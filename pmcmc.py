import numpy as np
import numba as nb
from particle_filter import pf_validator
from numpy.linalg import cholesky,LinAlgError

def pmcmc_validator(data,pmcmc_params,pf_params,rng,req_jit):

    '''
    This function is used to check the user has included the correct parameters to run pMCMC. 

    Args:
        data: A (observation_dim,T) matrix of observations of the system. 
        pmcmc_params: A dictionary of parameters and their values pertaining to pMCMC. 
        pf_params: A dictionary of parameters and their values pertaining to the particle filter. 
        req_jit: A boolean indicating whether to jit compile the particle filter. 

    Returns: 
        The specific version of the particle filter to be fed into the particlemcmc_internal for computation. 
    '''

    #Check mandatory pmcmc keys for particle filtering specification.
    key_list = ['iterations','init_params','prior','init_cov','burn_in'] 
    for key in key_list:
        if not (key in pmcmc_params.keys()):
            raise AttributeError(f'Required key {key} not found in pmcmc_params.') 

    if((pmcmc_params['init_params'].shape[0] != pmcmc_params['init_cov'].shape[0]) or (pmcmc_params['init_params'].shape[0] != pmcmc_params['init_cov'].shape[1])):
        raise AttributeError('init_cov shape must match init_params.')

    #Run the validator for particle_filter to check its parameters. 
    p_filter = pf_validator(data,pmcmc_params['init_params'],pf_params,rng,req_jit)

    return p_filter

def particlemcmc(data,pmcmc_params,pf_params,rng,req_jit = False,adaptive = False): 
    '''
    This function is the public interface for pMCMC. It will first run the validator to obtain the particle filter function 
    and then the pmcmc loop. 

    Args: 
        data: A (observation_dim,T) matrix of observations of the system. 
        pmcmc_params: A dictionary of parameters and their values pertaining to pMCMC. 
        pf_params: A dictionary of parameters and their values pertaining to the particle filter. 
        req_jit: A boolean indicating whether to jit compile the particle filter. 
        adaptive: A boolean indicating whether to use adaptive mcmc.  

    Returns: 
        A dictionary containing the output information of the pMCMC. The markov chain over the parameters is accepted_params, the associated log likelihoods are in Log Likelihood, 
        and the particle and observation distributions from pf associated with the MLE are in MLE_particle_dist and MLE_particle_observations respectively. 

    '''


    p_filter = pmcmc_validator(data,pmcmc_params,pf_params,rng,req_jit)
    param_set,LL,MLE_Particles,MLE_Observations = particlemcmc_internal(data,
                          num_particles = pf_params['num_particles'],
                          model_dim = pf_params['model_dim'],
                          init_params = pmcmc_params['init_params'],
                          prior = pmcmc_params['prior'],
                          iterations = pmcmc_params['iterations'],
                          rng = rng,
                          p_filter=p_filter,
                          adaptive=adaptive,
                          init_cov= pmcmc_params['init_cov'],
                          burn_in = pmcmc_params['burn_in']
                          )
   
    return {'accepted_params':param_set,'Log Likelihood':LL,'MLE_particle_dist':MLE_Particles,'MLE_particle_observations':MLE_Observations}

def particlemcmc_internal(data,num_particles,model_dim,init_params,init_cov,prior,iterations,rng,p_filter,adaptive,burn_in):
    '''
    The internals of pmcmc. 

    Args:
      data: A (observation_dim,T) matrix of observations of the system. 
      num_particles: The number of particles to use for the particle filter. 
      model_dim: The number of state variables in the system of interest. 
      init_params: The initial guess for the estimated parameters . 
      init_cov: The initial covariance matrix for the proposal distribution. 
      prior: A function which when evaluated yields the joint log probability of the current parameters under the prior. 
      iterations: The number of metropolis-hastings steps. 
      rng: The random number generator used in sampling. 
      p_filter: The particle filter function. This will come from pf_validator. 
      adaptive: A boolean telling the algorithm whether to use the adaptive mcmc.
      burn_in: The number of iterations the run before turning on the adaptive covariance. Irrelevant if adaptive = False. 

      Returns:
        Tuple of arrays representing the parameters, log likelihood, MLE of the particles and the MLE of the observations. 

    '''

    #Create matrices to hold the MLE of the particles and the observations as the algorithm progresses. 
    MLE_Particles = np.zeros((num_particles,model_dim,data.shape[1])) 
    MLE_Observations = np.zeros((num_particles,data.shape[0],data.shape[1]),dtype=np.float64)

    MLE = -5000000 #Start the MLE at a very small value.  

    #Create matrices to hold the markov chain of theta and the current value of the log_likelihood. 
    theta = np.zeros((len(init_params),iterations))
    LL = np.zeros((iterations,))

    #These two variables store covariance information for the adaptive step. 
    mu = np.zeros(len(init_params))
    cov = init_cov

    theta[:,0] = init_params
    LL[0] = prior(init_params) 

    if(np.isfinite(LL[0])): #Only run the particle filter if the parameter value is within the support of the prior. 
        particles,particle_observations,weights,likelihood = p_filter(
                                data = data,
                                model_params= theta[:,0],
                                rng = rng)
        

        LL[0] += np.sum(likelihood)

        MLE = LL[0]
        MLE_Particles = particles
        MLE_Observations = particle_observations

    #create a zero vector to store the acceptance rate
    acc_record = np.zeros((iterations,))

    '''PMCMC Loop'''

    for iter in range(1,iterations): 
        
        if(iter % 10 == 0):
            #print the acceptance rate and likelihood every 10 iterations
            print(f"iteration: {iter}" + f"| Acceptance rate: {np.sum(acc_record[:iter])/iter}" + f"| Log-Likelihood: {LL[iter-1]}" + f"| Proposal {theta[:,iter - 1]}")

        '''Generating the next proposal using the multivariate normal distribution'''

        theta_prop = rng.multivariate_normal(theta[:,iter - 1],(2.38**2/len(theta[:,iter - 1])) * cov)

        LL_new = prior(theta_prop)

        if(np.isfinite(LL_new)): #Only run the particle filter if the parameter value is within the support of the prior. 
            particles,particle_observations,weights,likelihood = p_filter(
                                    data = data,
                                    model_params= theta_prop,
                                    rng = rng)
            
            
            
            LL_new += np.sum((likelihood)) #Likeihood from PF is given elementwise, so sum over the whole series. 

            '''Store the running maximum likelihood estimate'''
            if(LL_new > MLE):
                MLE = LL_new
                MLE_Particles = particles
                MLE_Observations = particle_observations

        ratio = (LL_new - LL[iter-1])

            ###Random number for acceptance criteria
        u = rng.uniform(0.,1.)
        if np.log(u) < ratio: 
            theta[:,iter] = theta_prop
            LL[iter] = LL_new
            acc_record[iter] = 1
        else: 
            theta[:,iter] = theta[:,iter - 1]
            LL[iter] = LL[iter-1]

        if adaptive and (iter > burn_in): #Update the covariance if adaptive is on. 
            mu, cov = cov_update(cov,mu,theta[:,iter],iter,burn_in)
        

    return theta,LL,MLE_Particles,MLE_Observations

def cov_update(cov, mu, theta_val,iteration,burn_in):

    '''Adaptive update step, geometric cooling g ensures ergodicity of the markov chain 
    as the iteration count goes to infinity. '''

    g = (iteration - burn_in + 1) ** (-0.4)
    mu = (1.0 - g) * mu + g * theta_val
    m_theta = theta_val - mu
 
    try: 
        r_cov = (1.0 - g) * cov + g * np.outer(m_theta,m_theta.T)
    except LinAlgError: 
        r_cov = cov

    return mu,r_cov

