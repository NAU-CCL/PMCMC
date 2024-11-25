import numpy as np
from scipy.stats import multivariate_normal
from numba import njit


'''We start each time step by running the tau leaping algorithm on our
day time population. The day time tau leap is moving compartments to different locations.
We assume that the time clock starts at 8am in the morning. We run the tau leap for 1/3 of the day, 
and then we run the night time tau leap for 2/3 of the day.'''
tau_day = 1 / 3 # Time step for daytime tau leap
tau_night = 2 / 3 # Time step for night time tau leap   

def gen_movement(rng,population,min=0.03, max=0.08, Mov=1, Chain=1):
    '''Generates a random movement matrix for testing the simulation, chain controls whether the movement is seqential between nodes. '''
    movement = np.zeros((len(population), len(population)))
    if Mov==1:
        for i in range(len(population)):
            for j in range(len(population)):
                movement[i][j] = rng.integers(min * population[j], max * population[j])
        np.fill_diagonal(movement, 0)
        #if Chain==1, only allow movement from i to i+1, everything else is 0
    if Chain==1:
        movement = np.zeros((len(population), len(population)))
        for i in range(1,len(population)):
            movement[i][i-1] = rng.integers(min * population[i-1], max * population[i-1])

            
    return movement

@njit
def movers_calc(S,I, mov_ratio):
    '''Performs matrix algebra to compute the number of movers in both the S and I compartments and their destination
    if mov_ratio is a nxn matrix and S,I nx1 vectors then mov_S and mov_I are nxn matrices of movers. 
    '''
    mov_S = mov_ratio@np.diag(S)
    mov_I = mov_ratio@np.diag(I)
    return mov_S, mov_I

@njit
def mover_infection(movement,mov_S, mov_I, beta, tau_day):
    '''Calculates the rate of flux from S to I for movers. 
    Elementwise division and NaNs are handled in np.where'''
    mov_SI= (mov_I * mov_S)/movement

    mov_SI = np.where(~np.isfinite(mov_SI),0,mov_SI)
    #tau leap for the day time movement infection force

    return np.diag(beta)@mov_SI*tau_day



def SIR_tau_leap(rng, population, movement, initial_cond, theta):
    '''Runs a single iteration of tau leaping on a '''

    n = len(population)
    #initialize the result array
    result = np.zeros((n,3,2))
    result[:,:,0] = initial_cond   

    mov_ratio = movement@np.linalg.inv(np.diag(population))

    #start the tau leap for the day time
    #separate the movers in S,I,R compartments with movement matrix
    #simulate the trsmission among movers
    #first compute the in movement of people in each compartment

    mov_S,mov_I = movers_calc(result[:,0,0],result[:,1,0],mov_ratio)

    #then compute the transmitted people from S to I
    #with the transmission rate being beta_i, which is the destination of the movement
    #movement is the total population movement

    #tau leap for the day time movement infection force

    transfer_SI = rng.poisson(mover_infection(movement, mov_S, mov_I, theta[:n], tau_day))

    # #update the S,I,R compartments with the local transmission rate
    # #extract the S,I,R compartments
    # #subtract the movers
    # #last term mov_ratio@result[:,0,i-1] records the number of sus/infected people that are moving in the destination location
    S = result[:,0,0]-np.sum(mov_S, axis=0).T + mov_ratio@result[:,0,0] 
    I = result[:,1,0]-np.sum(mov_I, axis=0).T + mov_ratio@result[:,1,0] 
    R = result[:,2,0]
    
    #generate a poisson random number for each time step
    force_of_infection = rng.poisson(tau_day*theta[:n]*S*I/population)
    result[:,0,1] = S - force_of_infection
    #recover rate is 0.2 per day
    force_of_recovery = rng.poisson(theta[-1]*I*tau_day)
    result[:,1,1] = I + force_of_infection - force_of_recovery
    result[:,2,1] = R + force_of_recovery

    #start the tau leap for the night time
    #add the newly infected people back to their home location
    result[:,0,1]=result[:,0,1]+np.sum(mov_S, axis=0).T - np.sum(transfer_SI, axis=0).T - mov_ratio@result[:,0,0] 
    result[:,1,1]=result[:,1,1]+np.sum(mov_I, axis=0).T + np.sum(transfer_SI, axis=0).T - mov_ratio@result[:,1,0] 
    #find the negative values in ressult[:,1,i]
    #The reason for the negative values is that the force of infection is too high
    #find the index of the negative values in S
    neg_index_S = np.where(result[:,0,1]<0)
    #if neg_index_S is not empty
    if neg_index_S[0].size != 0:
        #save the negative values
        neg_value_S = result[neg_index_S,0,1]
        #set the negative values to 0
        result[neg_index_S,0,1] = 0
        #add the negative values to the infected compartment
        result[neg_index_S,1,1] = result[neg_index_S,1,1] + neg_value_S

    #find the index of the negative values in I 
    neg_index_I = np.where(result[:,1,1]<0)
    #if neg_index is not empty
    if neg_index_I[0].size != 0: 

        #save the negative values
        neg_valu_I = result[neg_index_I,1,1]
        #set the negative values to 0
        result[neg_index_I,1,1] = 0
        #add the negative values to the recovered compartment
        result[neg_index_I,2,1] = result[neg_index_I,2,1] + neg_valu_I 
    #update the S,I,R compartments with the local transmission rate
    #extract the S,I,R compartments
    S = result[:,0,1]
    I = result[:,1,1]
    R = result[:,2,1]
    #generate a poisson random number for each time step
    force_of_infection = rng.poisson(tau_night*theta[:n]*S*I/population)
    result[:,0,1] = S - force_of_infection
    #recover rate is 0.2 per day
    force_of_recovery = rng.poisson(theta[-1]*I*tau_night)
    result[:,1,1] = I + force_of_infection - force_of_recovery
    result[:,2,1] = R + force_of_recovery

    #find the negative values in ressult[:,1,i]
    #The reason for the negative values is that the force of infection is too high
    #find the index of the negative values in S
    neg_index_S = np.where(result[:,0,1]<0)
    #if neg_index_S is not empty
    if neg_index_S[0].size != 0:
        #save the negative values
        neg_value_S = result[neg_index_S,0,1]
        #set the negative values to 0
        result[neg_index_S,0,1] = 0
        #add the negative values to the infected compartment
        result[neg_index_S,1,1] = result[neg_index_S,1,1] + neg_value_S

    #find the index of the negative values in I 
    neg_index_I = np.where(result[:,1,1]<0)
    #if neg_index is not empty
    if neg_index_I[0].size != 0: 

        #save the negative values
        neg_valu_I = result[neg_index_I,1,1]
        #set the negative values to 0
        result[neg_index_I,1,1] = 0
        #add the negative values to the recovered compartment
        result[neg_index_I,2,1] = result[neg_index_I,2,1] + neg_valu_I 
    
    return result

@njit
def mm_init(num_particles, model_dim, rng):
    particles_0 = np.zeros((num_particles,model_dim))
    particles_0[:,0:2] = np.array([500,200])

    particles_0[:,2:4] = rng.integers(1,10,size = (num_particles,2))
    particles_0[:,0:2] -= particles_0[:,2:4]
    

    return particles_0

def mm_obs(data_point, particle_observations, theta):
    weights = np.zeros(len(particle_observations))
    for p in range(len(weights)):
        weights[p] = multivariate_normal.logpdf(data_point, particle_observations[p,:],np.diag(data_point ** 2)) 
    return weights