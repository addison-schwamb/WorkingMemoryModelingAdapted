import numpy as np
from scipy import sparse

class Network:
    def __init__(self, params, seed = 0):
        # initialize properties of the network
        # very slightly adapted from Elham's initialize_net method
        
        # Set parameters
        print('constructing Network class')
        N = params['N']
        rng = np.random.RandomState(seed)
        std = 1/np.sqrt(params['pg']*N)
        
        # initialize weighting matrices
        Pw = np.eye(N)/params['alpha_w']
        Pd = np.eye(N)/params['alpha_d']
        J = std * sparse.random(N, N, density=params['pg'], random_state = seed, data_rvs=rng.randn).toarray()
        
        if params['init_dist'] == 'Gauss':
            x = 0.1*rng.randn(N,1)
            wf = (1. * rng.randn(N, params['d_output'])) / params['fb_var']
            wi = (1. * rng.randn(N, params['d_input'])) / params['input_var']
            wfd = (1. * rng.randn(N, params['d_input'])) / params['fb_var']
        
        elif params['init_dist'] == 'Uniform':
            print('Uniform Initialization')
            x = 0.1 * (2 * rng.rand(N,1) - 1)
            wf = (2 * rng.rand(N, params['d_output']) - 1) / params['fb_var']
            wi = (2 * rng.rand(N, params['d_input']) - 1) / params['input_var']
            wfd = (2 * rng.rand(N, params['d_input']) - 1) / params['fb_var']
            
        else:
            print('Invalid Distribution: Supported distributions are \'Gauss\' and \'Uniform\'')
        
        wo = np.zeros([N,params['d_output']])
        wd = np.zeros([N,params['d_input']])
        
        model_prs = {'Pw': Pw, 'Pd': Pd, 'J': J, 'x': x, 'wf': wf, 'wo': wo, 'wfd': wfd, 'wd': wd, 'wi': wi}
        
        self.params = params;
        self.params.update(model_prs)

    
    def memory_trial():
        # run a single trial of the memory task, for training or testing
        pass
    
    def update_weights():
        # update the weights of the neural network during training
        pass
    
    def save_ICs():
        # save the initial conditions leading to a given response in testing
        pass
    
    def remove_neurons():
        # remove neurons, i.e., set cells in the JT matrix to 0
        pass