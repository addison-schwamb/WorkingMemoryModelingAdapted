import numpy as np
from scipy import sparse

class Network:
	def __init__(params, seed = 0):
		# initialize properties of the network
		# very slightly adapted from Elham's initialize_net method
		
		# Set parameters
		net_prs = params['network']
		train_prs = params['train']
		N = net_prs['N']
		rng = np.random.RandomState(seed)
		std = 1/np.sqrt(net_prs['pg']*N)
		
		# initialize weighting matrices
		Pw = np.eye(N)/train_prs['alpha_w']
		Pd = np.eye(N)/train_prs['alpha_d']
		J = std * sparse.random(N, N, density=net_prs['pg'], random_state = seed, data_rvs=rng.randn).toarray()
		
		if train_prs['init_dist'] == 'Gauss':
			x = 0.1*rng.randn(N,1)
			wf = (1. * rng.randn(N, net_prs['d_output'])) / net_prs['fb_var']
			wi = (1. * rng.randn(N, net_prs['d_input'])) / net_prs['input_var']
			wfd = (1. * rng.randn(N, net_prs['d_input'])) / net_prs['fb_var']
		
		elif train_prs['init_dist'] == 'Uniform':
			print('Uniform Initialization')
			x = 0.1 * (2 * rng.rand(N,1) - 1)
			wf = (2 * rng.rand(N, net_prs['d_output']) - 1) / net_prs['fb_var']
			wi = (2 * rng.rand(N, net_prs['d_input']) - 1) / net_prs['input_var']
			wfd = (2 * rng.rand(N, net_prs['d_input']) - 1) / net_prs['fb_var']
			
		else:
			print('Invalid Distribution: Supported distributions are \'Gauss\' and \'Uniform\'')
		
		wo = np.zeros([N,net_prs['d_output']])
		wd = np.zeros([N,net_prs['d_input']])
		
		model_prs = {'Pw': Pw, 'Pd': Pd, 'J': J, 'x': x, 'wf': wf, 'wo': wo, 'wfd': wfd, 'wd': wd, 'wi': wi}
		
		self.params = net_prs
		self.params.update(train_prs)
		self.params.update(model_prs)
		
		print(self.params.keys())

	
	def memory_trial():
		# run a single trial of the memory task, for training or testing
	
	def update_weights():
		# update the weights of the neural network during training
	
	def save_ICs():
		# save the initial conditions leading to a given response in testing
	
	def remove_neurons():
		# remove neurons, i.e., set cells in the JT matrix to 0