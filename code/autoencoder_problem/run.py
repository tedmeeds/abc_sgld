# TOOD: Adapt this for NN case
# Construct NN, load data
# Pass it to nn_wrapper

import numpy as np
import pylab as pp
import scipy as sp
from scipy import stats as spstats
from nn_wrapper import *
from working_code import *
from nn import *

keep_x        = True
init_seed     = 1
T             = 5000 # nbr of samples
verbose_rate  = 1000
C             = 20.01    # injected noise variance parameter
eta           = 1e-6 # step size for Hamiltoniam dynamics
#h = 0.0005

# params for gradients
d_theta = 0.001  # step size for gradient estimate
S       = 5
grad_params = {}

# keep theta within bounds
lower_bounds = np.array([0.001])
upper_bounds = np.array([np.inf])

if __name__ == "__main__":
  # pp.close('all')
  problem = NeuralNetworkProblem( NeuralNetwork([28*28, 100, 28*28], d_theta), load_mnist() )
  # Cheat by loading previous training NN
  # problem.nn.load('saved_at_counter-0.json')
  chain_id = 1
  params = {}
  params["chain_id"]  = chain_id
  params["init_seed"] = init_seed
  params["T"]       = T
  params["S"]       = S
  params["d_theta"] = d_theta
  params["eta"]     = eta
  params["C"]       = C
  params["batch_size"] = 100
  params["verbose_rate"] = verbose_rate
  params["grad_params"]  = {"logs":{"true":[],"true_abc":[],"2side_keps":[],"2side_sl":[]},\
                            "record_2side_sl_grad":False, "record_2side_keps_grad":False,"record_true_abc_grad":False,"record_true_grad":False,
                            "2side_keps": {"R": 5}}
  params["lower_bounds"] = lower_bounds
  params["upper_bounds"] = upper_bounds
  params["keep_x"]       = keep_x

  theta0 = problem.flatten(problem.nn.biases, problem.nn.weights)
  C = [C]*len(theta0)
  x0     = None

  # initialize the simulator
  np.random.seed(init_seed)
  params["grad_func"] = problem.two_sided_keps_gradient

  # init randomly for MCMC chain
  np.random.seed(init_seed + 1000*chain_id)

  # run algorithm
  outs = run_thermostats( problem, params, theta0, x0 )
  #outs = run_sghmc( problem, params, theta0, x0 )
  # outs = run_sgld( problem, params, theta0, x0 )
  #outs = run_mcmc( problem, params, theta0, x0 )

  # view results of single chain
  # problem.view_single_chain( outs )

  # pp.show()
