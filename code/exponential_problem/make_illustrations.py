import numpy as np
import pylab as pp
import scipy as sp
from scipy import stats as spstats
from exp_wrapper import *
from abc_sgld.code.working_code import *

keep_x        = True 
init_seed     = 1
T             = 1000 # nbr of samples
verbose_rate  = 1000
C             = 1.01    # injected noise variance parameter
eta           = 0.01 # step size for Hamiltoniam dynamics
#h = 0.0005

# params for gradients
d_theta = 0.01  # step size for gradient estimate
S       = 5 
grad_params = {}
  
# keep theta within bounds
lower_bounds = np.array([0.001])
upper_bounds = np.array([np.inf])


# ----------------------- #
# common ransom seeds     #
# ----------------------- #
use_omega    = False    # use fixed random seeds for simulations
omega_rate   = 1.1    # probabilty of changing omegas
omega_switch = False    # if true, randomly change omegas
omega_sample = True   # sample omegas instead
assert (omega_switch * omega_sample) ==0, "only get to do one type of omega update"
omega_params = {"use_omega":use_omega, \
                "omega_rate":omega_rate, \
                "omega_switch":omega_switch,\
                "omega_sample":omega_sample}  

if __name__ == "__main__": 
  pp.close('all')
  
  chain_id = 1
  params = {}
  params["chain_id"]  = chain_id
  params["init_seed"] = init_seed
  params["T"]       = T
  params["S"]       = S
  params["d_theta"] = d_theta
  params["eta"]     = eta
  params["C"]       = C
  params["omega_params"] = omega_params
  params["verbose_rate"] = verbose_rate
  params["grad_params"]  = {"logs":{"true":[],"true_abc":[],"2side_keps":[],"2side_sl":[]},\
                            "record_2side_sl_grad":False, "record_2side_keps_grad":False,"record_true_abc_grad":True,"record_true_grad":False}
  params["lower_bounds"] = lower_bounds
  params["upper_bounds"] = upper_bounds
  params["keep_x"]       = keep_x
  
  theta0 = np.array( [0.2])
  x0     = None
  
  # initialize the simulator
  np.random.seed(init_seed)
  problem = generate_exponential( exp_problem )
  params["grad_func"] = problem.two_sided_sl_gradient

  # init randomly for MCMC chain
  np.random.seed(init_seed + 1000*chain_id)
  
  # run algorithm
  np.random.seed(init_seed + 1000*chain_id)
  sgnht = run_thermostats( problem, params, theta0, x0 )
  
  np.random.seed(init_seed + 1000*chain_id)
  sghmc = run_sghmc( problem, params, theta0, x0 )
  
  np.random.seed(init_seed + 1000*chain_id)
  sgld = run_sgld( problem, params, theta0, x0 )
  
  np.random.seed(init_seed + 1000*chain_id)
  mcmc = run_mcmc( problem, params, theta0, x0 )
  
  pp.figure(1)
  pp.clf()
  pp.plot( mcmc["THETA"][-1000:])
  pp.plot( sgld["THETA"][-1000:])
  pp.plot( sghmc["THETA"][-1000:])
  pp.plot( sgnht["THETA"][-1000:])
  pp.legend( ["MCMC","SGLD","SGHMC","SGNHT"])
  # view results of single chain
  #problem.view_single_chain( sgnht )
    
  pp.show()