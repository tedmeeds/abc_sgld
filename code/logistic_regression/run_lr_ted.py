# TOOD: Adapt this for NN case
# Construct NN, load data
# Pass it to nn_wrapper

import numpy as np
import pylab as pp
import scipy as sp
from scipy import stats as spstats
from lr_wrapper_ted import *
from working_code_ted import *
from logistic_regression_ted import *
import json

keep_x        = False
init_seed     = 1
T             = 10000 # nbr of samples
verbose_rate  = 50
C             = 1.1    # injected noise variance parameter
#eta           = 1e-3 # step size for Hamiltoniam dynamics
eta           = 1e-4 # step size for Hamiltoniam dynamics
#h = 0.0005

# params for gradients
d_theta = 1e-2  # step size for gradient estimate
S       = 5
grad_params = {}

# keep theta within bounds
lower_bounds = np.array([0.001])
upper_bounds = np.array([np.inf])

if __name__ == "__main__":
  # pp.close('all')
  digits = [0,1]
  ndigits = len(digits)
  K = ndigits
  mnist = load_mnist(digits)
  (X_train,t_train),(X_valid,t_valid),(X_test,t_test) = mnist
  T_train = onehot( t_train, ndigits )
  T_valid = onehot( t_valid, ndigits )
  T_test  = onehot( t_test, ndigits )
  m = X_train.mean(0)
  s = X_train.std(0)
  ok = pp.find(s>0)
  X_train -= m
  X_valid -= m
  X_test  -= m
  X_train[:,ok] /= s[ok]
  X_valid[:,ok] /= s[ok]
  X_test[:,ok] /= s[ok]
  gamma = 1e-3
  q = 0.5
  std = np.sqrt(gamma)
  problem = LogisticRegressionProblem(\
                        MulticlassLogisticRegression(T_train, X_train, T_test, X_test, std), \
                        K, gamma, q )
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
  # params["batch_size"] = len(X_train)
  params["verbose_rate"] = verbose_rate
  params["grad_params"]  = {"logs":{"true":[],"true_abc":[],"2side_keps":[],"2side_sl":[]},\
                            "record_2side_sl_grad":False, "record_2side_keps_grad":False,"record_true_abc_grad":False,"record_true_grad":False,
                            "2side_keps": {"R": 10,"percent_to_change":0.0, 'prior_penalty': 1e-2}}
  params["lower_bounds"] = lower_bounds
  params["upper_bounds"] = upper_bounds
  params["keep_x"]       = keep_x
  params["propose"] = {'q': 0.5}
  params['prior'] = {'gamma': 1e-3}

  file = open('LR3.json', "r")
  data = json.load(file)
  file.close()
  W = np.array([np.array(w) for w in data["weights"][-1]])
  problem.lr.W = W
  problem.W_MAP = W.copy()
  problem.w_MAP = problem.lr.W.flatten()
  problem.random_proj = np.random.randn( len(problem.w_MAP),2 )

  theta0 = problem.w_MAP #
  theta0=problem.w_MAP #problem.lr.W.flatten()
  # C = [C]*len(theta0)
  x0     = None

  # initialize the simulator
  np.random.seed(init_seed)
  params["grad_func"] = problem.two_sided_keps_gradient

  # init randomly for MCMC chain
  # np.random.seed(init_seed + 1000*chain_id)
  print params
  # run algorithm
  #outs = run_sgld( problem, params, theta0, x0 )
  outs = run_thermostats( problem, params, theta0, x0 )
  
  rMAP = np.dot( problem.w_MAP, problem.random_proj )
  rtheta = np.dot( outs["THETA"], problem.random_proj )
  
  pp.figure()
  pp.plot( rtheta[:,0], rtheta[:,1], 'k-' )
  pp.plot( rtheta[:,0], rtheta[:,1], 'bo' )
  pp.plot( [rMAP[0]], [rMAP[1]], 'ro',ms=10 )
  pp.show()
  # outs = run_sghmc( problem, params, theta0, x0 )
  # outs = run_sgld( problem, params, theta0, x0 )
  # outs = run_mcmc( problem, params, theta0, x0 )

  # view results of single chain
  # problem.view_single_chain( outs )

  # pp.show()
