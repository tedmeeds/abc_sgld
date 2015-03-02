# TOOD: Adapt this for NN case
# Construct NN, load data
# Pass it to nn_wrapper

import numpy as np
import pylab as pp
import scipy as sp
from scipy import stats as spstats
from lr_wrapper import *
from working_code import *
from logistic_regression import *
import json

keep_x        = False
init_seed     = 1
T             = 50000 # nbr of samples
verbose_rate  = 1000
C             = 5.01    # injected noise variance parameter
eta           = 1e-4 # step size for Hamiltoniam dynamics
#h = 0.0005

# params for gradients
d_theta = 1e-4  # step size for gradient estimate
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
  std = 0.0001
  problem = LogisticRegressionProblem(MulticlassLogisticRegression(T_train, X_train, T_test, X_test, std), K)
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
                            "2side_keps": {"R": 10,"percent_to_change":0.0, 'prior_penalty': 1e-2}}
  params["lower_bounds"] = lower_bounds
  params["upper_bounds"] = upper_bounds
  params["keep_x"]       = keep_x
  params["propose"] = {'q': 0.5}
  params['prior'] = {'gamma': 1e-3}

  file = open('LR.json', "r")
  data = json.load(file)
  file.close()
  W = np.array([np.array(w) for w in data["weights"][-1]])
  problem.lr.W = W

  theta0 = problem.lr.W.flatten()
  # C = [C]*len(theta0)
  x0     = None

  # initialize the simulator
  np.random.seed(init_seed)
  params["grad_func"] = problem.two_sided_keps_gradient

  # init randomly for MCMC chain
  # np.random.seed(init_seed + 1000*chain_id)
  print params
  # run algorithm
  # outs = run_thermostats( problem, params, theta0, x0 )
  # outs = run_sghmc( problem, params, theta0, x0 )
  outs = run_sgld( problem, params, theta0, x0 )
  # outs = run_mcmc( problem, params, theta0, x0 )

  # view results of single chain
  # problem.view_single_chain( outs )

  # pp.show()
