import numpy as np
import scipy as sp
from scipy import stats as spstats
import pylab as pp
import pdb
from logistic_regression import *

# def get_omega(problem, batch_size):
#   get_omega.counter += 1
#   mini_batches = get_omega.mini_batches
#   if not mini_batches:
#     reconstruction_error, num_correct = problem.nn.test(problem.validation_set)
#     print "Reconstruction error: {}, average percentage of pixels correct: {}%".format(reconstruction_error, num_correct*100)
#     N = len(problem.training_set)
#     perms = np.random.permutation(N)
#     mini_batches += [perms[k:k+batch_size] for k in range(0, N, batch_size)]
#   if get_omega.counter == 10:
#     get_omega.counter = 0
#     reconstruction_error, num_correct = problem.nn.test(problem.validation_set)
#     print "Reconstruction error: {}, average percentage of pixels correct: {}%".format(reconstruction_error, num_correct*100)
#   return mini_batches.pop()
# get_omega.mini_batches = []
# get_omega.counter = 0

def get_omega(problem, batch_size):
  get_omega.counter += 1
  mini_batches = get_omega.mini_batches
  if not mini_batches:
    N = problem.N
    perms = np.random.permutation(N)
    mini_batches += [perms[k:k+batch_size] for k in range(0, N, batch_size)]
  if get_omega.counter % 10 == 0:
    LL, Y = problem.lr.loglikelihood( problem.lr.W, return_Y = True )
    Ytest, logYtest = softmax( np.dot( problem.lr.Xtest, problem.lr.W ), return_log = True )
    y_test = np.argmax(Ytest,1)
    if get_omega.average_counter == 0:
      get_omega.Ytest_avg = Ytest.copy()
    else:
      get_omega.Ytest_avg = get_omega.Ytest_avg + Ytest
      #get_omega.y_test_avg /= get_omega.average_counter+1

    get_omega.average_counter+=1
    get_omega.y_test_avg = np.argmax(get_omega.Ytest_avg/float(get_omega.average_counter),1)
    error = classification_error( problem.lr.t_test, y_test )
    avg_error = classification_error( problem.lr.t_test, get_omega.y_test_avg )
    print "LL: {}, Classification error: {}%   {}%   max abs W {}".format(LL, error*100,avg_error*100,np.mean( np.abs( problem.lr.W)))
    #print "avg abs W :", np.mean( np.abs( problem.lr.W))
    get_omega.test_errors.append(error)
    get_omega.avg_test_errors.append(avg_error)
    # get_omega.weights.append(problem.lr.W)
    get_omega.LLs.append(LL)

    data = {
      'test_errors': get_omega.test_errors,
      'avg_test_errors': get_omega.avg_test_errors,
      'weights': [w.tolist() for w in get_omega.weights],
      'LLs': [ll.tolist() for ll in get_omega.LLs]
    }

    file = open("sampling-true-posterior-mcmc.json", "w")
    json.dump(data, file)
    file.close()
    print 'saved to file'

  return mini_batches.pop()
get_omega.mini_batches = []
get_omega.counter = 1
get_omega.average_counter = 0
get_omega.y_test_avg = 0
get_omega.Ytest_avg = 0
get_omega.test_errors = []
get_omega.avg_test_errors = []
get_omega.weights = []
get_omega.LLs = []

def bounce_off_boundaries( theta, p, lower_bounds, upper_bounds ):
  D = len(theta)

  for d in range(D):
    if lower_bounds is not None:
      if theta[d] < lower_bounds[d]:
        theta[d] = lower_bounds[d] + (lower_bounds[d]-theta[d]) #from Neal Handbook, p37
        p[d] *= -1

    if upper_bounds is not None:
      if theta[d] > upper_bounds[d]:
        theta[d] = upper_bounds[d] - (theta[d]-upper_bounds[d]) #from Neal Handbook, p37
        p[d] *= -1
  return theta, p


def accept_move(log_acceptance):
  accept = False
  if log_acceptance < 0:
    if np.random.rand() < np.exp( log_acceptance ):
      # accept downward move
      accept=True
  else:
    # accept upward move
    accept = True
  return accept

def run_mcmc( problem, params, theta, x = None ):
  T             = params["T"]
  S             = params["S"]
  verbose_rate  = params["verbose_rate"]
  batch_size  = params["batch_size"]
  # keep theta within bounds
  lower_bounds = params["lower_bounds"]
  upper_bounds = params["upper_bounds"]
  propose_params = params['propose']

  keep_x        = True
  omega = get_omega(problem, batch_size)
  outputs = {}

  # pseudo-data
  if keep_x:
    if x is None:
      x = problem.simulate( theta, omega, S )
    assert len(x.shape) == 2, "x should be S by dim"
    loglike_x = problem.loglike_x( x, omega )
    X      = [x]
    LL     = [loglike_x]
  else:
    x         = None
    loglike_x = None

  p_dummy = np.random.randn(len(theta))

  THETAS = [theta]
  OMEGAS = [omega]
  num_accepts = 0
  # sample...
  for t in xrange(T):

    # -------------------- #
    # sample (theta, x)    #
    # -------------------- #
    theta_proposal = problem.propose( theta, params )
    # bounce position off parameter boundaries
    # theta_proposal, p_dummy = bounce_off_boundaries( theta_proposal, p_dummy, lower_bounds, upper_bounds )

    x_proposal     = problem.simulate( theta_proposal, omega, S )

    # using kernel_epsilon( observations | x )
    loglike_x_proposal = problem.loglike_x( x_proposal, omega )

    # a log-normal proposal, so we need to compute this
    # loglike_q_from_proposal_to_theta = problem.loglike_proposal_theta( theta, theta_proposal )
    # loglike_q_from_theta_to_proposal = problem.loglike_proposal_theta( theta_proposal, theta )
    loglike_q_from_proposal_to_theta = 0
    loglike_q_from_theta_to_proposal = 0

    # loglike_prior_theta
    loglike_prior_theta           = problem.loglike_prior( theta, params )
    loglike_prior_theta_proposal  = problem.loglike_prior( theta_proposal, params )

    log_acceptance =  np.sum(loglike_x_proposal) + np.sum(loglike_prior_theta_proposal) + loglike_q_from_proposal_to_theta \
                    - np.sum(loglike_x)          - np.sum(loglike_prior_theta)          - loglike_q_from_theta_to_proposal

    if accept_move(log_acceptance):
      num_accepts += 1
      x         = x_proposal
      theta     = theta_proposal
      loglike_x = loglike_x_proposal

    if t % 10 == 0:
      print "Acceptance percentage: {}%".format(num_accepts*100.0/(t+1))

    # --------------- #
    # samples omegas  #
    # --------------- #
    omega = get_omega(problem, batch_size)


    if keep_x:
      #x = problem.simulate( theta, omega, S )
      #loglike_x = problem.loglike_x( x )
      LL.append(loglike_x)
      X.append(x)

    THETAS.append(theta)
    OMEGAS.append(omega)


    if np.mod(t+1, verbose_rate) == 0:
      get_omega.weights.append(theta)
      print "t = %04d    loglik = %3.3f    theta = %3.3f    x = %3.3f"%(t+1,loglike_x,theta[0],x[0][0])

  outputs["THETA"] = np.squeeze(np.array( THETAS))
  outputs["OMEGA"] = np.squeeze(np.array(OMEGAS))

  if keep_x:
    outputs["X"] = np.squeeze(np.array(X))
    outputs["LL"] = np.squeeze(np.array(LL))
  return outputs

def run_sgld( problem, params, theta, x = None ):

  T             = params["T"]
  S             = params["S"]
  verbose_rate  = params["verbose_rate"]
  batch_size    = params["batch_size"]
  eta           = params["eta"] # Hamiltonain step size
  d_theta       = params["d_theta"] # step used for estimating gradients
  grad_U_func   = params["grad_func"]
  grad_U_params = params["grad_params"]

  keep_x        = params["keep_x"] # the HABC methods do not necessarily have x in the state space

  # keep theta within bounds
  lower_bounds = params["lower_bounds"]
  upper_bounds = params["upper_bounds"]

  D   = len(theta)

  outputs = {}
  # two_gradients = []
  # one_gradients = []
  # true_gradients = []

  omega = get_omega(problem, batch_size)

  # pseudo-data
  if keep_x:
    if x is None:
      x = problem.simulate( theta, omega, S )
    # (??) Any reason for this?
    # assert len(x.shape) == 2, "x should be S by dim"
    loglike_x = problem.loglike_x( x, omega )
    X      = [x]
    LL     = [loglike_x]
  else:
    x         = None
    loglike_x = None



  THETAS = [theta]
  OMEGAS = [omega]

  grads = np.zeros((T,D))

  # sample...
  for t in xrange(T):

    # ----------------------------- #
    # simulate trajectory for theta #
    # ----------------------------- #

    # estimate stochastic gradient
    grad_U = grad_U_func( theta, d_theta, omega, S, dict(grad_U_params.items() + params.items()))
    #
    if grad_U_params["record_2side_sl_grad"]:
      grad_U_dummy = problem.two_sided_sl_gradient( theta, d_theta, omega, S, grad_U_params )
    if grad_U_params["record_2side_keps_grad"]:
      grad_U_dummy = problem.two_sided_keps_gradient( theta, d_theta, omega, S, grad_U_params )
    if grad_U_params["record_true_abc_grad"]:
      grad_U_dummy = problem.true_abc_gradient( theta, d_theta, S, grad_U_params )
    if grad_U_params["record_true_grad"]:
      grad_U_dummy = problem.true_gradient( theta, grad_U_params )

    # print grad_U
    grads[t,:] = grad_U

    # initialize momentum
    p = np.random.randn(D)

    # take step momentum
    p = p - grad_U*eta/2.0

    # full step position
    theta = theta + p*eta

    # bounce position off parameter boundaries
    # theta, p = bounce_off_boundaries( theta, p, lower_bounds, upper_bounds )

    # --------------- #
    # samples omegas  #
    # --------------- #
    omega = get_omega(problem, batch_size)
    # d_theta *= 0.995
    # d_theta = max(d_theta, 1e-10)
    if keep_x:
      # probably too many simulations
      x = problem.simulate( theta, omega, S )
      loglike_x = problem.loglike_x( x, omega )
      LL.append(loglike_x)
      X.append(x)

    THETAS.append(theta)
    OMEGAS.append(omega)


    if np.mod(t+1,verbose_rate)==0:
      get_omega.weights.append(theta)
      if keep_x:
        print "t = %04d    loglik = %3.3f    theta0 = %3.3f    x0 = %3.3f"%(t+1,loglike_x,theta[0],x[0][0])
      else:
        print "t = %04d    theta0 = %3.3f"%(t+1,theta[0])



  outputs["THETA"] = np.squeeze(np.array( THETAS))
  outputs["OMEGA"] = np.squeeze(np.array(OMEGAS))
  outputs["GLOGS"] = grad_U_params["logs"]
  if keep_x:
    outputs["X"] = np.squeeze(np.array(X))
    outputs["LL"] = np.squeeze(np.array(LL))
  return outputs

def run_sghmc( problem, params, theta, x = None ):

  T             = params["T"]
  S             = params["S"]
  verbose_rate  = params["verbose_rate"]
  batch_size  = params["batch_size"]
  deltaC        = params["C"]  # injected noise added to Variance
  eta           = params["eta"] # Hamiltonain step size
  d_theta       = params["d_theta"] # step used for estimating gradients
  grad_U_func   = params["grad_func"]
  grad_U_params = params["grad_params"]

  keep_x        = params["keep_x"] # the HABC methods do not necessarily have x in the state space

  # keep theta within bounds
  lower_bounds = params["lower_bounds"]
  upper_bounds = params["upper_bounds"]

  D   = len(theta)

  outputs = {}
  # two_gradients = []
  # one_gradients = []
  # true_gradients = []

  omega = get_omega(problem, batch_size)

  # pseudo-data
  if keep_x:
    if x is None:
      x = problem.simulate( theta, omega, S )
    # assert len(x.shape) == 2, "x should be S by dim"
    loglike_x = problem.loglike_x( x, omega )
    X      = [x]
    LL     = [loglike_x]
  else:
    x         = None
    loglike_x = None

  # initialize momentum
  p = np.random.randn(D)

  THETAS = [theta]
  OMEGAS = [omega]

  grads = np.zeros((T,D))

  # sample...
  for t in xrange(T):
    # ----------------------------- #
    # simulate trajectory for theta #
    # ----------------------------- #

    # estimate stochastic gradient
    grad_U = grad_U_func( theta, d_theta, omega, S, grad_U_params)
    #grad_U = problem.one_sided_gradient( theta, x, omega, c )
    #true_grad = problem.true_abc_gradient( theta, true_gradients )
    #
    if grad_U_params["record_2side_sl_grad"]:
      grad_U_dummy = problem.two_sided_sl_gradient( theta, d_theta, omega, S, grad_U_params )
    if grad_U_params["record_2side_keps_grad"]:
      grad_U_dummy = problem.two_sided_keps_gradient( theta, d_theta, omega, S, grad_U_params )
    if grad_U_params["record_true_abc_grad"]:
      grad_U_dummy = problem.true_abc_gradient( theta, d_theta, S, grad_U_params )
    if grad_U_params["record_true_grad"]:
      grad_U_dummy = problem.true_gradient( theta, grad_U_params )

    grads[t,:] = grad_U
    V = np.var(grads[:t+1], axis=1)
    B = 0.5*eta*V
    C = B+deltaC
    # full step momentum
    p = p - C*p*eta - grad_U*eta + np.sqrt(2.0*(C-B)*eta)*np.random.randn(D)
    # full step position
    theta = theta + p*eta
    # bounce position off parameter boundaries
    # theta, p = bounce_off_boundaries( theta, p, lower_bounds, upper_bounds )

    # --------------- #
    # samples omegas  #
    # --------------- #
    omega = get_omega(problem, batch_size)
    if keep_x:
      x = problem.simulate( theta, omega, S )
      loglike_x = problem.loglike_x( x, omega )
      LL.append(loglike_x)
      X.append(x)

    THETAS.append(theta)
    OMEGAS.append(omega)


    if np.mod(t+1,verbose_rate)==0:
      get_omega.weights.append(theta)
      if keep_x:
        print "t = %04d    loglik = %3.3f    theta0 = %3.3f    x0 = %3.3f"%(t+1,loglike_x,theta[0],x[0])
      else:
        print "t = %04d    theta0 = %3.3f"%(t+1,theta[0])

  outputs["THETA"] = np.squeeze(np.array( THETAS))
  outputs["OMEGA"] = np.squeeze(np.array(OMEGAS))
  outputs["GLOGS"] = grad_U_params["logs"]
  if keep_x:
    outputs["X"] = np.squeeze(np.array(X))
    outputs["LL"] = np.squeeze(np.array(LL))
  return outputs

def run_thermostats( problem, params, theta, x = None ):

  T             = params["T"]
  S             = params["S"]
  verbose_rate  = params["verbose_rate"]
  batch_size  = params["batch_size"]
  C             = params["C"]  # injected noise
  eta           = params["eta"] # Hamiltonain step size
  d_theta       = params["d_theta"] # step used for estimating gradients
  grad_U_func   = params["grad_func"]
  grad_U_params = params["grad_params"]

  keep_x        = params["keep_x"] # the HABC methods do not necessarily have x in the state space

  # keep theta within bounds
  lower_bounds = params["lower_bounds"]
  upper_bounds = params["upper_bounds"]

  D   = len(theta)

  outputs = {}
  # two_gradients = []
  # one_gradients = []
  # true_gradients = []

  omega = get_omega(problem, batch_size)

  # pseudo-data
  if keep_x:
    if x is None:
      x = problem.simulate( theta, omega, S )
    # assert len(x.shape) == 2, "x should be S by dim"
    loglike_x = problem.loglike_x( x, omega )
    X      = [x]
    LL     = [loglike_x]
  else:
    x         = None
    loglike_x = None

  # initialize momentum
  p = np.random.randn(D)
  # initialize thermostat
  xi = C


  THETAS = [theta]
  OMEGAS = [omega]
  XI     = [xi]

  # sample...
  for t in xrange(T):
    # ----------------------------- #
    # simulate trajectory for theta #
    # ----------------------------- #

    # estimate stochastic gradient
    grad_U = grad_U_func( theta, d_theta, omega, S, grad_U_params)
    #grad_U = problem.one_sided_gradient( theta, x, omega, c )
    #true_grad = problem.true_abc_gradient( theta, true_gradients )
    #
    if grad_U_params["record_2side_sl_grad"]:
      grad_U_dummy = problem.two_sided_sl_gradient( theta, d_theta, omega, S, grad_U_params )
    if grad_U_params["record_2side_keps_grad"]:
      grad_U_dummy = problem.two_sided_keps_gradient( theta, d_theta, omega, S, grad_U_params )
    if grad_U_params["record_true_abc_grad"]:
      grad_U_dummy = problem.true_abc_gradient( theta, d_theta, S, grad_U_params )
    if grad_U_params["record_true_grad"]:
      grad_U_dummy = problem.true_gradient( theta, grad_U_params )

    # full step momentum
    p = p - xi*p*eta - grad_U*eta + np.sqrt(2.0*C*eta)*np.random.randn( D )
    # full step position
    theta = theta + p*eta

    # bounce position off parameter boundaries
    # theta, p = bounce_off_boundaries( theta, p, lower_bounds, upper_bounds )

    # update thermostat
    #xi = xi + eta*(p*p - 1.0)
    xi = xi + eta*(np.dot(p.T,p)/len(p) - 1.0)
    #print xi, p[0], grad_U[0]
    # d_theta *= 0.995
    # C *= 0.999
    # --------------- #
    # samples omegas  #
    # --------------- #
    omega = get_omega(problem, batch_size)

    if keep_x:
      x = problem.simulate( theta, omega, S )
      loglike_x = problem.loglike_x( x, omega )
      LL.append(loglike_x)
      X.append(x)

    THETAS.append(theta)
    OMEGAS.append(omega)
    XI     = [xi]


    if np.mod(t+1,verbose_rate)==0:
      get_omega.weights.append(theta)
      if keep_x:
        print "t = %04d    loglik = %3.3f    theta0 = %3.3f    x0 = %3.3f"%(t+1,loglike_x,theta[0],x[0][0])
      else:
        print "t = %04d    theta0 = %3.3f"%(t+1,theta[0])

  outputs["THETA"] = np.squeeze(np.array( THETAS))
  outputs["OMEGA"] = np.squeeze(np.array(OMEGAS))
  outputs["GLOGS"] = grad_U_params["logs"]
  outputs["XI"]    = np.squeeze(np.array(XI))
  if keep_x:
    outputs["X"] = np.squeeze(np.array(X))
    outputs["LL"] = np.squeeze(np.array(LL))
  return outputs
  #return np.squeeze(np.array( THETAS)), np.squeeze(np.array(X)), np.squeeze(np.array(LL)), np.squeeze(np.array(OMEGAS)), {"two_grads":np.array(two_gradients),"one_grads":np.array(one_gradients),"true_grads":np.array(true_gradients)}

