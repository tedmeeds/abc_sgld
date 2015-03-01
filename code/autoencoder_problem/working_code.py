import numpy as np
import scipy as sp
from scipy import stats as spstats
import pylab as pp
import pdb

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
  if get_omega.counter % 50 == 0:
    reconstruction_error, num_correct = problem.nn.test(problem.X_valid.T,problem.X_valid.T)
    print "Reconstruction error: {}, average percentage of pixels correct: {}%".format(reconstruction_error, num_correct*100)
    problem.nn.save('thermo-saved.json'.format())
  return mini_batches.pop()
get_omega.mini_batches = []
get_omega.counter = 0

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

  grads = np.array([])

  # sample...
  for t in xrange(T):

    # ----------------------------- #
    # simulate trajectory for theta #
    # ----------------------------- #

    # estimate stochastic gradient
    grad_U = grad_U_func( theta, d_theta, omega, S, grad_U_params)
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
    grads = np.append(grads, grad_U)

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

    if keep_x:
      # probably too many simulations
      x = problem.simulate( theta, omega, S )
      loglike_x = problem.loglike_x( x, omega )
      LL.append(loglike_x)
      X.append(x)

    THETAS.append(theta)
    OMEGAS.append(omega)


    if np.mod(t+1,verbose_rate)==0:
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
  omega_params  = params["omega_params"]
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

  omega = init_omega( omega_params, S )

  # pseudo-data
  if keep_x:
    if x is None:
      x = problem.simulate( theta, omega, S )
    assert len(x.shape) == 2, "x should be S by dim"
    loglike_x = problem.loglike_x( x )
    X      = [x]
    LL     = [loglike_x]
  else:
    x         = None
    loglike_x = None

  # initialize momentum
  p = np.random.randn(D)

  THETAS = [theta]
  OMEGAS = [omega]

  grads = np.array([])

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

    grads = np.append(grads, grad_U)

    V = np.var(grads[:t+1],0)*np.eye(D)
    B = 0.5*eta*V
    C = B+deltaC*np.eye(D)

    # full step momentum
    p = p - C*p*eta - grad_U*eta + np.sqrt(2.0*(C-B)*eta)*np.random.randn(D)

    # full step position
    theta = theta + p*eta

    # bounce position off parameter boundaries
    # theta, p = bounce_off_boundaries( theta, p, lower_bounds, upper_bounds )

    # --------------- #
    # samples omegas  #
    # --------------- #
    if omega_params["use_omega"]:
      if omega_params["omega_sample"]:
        # propose new omega and accept/reject using MH
        theta, x, omega, loglike_x = omega_sample(problem, theta, x, omega, loglike_x )
      elif omega_params["omega_switch"]:
        # randomly switch to new omega
        theta, x, omega, loglike_x = omega_switch(problem, theta, x, omega, loglike_x, omega_params )

    if keep_x:
      x = problem.simulate( theta, omega, S )
      loglike_x = problem.loglike_x( x )
      LL.append(loglike_x)
      X.append(x)

    THETAS.append(theta)
    OMEGAS.append(omega)


    if np.mod(t+1,verbose_rate)==0:
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
    xi = xi + eta*(p*p - 1.0)

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

