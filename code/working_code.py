import numpy as np
import scipy as sp
from scipy import stats as spstats
import pylab as pp
import pdb

max_int_for_omega = 10**8

def bin_errors_1d( bins, true_centered_probability, samples ):
  N = len(samples)
  cnts, bins = np.histogram( samples ,bins=bins)
  
  n = cnts.sum()
  
  probability = cnts / (float(N))
  
  missed = N - n
  
  double_error = np.sum(np.abs( true_centered_probability - probability )) + missed/float(N)
  
  #pdb.set_trace()
  return double_error/2.0

def cramer_vonMises_criterion( true_P, samples, bins ):
  # see wiki cramer_von_Mises
  N = len(samples)
  cnts, bins = np.histogram( samples ,bins=bins)
  
  n = cnts.sum()
  
  est_P = cnts / (float(N))
  
  # for samples outside bins, we assume true_P is zeros
  missed = N - n
  
  discrete_cvm_error = np.sum( ( true_P - est_P)**2) + (missed/float(N))**2
  
  #pdb.set_trace()
  return discrete_cvm_error
  
def bin_sq_errors_1d( bins, true_centered_probability, samples ):
  N = len(samples)
  cnts, bins = np.histogram( samples ,bins=bins)
  
  n = cnts.sum()
  
  probability = cnts / (float(N))
  
  missed = N - n
  
  double_error = np.sqrt( np.sum( ( true_centered_probability - probability)**2) + (missed/float(N))**2 )
  
  #pdb.set_trace()
  return double_error/2.0
  

        
def convergence_to_scalar( psi, psi_estimate, times ):
  nbrEstimate = len( psi_estimate )
  
  T     = len(times)
  means = np.zeros( T )
  vrs   = np.zeros( T )
  for time_id,t in zip(range(T),times):
    print time_id, t
    n = t
    if time_id <= nbrEstimate:
      mean_estimate = psi_estimate[:t+1].mean()
      var_estimate  = float(1.0/(t-1))*(psi - mean_estimate)**2
    
      means[time_id] = mean_estimate
      vrs[time_id]   = var_estimate
      
  means = means[:time_id+1]
  vrs = vrs[:time_id+1]  
  used_times = times[:time_id+1]  
  return used_times, means, vrs

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

def hamiltonian_accept( problem, theta, theta_proposal, x, x_proposal, p, p_proposal ):
  # using kernel_epsilon( observations | x )
  loglike_x          = problem.loglike_x( x )
  loglike_x_proposal = problem.loglike_x( x_proposal )
  
  # a log-normal proposal, so we need to compute this
  loglike_q_from_proposal_to_theta = -0.5*np.dot(p_proposal.T, p_proposal)
  loglike_q_from_theta_to_proposal = -0.5*np.dot(p.T,p)
  
  # loglike_prior_theta
  loglike_prior_theta           = problem.loglike_prior( theta )
  loglike_prior_theta_proposal  = problem.loglike_prior( theta_proposal )
  
  log_acceptance =  loglike_x_proposal + loglike_prior_theta_proposal + loglike_q_from_proposal_to_theta \
                  - loglike_x          - loglike_prior_theta          - loglike_q_from_theta_to_proposal
                  
  if accept_move(log_acceptance):
    x         = x_proposal
    theta     = theta_proposal
    loglike_x = loglike_x_proposal
    p         = p_proposal
    
  return theta, x, loglike_x 

def init_omega( omega_params, S ):
  if omega_params["use_omega"] is False:
    omega = None
  else:
    omega = np.zeros(S,dtype=int)
    for s in range(S):
      omega[s] = np.random.randint(max_int_for_omega)
  return omega
  
def omega_switch(problem, theta, x, omega, loglike_x, omega_params ):
  any_changes = False
  for s in range(len(omega)):
    if np.random.rand() < omega_params["omega_rate"]:
      omega[s]     = np.random.randint(max_int_for_omega)
      any_changes = True
  
  if any_changes:
    x         = problem.simulate( theta, omega, len(omega) )
    loglike_x = problem.loglike_x( x )
    
  return theta, x, omega, loglike_x
    
def omega_sample(problem, theta, x, omega, loglike_x ):
  
  omega_proposal     = np.random.randint(max_int_for_omega)
  x_proposal         = problem.simulate( theta, omega_proposal, len(omega) )
  loglike_x_proposal = problem.loglike_x( x_proposal )

  log_acceptance =  loglike_x_proposal  \
                  - loglike_x        
  
  if accept_move(log_acceptance):
    x         = x_proposal
    loglike_x = loglike_x_proposal
    omega     = omega_proposal
    
  return theta, x, omega, loglike_x
  
def run_mcmc( problem, params, theta, x = None ):
  T             = params["T"]
  S             = params["S"]
  verbose_rate  = params["verbose_rate"]
  omega_params  = params["omega_params"]
  # keep theta within bounds
  lower_bounds = params["lower_bounds"]
  upper_bounds = params["upper_bounds"]
  
  keep_x        = True
  omega = init_omega( omega_params, S )
  outputs = {}
  
  # pseudo-data
  if keep_x:
    if x is None:
      x = problem.simulate( theta, omega, S )
    assert len(x.shape) == 2, "x should be S by dim"
    loglike_x = problem.loglike_x( x )
    X      = [x.mean(1).reshape( (1,S))]
    LL     = [loglike_x]
  else:
    x         = None
    loglike_x = None
  
  p_dummy = np.random.randn(len(theta))
  
  THETAS = [theta]
  OMEGAS = [omega]
  
  # sample...
  for t in xrange(T):
    
    # -------------------- #
    # sample (theta, x)    #
    # -------------------- #
    theta_proposal = problem.propose( theta )
    # bounce position off parameter boundaries
    theta_proposal, p_dummy = bounce_off_boundaries( theta_proposal, p_dummy, lower_bounds, upper_bounds )
    
    x_proposal     = problem.simulate( theta_proposal, omega, S )
    
    # using kernel_epsilon( observations | x )
    loglike_x_proposal = problem.loglike_x( x_proposal )
    
    # a log-normal proposal, so we need to compute this
    loglike_q_from_proposal_to_theta = problem.loglike_proposal_theta( theta, theta_proposal )
    loglike_q_from_theta_to_proposal = problem.loglike_proposal_theta( theta_proposal, theta )
    
    # loglike_prior_theta
    loglike_prior_theta           = problem.loglike_prior( theta )
    loglike_prior_theta_proposal  = problem.loglike_prior( theta_proposal )
    
    log_acceptance =  loglike_x_proposal + loglike_prior_theta_proposal + loglike_q_from_proposal_to_theta \
                    - loglike_x          - loglike_prior_theta          - loglike_q_from_theta_to_proposal
                    
    if accept_move(log_acceptance):
      x         = x_proposal
      theta     = theta_proposal
      loglike_x = loglike_x_proposal
    
    
    
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
      #x = problem.simulate( theta, omega, S )
      #loglike_x = problem.loglike_x( x )
      LL.append(loglike_x)
      X.append(x.mean(1).reshape( (1,S)))
    
    THETAS.append(theta)
    OMEGAS.append(omega)
    
    
    if np.mod(t+1,verbose_rate)==0:
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
  omega_params  = params["omega_params"]
  eta           = params["eta"] # Hamiltonain step size
  d_theta       = params["d_theta"] # step used for estimating gradients
  grad_U_func   = params["grad_func"]
  grad_U_params = params["grad_params"]
  mh_correction = params["mh_correction"]
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
    
  
  
  THETAS = [theta]  
  OMEGAS = [omega]
  
  grads = np.zeros( (T,D) )
  
  # sample...
  for t in xrange(T):
    
    # ----------------------------- # 
    # simulate trajectory for theta #
    # ----------------------------- #
    
    # estimate stochastic gradient
    grad_U = grad_U_func( theta, d_theta, omega, S, grad_U_params)
    #
    # if grad_U_params["record_2side_sl_grad"]:
    #   grad_U_dummy = problem.two_sided_sl_gradient( theta, d_theta, omega, S, grad_U_params )
    # if grad_U_params["record_2side_keps_grad"]:
    #   grad_U_dummy = problem.two_sided_keps_gradient( theta, d_theta, omega, S, grad_U_params )
    # if grad_U_params["record_true_abc_grad"]:
    #   grad_U_dummy = problem.true_abc_gradient( theta, d_theta, S, grad_U_params )
    # if grad_U_params["record_true_grad"]:
    #   grad_U_dummy = problem.true_gradient( theta, grad_U_params )
    
    grads[t,:] = grad_U
    
    # initialize momentum  
    p = np.random.randn(D)
    current_p = p
    
    # take step momentum
    p = p - grad_U*eta/2.0
    
    current_theta = theta
    
    # full step position
    theta = theta + p*eta
    
    # bounce position off parameter boundaries
    theta, p = bounce_off_boundaries( theta, p, lower_bounds, upper_bounds )
    
    if mh_correction:
      x_proposal     = problem.simulate( theta, omega, S )
      
      theta, x, loglike_x = hamiltonian_accept( problem, current_theta, theta, x, x_proposal, current_p, p )
      
        
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
      # probably too many simulations
      if mh_correction is False:
        x = problem.simulate( theta, omega, S )
        loglike_x = problem.loglike_x( x )
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
  
  grads = np.zeros( (T,D) )
  
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
    
    V = np.var(grads[:t+1],0)*np.eye(D)
    B = 0.5*eta*V
    C = B+deltaC*np.eye(D)
    
    # full step momentum
    p = p - C*p*eta - grad_U*eta + np.sqrt(2.0*(C-B)*eta)*np.random.randn(D)
    
    # full step position
    theta = theta + p*eta
    
    # bounce position off parameter boundaries
    theta, p = bounce_off_boundaries( theta, p, lower_bounds, upper_bounds )
    
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
  omega_params  = params["omega_params"]
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
    theta, p = bounce_off_boundaries( theta, p, lower_bounds, upper_bounds )
    
    # update thermostat
    xi = xi + eta*( np.dot(p.T,p)/D - 1.0)
    
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

