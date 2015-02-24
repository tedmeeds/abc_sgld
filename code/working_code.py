import numpy as np
import scipy as sp
from scipy import stats as spstats
import pylab as pp


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

def omega_flip(problem, theta, x, omega, loglike_x ):
  
  omega     = np.random.randint(10**6)
  x         = problem.simulate( theta, omega )
  loglike_x = problem.loglike_x( x )
    
  return theta, x, omega, loglike_x
    
def omega_sample(problem, theta, x, omega, loglike_x ):
  
  omega_proposal     = np.random.randint(10**6)
  x_proposal         = problem.simulate( theta, omega_proposal )
  loglike_x_proposal = problem.loglike_x( x_proposal )

  log_acceptance =  loglike_x_proposal  \
                  - loglike_x        
  
  if accept_move(log_acceptance):
    x         = x_proposal
    loglike_x = loglike_x_proposal
    omega     = omega_proposal
    
  return theta, x, omega, loglike_x
  
def run_mcmc( problem, params, theta, x = None ):
  T         = params["T"]
  use_omega = params["use_omega"]

  # random seed init
  if use_omega is False:
    omega = None
  else:
    omega = np.random.randint(10**6)
    
  
  # pseudo-data
  if x is None:
    x = problem.simulate( theta, omega )
  
  loglike_x = problem.loglike_x( x )
  
  X      = [x]
  THETAS = [theta]
  OMEGAS = [omega]
  LL     = [loglike_x]
  
  # sample...
  for t in xrange(T):
    
    # -------------------- #
    # sample (theta, x)    #
    # -------------------- #
    theta_proposal = problem.propose( theta )
    x_proposal     = problem.simulate( theta_proposal, omega )
    
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
    
    X.append(x)
    THETAS.append(theta)
    LL.append(loglike_x)
    
    # --------------- #
    # samples omegas  #
    # --------------- #
    if use_omega and np.random.rand() < omega_rate:
      theta, x, omega, loglike_x = omega_sample_procedure(problem, theta, x, omega, loglike_x )
    
    OMEGAS.append(omega)
    
    
    if np.mod(t+1,verbose_rate)==0:
      print "t = %04d    loglik = %3.3f    theta = %3.3f    x = %3.3f"%(t+1,loglike_x,theta,x)
  return np.squeeze(np.array( THETAS)), np.squeeze(np.array(X)), np.squeeze(np.array(LL)), np.squeeze(np.array(OMEGAS))
  
def run_thermostats( problem, params, theta, x = None ):
  
  assert len(x.shape) == 2, "x should be S by dim"
  T            = params["T"]
  verbose_rate = params["verbose_rate"]
  use_omega    = params["use_omega_across_time"]
  use_omega    = params["use_omega_at_single_time"]
  dt           = params["dt"]
  A            = params["A"]
  c            = params["c"]
  S            = params["S"]
  keep_x       = params["keep_x"] # the HABC methods do not necessarily have x in the state space
  
  # keep theta within bounds
  lower_bounds = params["lower_bounds"]
  upper_bounds = params["upper_bounds"]
  
  S,dim = x.shape
  
  two_gradients = []
  one_gradients = []
  true_gradients = []
  
  if use_omega is False:
    omega = None
  else:
    omega = np.random.randint(T)
  
  # pseudo-data
  if x is None:
    x = problem.simulate( theta, omega )
  
  
  loglike_x = problem.loglike_x( x )
  
  m = 1.0
  p = np.sqrt(m)*np.random.randn()
  xi = A
  
  X      = [x]
  THETAS = [theta]  
  OMEGAS = [omega]
  LL     = [loglike_x]
  
  # sample...
  for t in xrange(T):
    if use_omega is False:
      omega = t
    #grad_U = problem.one_sided_gradient( theta, x, omega, c )
    true_grad = problem.true_abc_gradient( theta, true_gradients )
    grad_U = problem.two_sided_sl_gradient( theta, x, omega, c, two_gradients, S )
    #grad_U = problem.two_sided_gradient( theta, x, omega, c, two_gradients, S )
    #grad_U = true_grad
    # divide by theta to transform var to 
    #grad_U *= theta
    #m = 0.9*m+0.1*grad_U**2
    p = p - xi*p*h - grad_U*h + np.sqrt(2.0*A*h)*np.random.randn()/np.sqrt(m)
    #theta = theta + p*h
    #log_theta = log_theta + p*h/np.sqrt(m)
    
    theta = theta + p*h
    
    while theta < min_theta:
      #pdb.set-trace()
      theta = min_theta + (min_theta-theta) #from Neal Handbook, p37
      p=-p
    
    
    #print "theta = ",theta," M= ",m, "grad_U = ",grad_U
    xi = xi + h*(p*p - 1.0)
    x = problem.simulate( theta, omega )
    
    #print grad_U, p, theta, xi
    loglike_x = problem.loglike_x( x )
    
    
    
    THETAS.append(theta)
    LL.append(loglike_x)
    
    # --------------- #
    # samples omegas  #
    # --------------- #
    if use_omega and np.random.rand()<0.01:
      #theta, x, omega, loglike_x = omega_sample(problem, theta, x, omega, loglike_x )
      #x_computed_at_omega = True
      omega = np.random.randint(10**6)
    
    X.append(x)
    OMEGAS.append(omega)
    
    
    if np.mod(t+1,verbose_rate)==0:
      print "t = %04d    loglik = %3.3f    theta = %3.3f    x = %3.3f"%(t+1,loglike_x,theta,x)
  return np.squeeze(np.array( THETAS)), np.squeeze(np.array(X)), np.squeeze(np.array(LL)), np.squeeze(np.array(OMEGAS)), {"two_grads":np.array(two_gradients),"one_grads":np.array(one_gradients),"true_grads":np.array(true_gradients)}

