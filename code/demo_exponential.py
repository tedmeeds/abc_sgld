from abcpy.factories import *
from abcpy.problems.exponential import *
from sa_algorithms import *
from scipy import stats as spstats
import pylab as pp

problem_params = default_params()
problem_params["epsilon"] = 1.0
problem_params["N"] = 5
problem_params["q_stddev"] = 0.5
problem_params["theta_star"]      = 0.1
#problem_params["alpha"]           = 1.0
#problem_params["beta"]            = 1.0
problem_params["alpha"]           = 3.75
problem_params["beta"]            = 1.01

exp_problem = ExponentialProblem( problem_params, force_init = True )

state_params = state_params_factory.scrape_params_from_problem( exp_problem, S=1 )

# set is_marginal to true so using only "current" state will force a re-run
mcmc_params  = mcmc_params_factory.scrape_params_from_problem( exp_problem, type="mh", is_marginal = True, nbr_samples = 1 )

# ignore algo, just doing this to get state object
algo_params = { "modeling_approach"  : "kernel",
                "observation_groups" : exp_problem.get_obs_groups(),
                "state_params"       : state_params,
                "mcmc_params"        : mcmc_params,
                "algorithm"          : "model_mcmc"
              }
algo, model, state  = algo_factory.create_algo_and_state( algo_params )
recorder     = recorder_factory.create_recorder( {} )


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
  loglike_q_from_proposal_to_theta = -0.5*p_proposal*p_proposal
  loglike_q_from_theta_to_proposal = -0.5*p*p
  
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
  
def run_sgld( problem, T, h, A, c, theta=None, x = None, verbose_rate = 100, use_omega = False, S = 20 ):
  two_gradients = []
  one_gradients = []
  true_gradients = []
  
  if theta is None:
    theta = problem.prior_rand()
    
  # parameters
  THETAS = [theta]
  log_theta = np.log(theta)
  
  if use_omega is False:
    omega = 0
  else:
    omega = np.random.randint(T)
    
  alpha = 0.0
  OMEGAS = [omega]
  
  # pseudo-data
  x_computed_at_omega = False
  if x is None:
    x_computed_at_omega = True
    x = problem.simulate( theta, omega )
  X = [x]
  
  loglike_x = problem.loglike_x( x )
  
  #p = np.random.randn()
  xi = A
  
  LL = [loglike_x]
  min_theta = 0.01
  # sample...
  for t in xrange(T):
    
    if use_omega is False:
      # if not using omega, then need new x 
      omega = np.random.randint( 10**6)
    x = problem.simulate( theta, omega )
      
    p = np.random.randn()
    current_p = p
      
    true_grad = problem.true_abc_gradient( theta, true_gradients )
    
    two_side_grad_U = problem.two_sided_gradient( theta, x, omega, c, two_gradients, S )
    #one_side_grad_U = problem.one_sided_gradient( theta, x, omega, c, one_gradients )
    
    # divide by theta to transform var to 
    grad_U = two_side_grad_U#*theta
    #grad_U = one_side_grad_U#*theta
    #grad_U = true_grad
    p=current_p-grad_U*h/2.0
    
    theta_proposal = theta + p*h
    
    while theta_proposal < min_theta:
      print "rebounding from ",theta_proposal
      theta_proposal = min_theta + (min_theta-theta_proposal) #from Neal Handbook, p37
      p=-p
      print "to ",theta_proposal
    
    
    x_proposal = problem.simulate( theta_proposal, omega )
    x_computed_at_omega = True
    
    
    
    loglike_x = problem.loglike_x( x_proposal )
    
    if False:
      theta, x, loglike_x = hamiltonian_accept( problem, theta, theta_proposal, x, x_proposal, current_p, p )
    else:
      x=x_proposal
      theta=theta_proposal
    
    # --------------- #
    # samples omegas  #
    # --------------- #
    if use_omega and np.random.rand()<0.01:
      #theta, x, omega, loglike_x = omega_sample(problem, theta, x, omega, loglike_x )
      #x_computed_at_omega = True
      omega = np.random.randint(10**6)
     
    #h *= 0.9999

    X.append(x)
    THETAS.append(theta)
    LL.append(loglike_x)
    
    OMEGAS.append(omega)
    
    
    if np.mod(t+1,verbose_rate)==0:
      print "t = %04d    loglik = %3.3f    theta = %3.3f    x = %3.3f"%(t+1,loglike_x,theta,x)
      #print "   grads: True=  %3.3f    mom-2 = %3.3f    2-sided = %3.3f"%(true_grad,m_grad_U,two_side_grad_U)
  return np.squeeze(np.array( THETAS)), np.squeeze(np.array(X)), np.squeeze(np.array(LL)), np.squeeze(np.array(OMEGAS)), {"two_grads":np.array(two_gradients),"one_grads":np.array(one_gradients),"true_grads":np.array(true_gradients)}

def run_sghmc( problem, T, h, A, c, theta=None, x = None, verbose_rate = 100, use_omega = False, S=10 ):
  two_gradients = []
  one_gradients = []
  true_gradients = []
  
  max_theta=10.0
  min_theta=0.01
  if theta is None:
    theta = problem.prior_rand()
    
  # parameters
  THETAS = [theta]
  log_theta = np.log(theta)
  
  if use_omega is False:
    omega = None
  else:
    omega = np.random.randint(T)
    
  OMEGAS = [omega]
  # pseudo-data
  if x is None:
    x = problem.simulate( theta, omega )
  X = [x]
  
  loglike_x = problem.loglike_x( x )
  
  m = 1.0
  p = np.sqrt(m)*np.random.randn()
  
  deltaA=A
  B=0
  
  LL = [loglike_x]
  V = 0
  M=0
  grads = np.zeros(T)
  # sample...
  for t in xrange(T):
    if use_omega is False:
      omega = t
    #grad_U = problem.one_sided_gradient( theta, x, omega, c )
    true_grad = problem.true_abc_gradient( theta, true_gradients )
    grad_U = problem.two_sided_gradient( theta, x, omega, c, two_gradients, S )
    #grad_U = true_grad
    
    grads[t] = grad_U
    M += grad_U
    V = np.var(grads[:t+1])
    B = 0.5*h*V
    A=B+deltaA
    theta = theta + p*h
    
    p = p - A*p*h - grad_U*h + np.sqrt(2.0*(A-B)*h)*np.random.randn()/np.sqrt(m)
    
    while theta < min_theta:
      #pdb.set-trace()
      theta = min_theta + (min_theta-theta) #from Neal Handbook, p37
      p=-p
    
    x = problem.simulate( theta, omega )
    
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
      print "   ",V,B
  return np.squeeze(np.array( THETAS)), np.squeeze(np.array(X)), np.squeeze(np.array(LL)), np.squeeze(np.array(OMEGAS)), {"two_grads":np.array(two_gradients),"one_grads":np.array(one_gradients),"true_grads":np.array(true_gradients)}
  
def run_thermostats( problem, T, h, A, c, theta=None, x = None, verbose_rate = 100, use_omega = False, S=20 ):
  two_gradients = []
  one_gradients = []
  true_gradients = []
  
  max_theta=10.0
  min_theta=0.01
  if theta is None:
    theta = problem.prior_rand()
    
  # parameters
  THETAS = [theta]
  log_theta = np.log(theta)
  
  if use_omega is False:
    omega = None
  else:
    omega = np.random.randint(T)
    
  OMEGAS = [omega]
  # pseudo-data
  if x is None:
    x = problem.simulate( theta, omega )
  X = [x]
  
  loglike_x = problem.loglike_x( x )
  
  m = 1.0
  p = np.sqrt(m)*np.random.randn()
  xi = A
  
  LL = [loglike_x]
  
  # sample...
  for t in xrange(T):
    if use_omega is False:
      omega = t
    #grad_U = problem.one_sided_gradient( theta, x, omega, c )
    true_grad = problem.true_abc_gradient( theta, true_gradients )
    grad_U = problem.two_sided_gradient( theta, x, omega, c, two_gradients, S )
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
  
def run_mcmc( problem, T, theta=None, x = None, verbose_rate = 100, use_omega = False ):
  if theta is None:
    theta = problem.prior_rand()
    
  # parameters
  THETAS = [theta]
  
  if use_omega is False:
    omega = None
  else:
    omega = np.random.randint(T)
    
  OMEGAS = [omega]
  # pseudo-data
  if x is None:
    x = problem.simulate( theta, omega )
  X = [x]
  
  loglike_x = problem.loglike_x( x )
  
  LL = [loglike_x]
  
  # sample...
  for t in xrange(T):
    
    # -------------------- #
    # sample theta first   #
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
    if use_omega and np.random.rand()<0.5:
      theta, x, omega, loglike_x = omega_sample(problem, theta, x, omega, loglike_x )
    
    OMEGAS.append(omega)
    
    
    if np.mod(t+1,verbose_rate)==0:
      print "t = %04d    loglik = %3.3f    theta = %3.3f    x = %3.3f"%(t+1,loglike_x,theta,x)
  return np.squeeze(np.array( THETAS)), np.squeeze(np.array(X)), np.squeeze(np.array(LL)), np.squeeze(np.array(OMEGAS))

def run_true_mcmc( problem, T, theta=None, x = None, verbose_rate = 100, use_omega = False ):
  if theta is None:
    theta = problem.prior_rand()
    
  # parameters
  THETAS = [theta]
  
  if use_omega is False:
    omega = None
  else:
    omega = np.random.randint(T)
    
  OMEGAS = [omega]
  # pseudo-data
  if x is None:
    x = problem.simulate( theta, omega )
  X = [x]
  
  loglike_x = problem.loglike_x( x )
  
  LL = [loglike_x]
  
  # sample...
  for t in xrange(T):
    
    # -------------------- #
    # sample theta first   #
    # -------------------- #
    theta_proposal = problem.propose( theta )
    x_proposal     = problem.simulate( theta_proposal, omega )
    
    # using kernel_epsilon( observations | x )
    loglike_x_proposal = problem.loglike_x( x_proposal )
    
    # a log-normal proposal, so we need to compute this
    loglike_q_from_proposal_to_theta = problem.loglike_proposal_theta( theta, theta_proposal )
    loglike_q_from_theta_to_proposal = problem.loglike_proposal_theta( theta_proposal, theta )
    
    # loglike_prior_theta
    loglike_prior_theta           = problem.loglike_posterior( theta )
    loglike_prior_theta_proposal  = problem.loglike_posterior( theta_proposal )
    
    log_acceptance =  0 + loglike_prior_theta_proposal + loglike_q_from_proposal_to_theta \
                    - 0          - loglike_prior_theta          - loglike_q_from_theta_to_proposal
              
    print theta, "  --->  ", theta_proposal, "log_acc = ", log_acceptance
    accept = False
    if log_acceptance < 0:
      if np.random.rand() < np.exp( log_acceptance ):
        # accept downward move
        accept=True
    else:
      # accept upward move
      accept = True
      
    # for thetas
    if accept:
      x         = x_proposal
      theta     = theta_proposal
      loglike_x = loglike_x_proposal
    
    X.append(x)
    THETAS.append(theta)
    LL.append(loglike_x)
    
    # --------------- #
    # samples omegas  #
    # --------------- #
    if use_omega and np.random.rand()<1.1:
      omega_proposal = np.random.randint(T)
      x_proposal     = problem.simulate( theta, omega_proposal )
      loglike_x_proposal = problem.loglike_x( x_proposal )
      
    
      #log_acceptance =  -0.5*( (x_proposal-x)**2 /(problem.x_epsilon**2) )
      log_acceptance =  0 #-0.5*( (x_proposal-x)**2 /(problem.x_epsilon**2) )
                    
      accept = False
      if log_acceptance < 0:
        if np.random.rand() < np.exp( log_acceptance ):
          # accept downward move
          accept=True
      else:
        # accept upward move
        accept = True
      
      # for thetas
      if accept:
        x         = x_proposal
        loglike_x = loglike_x_proposal
        omega     = omega_proposal
    
    
    OMEGAS.append(omega)
    
    
    if np.mod(t+1,verbose_rate)==0:
      print "t = %04d    loglik = %3.3f    theta = %3.3f    x = %3.3f"%(t+1,loglike_x,theta,x)
  return np.squeeze(np.array( THETAS)), np.squeeze(np.array(X)), np.squeeze(np.array(LL)), np.squeeze(np.array(OMEGAS))


class generate_exponential( object ):
  def __init__( self, abcpy_problem, x_epsilon ):
    self.p = abcpy_problem
    self.kernel = spstats.norm( self.p.obs_statistics, self.p.epsilon )
    self.y = self.p.obs_statistics[0]
    self.x_epsilon = x_epsilon
    
    self.posterior = spstats.gamma( self.p.alpha+self.p.N, 0, 1.0/(self.p.beta+self.p.obs_sum ))
    
  def prior_rand(self):
    return self.p.theta_prior_rand()
    
  def propose( self, theta ):
    return self.p.theta_proposal_rand( theta )
  
  def loglike_x( self, x ):
    return self.kernel.logpdf( x )
    
  def loglike_prior( self, theta ):
    return self.p.theta_prior_logpdf( theta )

  def loglike_posterior( self, theta ):
    return self.posterior.logpdf( theta )
    
  def loglike_proposal_theta( self, to_theta, from_theta ):
    return self.p.theta_proposal_logpdf( to_theta, from_theta )
    
  def simulate( self, theta, seed = None, state = None ):
    if seed is not None:
      # save current state
      current_state = np.random.get_state()
      np.random.seed(seed)
      #print "saving state and setting seed to ", seed
    if state is not None:
      current_state = np.random.get_state()
      np.random.set_state( state )
      print "saving state and setting state to ", state[-1]
      
    raw_outputs = self.p.simulation_function( theta )
    x = self.p.statistics_function( raw_outputs)
    
    if seed is not None:
      # put back the current state
      np.random.set_state( current_state )
      #print "putting back state state "
    
    if state is not None:
      np.random.set_state( current_state )
        
    return x
  
  def true_gradient( self, theta, gradients ):
    
    #self.alpha+self.N, self.beta+self.obs_sum
    
    #grad = (self.p.alpha+self.p.N-1)*np.log(theta) - (self.p.beta+self.p.obs_sum)*theta
    grad = (self.p.alpha+self.p.N-1)/theta - (self.p.beta+self.p.obs_sum)
    
    gradients.append( np.squeeze( np.array([grad,theta])) )
    return -grad 

  def true_abc_gradient( self, theta, gradients, S = 1 ):
    theta=theta[0]
    x = 1.0/theta
    f = self.loglike_x(x)
    #theta_minus = max(0.01,theta-c)
    #theta_plus = theta_minus + 2*c
    cs = [0.001,0.01,0.1]
    
    plusXs = []
    minusXs = []
    for s in range(S):
      #state = np.random.get_state()
      c = cs[np.random.randint(len(cs))]
      c = 0.001 + (0.01-0.001)*np.random.rand()
      #c = x/100.0
      c=0.001
      theta_minus = max(0.01,theta-c)
      theta_plus = theta_minus + 2*c
      
      x_plus  = 1.0/theta_plus
      x_minus = 1.0/theta_minus
      
      plusXs.append( x_plus )
      minusXs.append( x_minus )
    plusXs=np.squeeze(np.array(plusXs))
    minusXs=np.squeeze(np.array(minusXs))
    x_plus  = plusXs.mean()
    x_minus = minusXs.mean()
    
    f_plus = logsumexp( self.loglike_x(plusXs) ) - np.log(S)
    f_minus = logsumexp( self.loglike_x(minusXs) ) - np.log(S)
    #pdb.set_trace()
    #
    #grad1 = exact_grad*(x_plus-x_minus)/(theta_plus-theta_minus) #+ (self.p.alpha-1)/theta - self.p.beta
    grad = (f_plus-f_minus)/(theta_plus-theta_minus) + ( (self.p.alpha-1)/max(0.01,theta) - self.p.beta )
    
    #grad = (self.p.alpha+self.p.N-1)/theta - (self.p.beta+self.p.obs_sum)
    # if grad < 0:
    #   grad = max(np.array([-15]),grad)
    # else:
    #   grad = min(np.array([15]),grad)
    gradients.append( np.squeeze( np.array([grad, theta, x,f, theta_plus, x_plus,f_plus,theta_minus, x_minus,f_minus])) )
    return -grad 
        
  def one_sided_gradient( self, theta, x, omega, c, gradients ):
    exact_grad = ( self.y-x )/self.p.epsilon**2
    
    #log_theta = np.log(theta)
    
    f = self.loglike_x(x)
    state = np.random.get_state()
    
    theta_plus = theta+c #np.exp( log_theta + c )
    
    x_plus = self.simulate( theta_plus, omega )
    
    f_plus = self.loglike_x(x_plus)
    
    grad = (f_plus-f)/c + (self.p.alpha-1)*np.log(theta) - self.p.beta*theta
    
    if grad < 0:
      grad = max(np.array([-15]),grad)
    else:
      grad = min(np.array([15]),grad)
    #grad = exact_grad*(x_plus-x)/(theta_plus-theta) + (self.p.alpha-1)/theta - self.p.beta
    
    #print "one-sided: ", theta_plus, x_plus, theta, x, "exact ", exact_grad, "grad_x  ", (x_plus-x)/(theta_plus-theta), "grad_prior: ", (self.p.alpha-1)*np.log(theta) - self.p.beta*theta
    
    #grad = exact_grad*(x_plus-x_minus)/(theta_plus-theta_minus)+ (self.p.alpha-1)*np.log(theta) - self.p.beta*theta
    #return -grad
    gradients.append( np.squeeze( np.array([grad, theta, x, f, theta_plus, x_plus, f_plus])) )
    return -grad

  def one_sided_gradient_old( self, theta, x, omega, c ):
    f = self.loglike_x(x)
    
    x_plus = self.simulate( theta+c, omega )
    
    f_plus = self.loglike_x(x_plus)
    
    grad = (f_plus-f)/c + (self.p.alpha-1)/theta - self.p.beta
    
    return -grad
        
  def two_sided_gradient( self, theta, x, omega, c, gradients, S):
    #print "S ",S
    exact_grad = ( self.y-x )/self.p.epsilon**2
    log_theta = np.log(theta)
    f = self.loglike_x(x)
    state = np.random.get_state()
    
     #np.exp( log_theta + c )
    theta_minus = max(0.01,theta-c)
    theta_plus = theta_minus + 2*c
    cs = [0.001,0.01,0.1]
    c0=c
    plusXs = []
    minusXs = []
    for s in range(S):
      state = np.random.get_state()
      c = cs[np.random.randint(len(cs))]
      c = 0.01 #+ (0.01-0.005)*np.random.rand()
      #c = x/100.0
      theta_minus = max(0.01,theta-c)
      theta_plus = theta_minus + 2*c
      
      x_plus = self.simulate( theta_plus, seed=omega+s ) #omega+s+1000 )
      x_minus = self.simulate( theta_minus, seed=omega+s ) #omega+s+1000 )
      np.random.randn()
      
      plusXs.append( x_plus )
      minusXs.append( x_minus )
    plusXs=np.squeeze(np.array(plusXs))
    minusXs=np.squeeze(np.array(minusXs))
    x_plus  = plusXs.mean()
    x_minus = minusXs.mean()
    
    #print plusXs, minusXs, plusXs- minusXs
    f_plus_old = self.loglike_x(x_plus)
    f_minus_old = self.loglike_x(x_minus)
    
    
    
    #f_plus2 = np.sum( self.loglike_x(plusXs) ) - np.log(S)
    #f_minus2 = np.sum( self.loglike_x(minusXs) ) - np.log(S)
    
    f_plus = logsumexp( self.loglike_x(plusXs) ) - np.log(S)
    f_minus = logsumexp( self.loglike_x(minusXs) ) - np.log(S)
    #pdb.set_trace()
    #
    #grad = exact_grad*(x_plus-x_minus)/(theta_plus-theta_minus) + (self.p.alpha-1)/theta - self.p.beta
    grad = (f_plus-f_minus)/(theta_plus-theta_minus) + (self.p.alpha-1)/max(0.01,theta) - self.p.beta
    #grad1=grad
    # if grad < 0:
    #   grad = max(np.array([-50]),grad)
    # else:
    #   grad = min(np.array([50]),grad)
    #print "before ",grad
    #pdb.set_trace()
    # gg=0.5
    # if grad < 0:
    #   grad = -np.abs(grad)**gg
    # else:
    #   grad = grad**gg
    #grad = np.array([grad[0]**0.5])
    #print "after ",grad
    #print f_plus,f_minus, theta_plus,theta_minus,c
    # grad2 = (f_plus_old-f_minus_old)/(theta_plus-theta_minus) + (self.p.alpha-1)/theta - self.p.beta
    #grad = exact_grad*(x_plus-x_minus)/(theta_plus-theta_minus) + (self.p.alpha-1)/theta - self.p.beta
#     grad4 = exact_grad*(plusXs-minusXs)/(theta_plus-theta_minus) + (self.p.alpha-1)/theta - self.p.beta
#     #pdb.set_trace()
#     #print "two-sided: ", theta_plus, x_plus, theta_minus, x_minus, "exact ", exact_grad, "grad_x  ", (x_plus-x_minus)/(theta_plus-theta_minus), "grad_prior: ", (self.p.alpha-1)*np.log(theta) - self.p.beta*theta
#     true_grad = self.true_gradient( theta )
#     #grad = exact_grad*(x_plus-x_minus)/(theta_plus-theta_minus)+ (self.p.alpha-1)/theta - self.p.beta
#
#     print -grad4
#     print np.var(( self.loglike_x(plusXs)-self.loglike_x(minusXs))/(theta_plus-theta_minus))
#     print "THETA = ", theta, "TRUE = ", true_grad, "EST = ", -grad, "EST2 = ", -grad2, "EST3 = ", -grad3
#pdb.set_trace()
    gradients.append( np.squeeze( np.array([grad, theta, x,f, theta_plus, x_plus,f_plus,theta_minus, x_minus,f_minus])) )
    return -grad
    
    
  def posterior( self, thetas ):
    return np.exp( self.p.true_posterior_logpdf_func( thetas) )

  def view_results( self, thetas, stats, nsims, total_sims ):
    # plotting params
    nbins       = 20
    alpha       = 0.5
    label_size  = 8
    linewidth   = 3
    linecolor   = "r"
    
    # extract from states
    #thetas = states_object.get_thetas()[burnin:,:]
    #stats  = states_object.get_statistics()[burnin:,:]
    #nsims  = states_object.get_sim_calls()[burnin:]
    
    # plot sample distribution of thetas, add vertical line for true theta, theta_star
    f = pp.figure()
    sp = f.add_subplot(111)
    pp.plot( self.p.fine_theta_range, self.p.posterior, linecolor+"-", lw = 1)
    ax = pp.axis()
    pp.hist( thetas, self.p.nbins_coarse, range=self.p.range,normed = True, alpha = alpha )
    
    pp.fill_between( self.p.fine_theta_range, self.p.posterior, color="m", alpha=0.5)
    
    pp.plot( self.p.posterior_bars_range, self.p.posterior_bars, 'ro')
    pp.vlines( thetas.mean(), ax[2], ax[3], color="b", linewidths=linewidth)
    #pp.vlines( self.theta_star, ax[2], ax[3], color=linecolor, linewidths=linewidth )
    pp.vlines( self.p.posterior_mode, ax[2], ax[3], color=linecolor, linewidths=linewidth )
    
    pp.xlabel( "theta" )
    pp.ylabel( "P(theta)" )
    pp.axis([self.p.range[0],self.p.range[1],ax[2],ax[3]])
    set_label_fonsize( sp, label_size )
    
    #total_sims = states_object.get_sim_calls().sum()
    all_sims = nsims.sum()
    at_burnin = total_sims-all_sims
    errs = []
    time_ids = []
    nbr_sims = []
    
    for time_id in [1,5,10,25,50,75,100,200,300,400,500,750,1000,1500,2000,3000,4000,5000,7500,10000,12500,15000,17500,20000,25000,30000,35000,40000,45000,50000]:
      if time_id <= len(thetas):
        errs.append( bin_errors_1d(self.p.coarse_theta_range, self.p.posterior_cdf_bins, thetas[:time_id]) )
        time_ids.append(time_id)
        nbr_sims.append(nsims[:time_id].sum()+at_burnin)
        
    errs = np.array(errs)
    time_ids = np.array(time_ids)
    nbr_sims = np.array(nbr_sims)
    
    
    f2 = pp.figure()
    sp1 = f2.add_subplot(1,3,1)
    pp.semilogx( time_ids, errs, "bo-", lw=2)
    pp.xlabel( "nbr samples")
    pp.ylabel( "err")
    pp.grid('on')
    sp2 = f2.add_subplot(1,3,2)
    pp.semilogx( nbr_sims, errs, "ro-", lw=2)
    pp.xlabel( "nbr sims")
    pp.ylabel( "err")
    pp.grid('on')
    sp3 = f2.add_subplot(1,3,3)
    pp.semilogx( time_ids, errs, "bo-", lw=2)
    pp.semilogx( nbr_sims, errs, "ro-", lw=2)
    pp.xlabel( "time")
    pp.ylabel( "err")
    pp.grid('on')
    pp.show()
    #pdb.set_trace()
    print "ERROR    ",bin_errors_1d( self.p.coarse_theta_range, self.p.posterior_cdf_bins, thetas )
    #print "ACC RATE ", states_object.acceptance_rate()
    print "SIM      ", total_sims
    # return handle to figure for further manipulation
    print "times  ", time_ids
    print "errors ", errs
    return f

    
      
if __name__ == "__main__":
  pp.close('all')
  T             = 15000
  A = 5.01
  h = 0.002
  #h = 0.0005
  c = 0.001
  
  #injected_gradient_noise_std = 4.0
  np.random.seed(1)
  problem = generate_exponential( exp_problem, x_epsilon=200.0 )
  
  #THETA,X,LL,OMEGAS = run_true_mcmc( problem, T, theta=np.array([0.5]), verbose_rate=1000, use_omega = False  )
  #THETA,X,LL,OMEGAS = run_true_sgld( problem, T, h, A, c, theta=np.array([0.1]), verbose_rate=1000, use_omega = False  )
  #THETA,X,LL,OMEGAS = run_true_thermostats( problem, T, h, A, c, theta=np.array([0.1]), verbose_rate=1000, use_omega = False  )
  #THETA,X,LL,OMEGAS = run_true_rmsprop( problem, T, h, A, c, theta=np.array([0.1]), verbose_rate=1000, use_omega = False  )
  #THETA,X,LL,OMEGAS, grads = run_sgld( problem, T, h, A, c, theta=np.array([0.2]), verbose_rate=1000, use_omega = True, S = 2  )
  #THETA,X,LL,OMEGAS, grads = run_sghmc( problem, T, h, A, c, theta=np.array([0.2]), verbose_rate=1000, use_omega = True, S=2  )
  #THETA,X,LL,OMEGAS = run_mcmc( problem, T, theta=np.array([0.5]), verbose_rate=1000, use_omega = True  )
  THETA,X,LL,OMEGAS, grads = run_thermostats( problem, T, h, A, c, theta=np.array([0.2]), verbose_rate=1000, use_omega = True, S=2  )
  
  alpha=0.5
  alphav=0.9

  mg = []
  vg = []
  for g in range(T):
    if g==0:
      mg.append( grads["two_grads"][g,0] )
      vg.append( grads["two_grads"][g,0]**2 )
    else:
      mg.append( alpha*mg[-1] + (1-alpha)*grads["two_grads"][g,0] )
      vg.append( alphav*vg[-1] + (1-alphav)*grads["two_grads"][g,0]**2 )
  mg=np.array(mg)
  vg=np.array(vg)

  pp.figure(7)
  pp.clf()
  pp.subplot(1,2,1)
  pp.plot( grads["true_grads"][:,1],grads["true_grads"][:,0], 'b.')
  pp.plot( grads["two_grads"][:,1],grads["two_grads"][:,0], 'g.')

  pp.legend(["true","two","mg"]) #,"two"])
  pp.xlabel("theta")
  pp.ylabel("grad")
  pp.subplot( 1,2,2)
  for g in range(100):
    pp.plot( [grads["two_grads"][g,4],grads["two_grads"][g,7]], \
             [grads["two_grads"][g,6],grads["two_grads"][g,9]], 'b.-')


  pp.show()
  #assert False
  theta_range = np.linspace( 0.01,0.5,500)
  pp.figure(1)
  pp.clf()
  pp.subplot( 2,2,1)
  pp.plot( theta_range, problem.posterior.pdf(theta_range), 'k--', lw =2 )
  pp.hist( THETA[1000:],20, normed=True, alpha=0.5)
  pp.title( "Posterior")
  pp.ylabel( "p(theta)")
  pp.xlabel( "theta")
  
  pp.subplot( 2,2,2)
  pp.plot( THETA[-5000:], X[-5000:], 'b.-', alpha=0.5 )
  pp.hlines( problem.p.obs_statistics, 0, max( THETA[-5000:]),lw=4,alpha=0.5)
  pp.fill_between( [0, max( THETA[-5000:])], problem.y-problem.p.epsilon, problem.y+problem.p.epsilon, alpha=0.5,color='r')
  pp.fill_between( [0, max( THETA[-5000:])], problem.y-2*problem.p.epsilon, problem.y+2*problem.p.epsilon, alpha=0.25,color='r')
  pp.xlabel( "theta")
  pp.ylabel( "x")

  pp.subplot( 2,2,3)
  pp.plot( THETA[-5000:], 'b.', alpha=0.5 )
  pp.xlabel( "time")
  pp.ylabel( "theta")
  
  pp.subplot( 2,2,4)
  pp.hlines( problem.p.obs_statistics, 0, len(X[-5000:]),lw=4,alpha=0.5)
  pp.fill_between( np.arange(len(X[-5000:])), problem.y-problem.p.epsilon, problem.y+problem.p.epsilon, alpha=0.5,color='r')
  pp.fill_between( np.arange(len(X[-5000:])), problem.y-2*problem.p.epsilon, problem.y+2*problem.p.epsilon, alpha=0.25,color='r')
  pp.plot( X[-5000:], 'b.', alpha=0.5 )
  pp.xlabel( "time")
  pp.ylabel( "x")
  
  problem.view_results( THETA, X, np.ones(T+1), T+1 )
  
  pp.show()
