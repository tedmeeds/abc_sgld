from abcpy.factories import *
from abcpy.problems.exponential import *
#from sa_algorithms import *
from scipy import stats as spstats
import pylab as pp

problem_params = default_params()

problem_params["N"] = 50
problem_params["q_stddev"] = 0.5
problem_params["theta_star"]      = 0.2
problem_params["epsilon"] = 0.1*np.sqrt( 1.0 / (problem_params["N"]*problem_params["theta_star"]**2) )
problem_params["alpha"]           = 1.0
problem_params["beta"]            = 1.0
#problem_params["alpha"]           = 3.75
#problem_params["beta"]            = 1.01

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


class generate_exponential( object ):
  def __init__( self, abcpy_problem ):
    self.p = abcpy_problem
    self.kernel = spstats.norm( self.p.obs_statistics, self.p.epsilon )
    self.y = self.p.obs_statistics[0]
    self.x_epsilon = None
    
    self.posterior = spstats.gamma( self.p.alpha+self.p.N, 0, 1.0/(self.p.beta+self.p.obs_sum ))
    
  def prior_rand(self):
    return self.p.theta_prior_rand()
    
  def propose( self, theta ):
    return self.p.theta_proposal_rand( theta )
  
  def loglike_x( self, x ):
    S,J = x.shape
    L = logsumexp( self.kernel.logpdf( x ), 0 ) - np.log(S)
    return L
    
  def loglike_prior( self, theta ):
    return self.p.theta_prior_logpdf( theta )

  def loglike_posterior( self, theta ):
    return self.posterior.logpdf( theta )
    
  def loglike_proposal_theta( self, to_theta, from_theta ):
    return self.p.theta_proposal_logpdf( to_theta, from_theta )
    
  def simulate( self, theta, seed = None, S = 1 ):
    X = np.zeros( (S,1))  
    
    for s in range(S):
      if seed is not None:
        # save current state
        current_state = np.random.get_state()
        np.random.seed(seed+s)
    
    
      raw_outputs = self.p.simulation_function( theta )
      X[s] = self.p.statistics_function( raw_outputs)
      
      if seed is not None:
        # put back the current state
        np.random.set_state( current_state )
    
    return X
  
  def true_gradient( self, theta, gradients ):
    
    grad = (self.p.alpha+self.p.N-1)/theta - (self.p.beta+self.p.obs_sum)
    
    params["logs"]["true"].append( np.squeeze( np.array([grad,theta])) )
    return -grad 

  def true_abc_gradient( self, theta, d_theta, S, params ):
    theta=theta[0]
    c=0.001
    theta_minus = max(0.01,theta-d_theta)
    theta_plus = theta_minus + 2*d_theta
      
    mu_plus  = 1.0/theta_plus
    mu_minus = 1.0/theta_minus
    
    var_plus  = 1.0/(problem_params["N"]*theta_plus*theta_plus)
    var_minus = 1.0/(problem_params["N"]*theta_minus*theta_minus)
    
    Lplus  = spstats.norm( mu_plus, np.sqrt(var_plus+self.p.epsilon**2) )
    Lminus = spstats.norm( mu_minus, np.sqrt(var_minus+self.p.epsilon**2) )
    
    f_plus  = Lplus.logpdf( self.y )
    f_minus  = Lminus.logpdf( self.y )
    grad = (f_plus-f_minus)/(theta_plus-theta_minus) + (self.p.alpha-1)/theta - self.p.beta
    
    params["logs"]["true_abc"].append( np.squeeze( np.array([grad, theta, theta_plus, mu_plus,f_plus,theta_minus, mu_minus,f_minus])) )
    return -grad 
        
  # def one_sided_gradient( self, theta, x, omega, c, gradients ):
  #   exact_grad = ( self.y-x )/self.p.epsilon**2
  #
  #   #log_theta = np.log(theta)
  #
  #   f = self.loglike_x(x)
  #   state = np.random.get_state()
  #
  #   theta_plus = theta+c #np.exp( log_theta + c )
  #
  #   x_plus = self.simulate( theta_plus, omega )
  #
  #   f_plus = self.loglike_x(x_plus)
  #
  #   grad = (f_plus-f)/c + (self.p.alpha-1)*np.log(theta) - self.p.beta*theta
  #
  #   if grad < 0:
  #     grad = max(np.array([-15]),grad)
  #   else:
  #     grad = min(np.array([15]),grad)
  #   #grad = exact_grad*(x_plus-x)/(theta_plus-theta) + (self.p.alpha-1)/theta - self.p.beta
  #
  #   #print "one-sided: ", theta_plus, x_plus, theta, x, "exact ", exact_grad, "grad_x  ", (x_plus-x)/(theta_plus-theta), "grad_prior: ", (self.p.alpha-1)*np.log(theta) - self.p.beta*theta
  #
  #   #grad = exact_grad*(x_plus-x_minus)/(theta_plus-theta_minus)+ (self.p.alpha-1)*np.log(theta) - self.p.beta*theta
  #   #return -grad
  #   gradients.append( np.squeeze( np.array([grad, theta, x, f, theta_plus, x_plus, f_plus])) )
  #   return -grad
  #
  # def one_sided_gradient_old( self, theta, x, omega, c ):
  #   f = self.loglike_x(x)
  #
  #   x_plus = self.simulate( theta+c, omega )
  #
  #   f_plus = self.loglike_x(x_plus)
  #
  #   grad = (f_plus-f)/c + (self.p.alpha-1)/theta - self.p.beta
  #
  #   return -grad
    
  def simulate_for_gradient(self, theta, d_theta, omega, S, params ): 
    seed = None
    
    seeds   = []
    X_plus  = np.zeros( (S,1))
    X_minus = np.zeros( (S,1))
    for s in range(S):
      theta_minus = max(0.01,theta-d_theta)
      theta_plus = theta_minus + 2*d_theta
      
      if omega is not None:
        seed = omega+s
       
      if seed is None:
        state = np.random.get_state()
      
      X_plus[s] = self.simulate( theta_plus, seed=seed )[0]
      
      if seed is None:
        np.random.set_state(state)
        
      X_minus[s] = self.simulate( theta_minus, seed=seed )[0]
      
      seeds.append(seed)
      
    return X_plus, X_minus, seeds, theta_plus, theta_minus
       
  def two_sided_keps_gradient( self, theta, d_theta, omega, S, params ):
    X_plus, \
    X_minus, \
    seeds, \
    theta_plus, \
    theta_minus = self.simulate_for_gradient( theta, d_theta, omega, S, params )
    
    f_plus = logsumexp( self.loglike_x(X_plus),0 ) - np.log(S)
    f_minus = logsumexp( self.loglike_x(X_minus),0 ) - np.log(S)
    
    grad = (f_plus-f_minus)/(theta_plus-theta_minus) + (self.p.alpha-1)/theta - self.p.beta
    
    params["logs"]["2side_keps"].append( np.squeeze( np.array([grad, theta, theta_plus, x_plus,f_plus,theta_minus, x_minus,f_minus])) )
    return -grad
    
  def two_sided_sl_gradient( self, theta, d_theta, omega, S, params ):
    X_plus, \
    X_minus, \
    seeds, \
    theta_plus, \
    theta_minus = self.simulate_for_gradient( theta, d_theta, omega, S, params )
    
    mu_plus  = X_plus.mean()
    mu_minus = X_minus.mean()
    
    var_plus  = np.var(X_plus,ddof=0)
    var_minus = np.var(X_minus,ddof=0)
    
    Lplus  = spstats.norm( mu_plus, np.sqrt(var_plus+self.p.epsilon**2) )
    Lminus = spstats.norm( mu_minus, np.sqrt(var_minus+self.p.epsilon**2) )
    
    f_plus  = Lplus.logpdf( self.y )
    f_minus = Lminus.logpdf( self.y )
    
    grad = (f_plus-f_minus)/(theta_plus-theta_minus) + (self.p.alpha-1)/theta - self.p.beta
    
    params["logs"]["2side_sl"].append( np.squeeze( np.array([grad, theta, theta_plus, X_plus,f_plus,theta_minus, X_minus,f_minus])) )
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
    
  def view_single_chain( self, outs ):
    problem = self
    THETA = outs["THETA"]
    T = len(THETA)
    X = outs["X"]
     
    alpha=0.5
    alphav=0.9

    # mg = []
    # vg = []
    # for g in range(T):
    #   if g==0:
    #     mg.append( grads["two_grads"][g,0] )
    #     vg.append( grads["two_grads"][g,0]**2 )
    #   else:
    #     mg.append( alpha*mg[-1] + (1-alpha)*grads["two_grads"][g,0] )
    #     vg.append( alphav*vg[-1] + (1-alphav)*grads["two_grads"][g,0]**2 )
    # mg=np.array(mg)
    # vg=np.array(vg)
    #
    # pp.figure(7)
    # pp.clf()
    # pp.subplot(1,2,1)
    # pp.plot( grads["true_grads"][:,1],grads["true_grads"][:,0], 'b.')
    # pp.plot( grads["two_grads"][:,1],grads["two_grads"][:,0], 'g.')
    #
    # pp.legend(["true","two","mg"]) #,"two"])
    # pp.xlabel("theta")
    # pp.ylabel("grad")
    # pp.subplot( 1,2,2)
    # for g in range(100):
    #   pp.plot( [grads["two_grads"][g,4],grads["two_grads"][g,7]], \
    #            [grads["two_grads"][g,6],grads["two_grads"][g,9]], 'b.-')
    #
    #
    # pp.show()
    #assert False
    theta_range = np.linspace( 0.01,0.5,500)
    pp.figure(1)
    pp.clf()
    pp.subplot( 2,2,1)
    pp.plot( theta_range, problem.posterior.pdf(theta_range), 'k--', lw =2 )
    pp.hist( THETA[1000:],40, normed=True, alpha=0.5)
    pp.title( "Posterior")
    pp.ylabel( "p(theta)")
    pp.xlabel( "theta")
  
    pp.subplot( 2,2,2)
    pp.plot( THETA[-5000:], X[-5000:], 'b.-', alpha=0.5 )
    pp.hlines( problem.p.obs_statistics, 0, max( THETA[-5000:]),lw=4,alpha=0.5)
    pp.fill_between( [0, max( THETA[-5000:])], problem.y-problem.p.epsilon, problem.y+problem.p.epsilon, alpha=0.5,color='r')
    pp.fill_between( [0, max( THETA[-5000:])], problem.y-2*problem.p.epsilon, problem.y+2*problem.p.epsilon, alpha=0.25,color='r')
  
    theta_range = np.linspace( min( THETA[-5000:]), max( THETA[-5000:]), 200 )
    mn = 1.0/theta_range
    sd = np.sqrt( 1.0/(problem.p.N*theta_range**2) )
    pp.fill_between( theta_range, mn-sd, mn+sd, alpha=0.5,color='g')
    pp.fill_between( theta_range, mn-2*sd, mn+2*sd, alpha=0.25,color='k')
  
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