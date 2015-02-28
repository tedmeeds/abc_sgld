from abcpy.factories import *
from abcpy.problems.blowfly.blowfly import *

import numpy as np
import pylab as pp


from abc_sgld.code.working_code import *
#from sa_algorithms import *
from scipy import stats as spstats
import pylab as pp

default_epsilons = 0.5 + 0*np.array([ 0.08257003,  0.01013485,  0.01,  0.01, 0.01 , 0.01,  0.01,0.01,0.53027013,0.13338243] )

# exponential distributed observations with Gamma(alpha,beta) prior over lambda
problem_params = default_params()
problem_params["epsilon"] = default_epsilons
#problem_params["min_epsilon"] = 0.1*np.ones(10)
problem_params["q_factor"] = 0.2
problem_params["blowfly_filename"] = "./data/blowfly.txt"
#problem_params["tau_is_log_normal"] = True
bf_problem = BlowflyProblem( problem_params, force_init = True )

state_params = state_params_factory.scrape_params_from_problem( bf_problem, S=1 )

# set is_marginal to true so using only "current" state will force a re-run
mcmc_params  = mcmc_params_factory.scrape_params_from_problem( bf_problem, type="mh", is_marginal = False, nbr_samples = 1 )

# ignore algo, just doing this to get state object
algo_params = { "modeling_approach"  : "kernel",
                "observation_groups" : bf_problem.get_obs_groups(),
                "state_params"       : state_params,
                "mcmc_params"        : mcmc_params,
                "algorithm"          : "model_mcmc"
              }
algo, model, state  = algo_factory.create_algo_and_state( algo_params )
recorder     = recorder_factory.create_recorder( {} )


class generate_blowfly( object ):
  def __init__( self, abcpy_problem ):
    self.p = abcpy_problem
    self.y = np.squeeze( self.p.obs_statistics )
    self.x_epsilon = None
    self.J=len(self.y)
    

    # multivariate uses variance (norm uses std dev)
    self.kernel = spstats.multivariate_normal( self.p.obs_statistics, self.p.epsilon**2 )
    
    #self.posterior = spstats.gamma( self.p.alpha+self.p.N, 0, 1.0/(self.p.beta+self.p.obs_sum ))
    
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
    X = np.zeros( (S,self.J))  
    
    for s in range(S):
      if seed is not None:
        # save current state
        current_state = np.random.get_state()
        np.random.seed(seed[s])
    
    
      raw_outputs = self.p.simulation_function( theta )
      X[s] = self.p.statistics_function( raw_outputs)
      
      if seed is not None:
        # put back the current state
        np.random.set_state( current_state )
    
    return X
  
  def logprior_gradient( self, theta ):
    return self.p.theta_prior_logpdf_grad(theta)
    
  def true_gradient( self, theta, gradients ):
    
    grad = (self.p.alpha+self.p.N-1)/theta - (self.p.beta+self.p.obs_sum)
    
    params["logs"]["true"].append( np.squeeze( np.array([grad,theta])) )
    return -grad 
    
  def simulate_for_gradient(self, theta, d_theta, omega, S, params ): 
    seed = None
    
    seeds   = []
    X_plus  = np.zeros( (S,self.J))
    X_minus = np.zeros( (S,self.J))
    for s in range(S):
      theta_minus = theta-d_theta
      theta_plus  = theta_minus + 2*d_theta
      
      if omega is not None:
        seed = omega[s]
       
      if seed is None:
        state = np.random.get_state()
      
      X_plus[s] = self.simulate( theta_plus, seed=[seed] )
      
      if seed is None:
        np.random.set_state(state)
        
      X_minus[s] = self.simulate( theta_minus, seed=[seed] )
      
      seeds.append(seed)
      
    return X_plus, X_minus, seeds, theta_plus, theta_minus
       
  def two_sided_keps_gradient( self, theta, d_theta, omega, S, params ):
    D = len(theta)
    grad = np.zeros(D)
    
    if params["method"] == "fdsa":
      R = D
      for r in range(R):
        mask = np.zeros(D)
        mask[r] = 1.0
        
        X_plus, \
        X_minus, \
        seeds, \
        theta_plus, \
        theta_minus = self.simulate_for_gradient( theta, d_theta*mask, omega, S, params )
    
        f_plus = logsumexp( self.loglike_x(X_plus),0 ) - np.log(S)
        f_minus = logsumexp( self.loglike_x(X_minus),0 ) - np.log(S)
    
        grad[r] = (f_plus-f_minus)/(2*d_theta) 
        
      grad = grad + self.logprior_gradient(theta)
      
    elif params["method"] == "spsa":
      R = params["R"]
      for r in range(R):
        mask = 2*np.random.binomial(1,0.5,D)-1
        
        X_plus, \
        X_minus, \
        seeds, \
        theta_plus, \
        theta_minus = self.simulate_for_gradient( theta, d_theta*mask, omega, S, params )
    
        f_plus = logsumexp( self.loglike_x(X_plus),0 ) - np.log(S)
        f_minus = logsumexp( self.loglike_x(X_minus),0 ) - np.log(S)
    
        grad += ( (f_plus-f_minus)/(2*d_theta) )/mask
  
      grad = grad/R + self.logprior_gradient(theta)
    return -grad
    
  def two_sided_sl_gradient( self, theta, d_theta, omega, S, params ):
    D = len(theta)
    grad = np.zeros(D)
    
    if params["method"] == "fdsa":
      R = D
      for r in range(R):
        mask = np.zeros(D)
        mask[r] = 1.0
    
        X_plus, \
        X_minus, \
        seeds, \
        theta_plus, \
        theta_minus = self.simulate_for_gradient( theta, d_theta*mask, omega, S, params )

        mu_plus  = X_plus.mean(0)
        mu_minus = X_minus.mean(0)

        var_plus  = np.var(X_plus,0,ddof=0)
        var_minus = np.var(X_minus,0,ddof=0)
    
        Lplus  = spstats.multivariate_normal( mu_plus, var_plus+self.p.epsilon**2 )
        Lminus = spstats.multivariate_normal( mu_minus, var_minus+self.p.epsilon**2 )
    
        f_plus  = Lplus.logpdf( self.y )
        f_minus = Lminus.logpdf( self.y )
        
        grad[r] = (f_plus-f_minus)/(2*d_theta) 
      grad = grad + self.logprior_gradient(theta)
    elif params["method"] == "spsa":
      R = params["R"]
      for r in range(R):
        mask = 2*np.random.binomial(1,0.5,D)-1
    
        X_plus, \
        X_minus, \
        seeds, \
        theta_plus, \
        theta_minus = self.simulate_for_gradient( theta, d_theta*mask, omega, S, params )
    
        mu_plus  = X_plus.mean(0)
        mu_minus = X_minus.mean(0)
    
        var_plus  = np.var(X_plus,0,ddof=0)
        var_minus = np.var(X_minus,0,ddof=0)
    
        Lplus  = spstats.multivariate_normal( mu_plus, var_plus+self.p.epsilon**2 )
        Lminus = spstats.multivariate_normal( mu_minus, var_minus+self.p.epsilon**2 )
    
        f_plus  = Lplus.logpdf( self.y )
        f_minus = Lminus.logpdf( self.y )
        
        grad += ( (f_plus-f_minus)/(2*d_theta) )/mask
  
      grad = grad/R + self.logprior_gradient(theta)
        
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
        #errs.append( bin_errors_1d(self.p.coarse_theta_range, self.p.posterior_cdf_bins, thetas[:time_id]) )
        er = cramer_vonMises_criterion( self.p.posterior_cdf_bins, thetas[:time_id], self.p.coarse_theta_range )
        
        #errs.append( bin_sq_errors_1d(self.p.coarse_theta_range, self.p.posterior_cdf_bins, thetas[:time_id]) )
        errs.append( er  )
        time_ids.append(time_id)
        nbr_sims.append(nsims[:time_id].sum()+at_burnin)
        
    errs = np.array(errs)
    time_ids = np.array(time_ids)
    nbr_sims = np.array(nbr_sims)
    
    
    f2 = pp.figure()
    sp1 = f2.add_subplot(1,3,1)
    pp.loglog( time_ids, errs, "bo-", lw=2)
    pp.xlabel( "nbr samples")
    pp.ylabel( "err")
    pp.grid('on')
    sp2 = f2.add_subplot(1,3,2)
    pp.loglog( nbr_sims, errs, "ro-", lw=2)
    pp.xlabel( "nbr sims")
    pp.ylabel( "err")
    pp.grid('on')
    sp3 = f2.add_subplot(1,3,3)
    pp.loglog( time_ids, errs, "bo-", lw=2)
    pp.loglog( nbr_sims, errs, "ro-", lw=2)
    pp.xlabel( "time")
    pp.ylabel( "err")
    pp.grid('on')
    pp.show()
    #pdb.set_trace()
    #print "ERROR    ",bin_errors_1d( self.p.coarse_theta_range, self.p.posterior_cdf_bins, thetas )
    er = cramer_vonMises_criterion( self.p.posterior_cdf_bins, thetas, self.p.coarse_theta_range )
    #print "ERROR    ",bin_sq_errors_1d( self.p.coarse_theta_range, self.p.posterior_cdf_bins, thetas )
    print "ERROR    ",er
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