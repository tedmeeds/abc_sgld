from abcpy.factories import *
from abcpy.problems.blowfly.blowfly import *
from sa_algorithms import *
from scipy import stats as spstats
import pylab as pp

problem_params = default_params()
problem_params["std_log_tau"] = 0.2
problem_params["tau_is_log_normal"] = True
problem_params["epsilon"] = 0.1 #0.01 #10*np.array([1.0,1.0,0.5,0.5,0.5,0.1,0.01,0.5,2.0,1])
#problem_params["epsilon"] = 0.1*np.array([0.1,0.1,0.1,0.1,0.1,0.01,0.01,0.1,1.0,1.0])
problem_params["epsilon"] = 0.01*np.array([  0.91064006,   0.12437852,   1.06735857,   1.70135243,1.10402222,   0.22966667, 0.08973333,   1.28127273, 10.        ,   9.        ])

problem_params["epsilon"] = 2*np.array([  0.05,   0.05,   0.05,   0.05, 0.05,  0.02, 0.02,  0.05, 0.01        ,  3.0        ])

problem_params["blowfly_filename"] = "./data/blowfly.txt"
#problem_params["blowfly_filename"] = "./data/bf2.txt"
problem = BlowflyProblem( problem_params, force_init = True )

state_params = state_params_factory.scrape_params_from_problem( problem, S=5 )

# set is_marginal to true so using only "current" state will force a re-run
mcmc_params  = mcmc_params_factory.scrape_params_from_problem( problem, type="mh", is_marginal = True, nbr_samples = 1 )

# ignore algo, just doing this to get state object
algo_params = { "modeling_approach"  : "kernel",
                "observation_groups" : problem.get_obs_groups(),
                "state_params"       : state_params,
                "mcmc_params"        : mcmc_params,
                "algorithm"          : "model_mcmc"
              }
algo, model, state  = algo_factory.create_algo_and_state( algo_params )
recorder     = recorder_factory.create_recorder( {} )

class LikelihoodFree( object ):
  def __init__( self, problem, model, state, recorder ):
    self.problem = problem
    self.model = model
    self.state = state
    self.recorder = recorder
    self.tau   = 15
    
    self.moving_average_statistics = None
    self.count = 0
  
  def add_prediction( self, x_statistics ):
    if self.moving_average_statistics is None:
      self.moving_average_statistics = x_statistics.mean(0).copy()
      self.count = 1
    else:
      self.moving_average_statistics = (float( self.count)/float(self.count+1))*self.moving_average_statistics + (float( 1)/float(self.count+1))*x_statistics.mean(0).copy()
      self.count +=1
    self.error = pow( (self.state.observation_statistics - self.moving_average_statistics)/np.abs(self.state.observation_statistics), 2 ).mean()
    #pdb.set_trace()
    
  def fix_w( self, w ):
    if w[-1] > 7:
      w[-1] = 7
    if w[-2] < -7:
      w[-2] = -7
    if w[-3] < -7:
      w[-3] = -7
    return w
    
  def train_cost( self, w, seed = None ):
    #theta = np.hstack( (w,self.tau))
    theta = w.copy()
    #theta[-1] = np.ceil(np.exp(w[-1]))
    #print theta
    self.state.theta = theta#.reshape( (1,len(theta)))
    self.state.loglikelihood_is_computed = False
    self.model.set_current_state( self.state )
    
    if seed is not None:
      np.random.seed(seed)
    #pdb.set_trace()
    #log_posterior = self.model.log_posterior()
    #pdb.set_trace()
    loglikelihood = self.model.current.loglikelihood()
    return -loglikelihood 
  
  def grad_prior( self, w ):
    theta = w.copy()
    #theta[-1] = np.ceil(np.exp(w[-1]))
    return self.problem.theta_prior_logpdf_grad( theta )
    
  def train_error( self, w, seed = None ):
    # ignore w (assume already computed stats)
    cost = self.train_cost( w, seed )
    self.add_prediction( self.state.simulation_statistics )
    recorder.add_state( self.model.current.theta, self.model.current.simulation_statistics.mean(0).reshape( (1,model.current.simulation_statistics.shape[1]) ), -cost )
    self.cost=cost
    return self.error
    
  def test_error( self, w, seed = None ):
    # ignore w (assume already computed stats)
    return self.cost
    
    
if __name__ == "__main__":
  pp.close('all')
  abc_problem = LikelihoodFree( problem, model, state, recorder )
  
  max_iters = 2000
  #q         = 1
  #c         = 0.2
  #alpha     = 0.1
  #gamma     = 0.75
  mom_beta1 = 0.9 # on gradient
  mom_beta2 = 0.9 # on gradient_squared
  #cs = [0.5,0.1,0.2,0.3]
  cs = [10*0.01] #np.array([0.5,0.1,0.1,0.5,0.01,0.1])]
  #cs = [0.01*np.array([ 2. ,  2. ,  2. ,  2. ,  2. ,  0.5])]
  gammas = [0.9999]
  moms = [0.0]
  qs = [5]
  result = []
  for c in cs:
    # for "grad"
    #alpha = 1e-8 #*3*c/(4)
    #alpha = 10*1e-1*min(problem_params["epsilon"])
    # for others
    alpha = 1e-6 #*3*c/(4)
    #alpha  = 1.001 #0.00002 #*3*c/(4)
    for gamma in gammas:
      for mom in moms:
        for q in qs:
          np.random.seed(3)
          theta_rand = problem.theta_prior_rand()
          w = theta_rand #[:5]
          #w = problem.prior_means
          w = np.array([ 4.35903234, -1.35623493,  4.85356335, -2.80869085, -0.98398321, 1.85635901])
          #w[-1] = np.log(w[-1])
          spall_abc_params = {"ml_problem":abc_problem, 
                              "recorder":recorder,
                              "max_iters":max_iters, 
                              "q":q,
                              "c":c,
                              "alpha":alpha, 
                              "gamma_alpha":1, #0.9999,
                              "gamma_c": 1, #0.9999,
                              "gamma_eps":1, #0.9995,
                              "mom_beta1":mom_beta1,
                              "mom_beta2":mom_beta2,
                              "update_method":"grad",
                              "init_seed":20,
                              "verbose_rate":5,
                              "hessian":False,
                              "h_delta":0.1,
                              "q_rate":1.00,
                              "sgld_alpha":0.1,
                              "max_steps": 10.1*problem.prior_stds
                              }
          
          #wout, errors, others = spall_abc( w, spall_abc_params )
          wout, errors, others = spall_abc_sgld( w, spall_abc_params )
          #wout, errors = spall_with_hessian( w, spall_params )
  
          result.append({"c":c,"alpha":alpha,"gamma":gamma, "mom":mom, "q":q, "errors":errors,"w":wout})
  
  print "-------------------------------------------"
  vals = []
  i=0
  for r in result:
    train_error = r["errors"][-1][0]  # train error 
    vals.append(train_error)
  vals = np.array(vals)
  iorder = np.argsort(vals)
  
  for idx in iorder:
    print "error = %0.4f   a = %0.4f   mom = %0.4f  g = %0.4f q = %d"%( vals[idx], result[idx]["alpha"],result[idx]["mom"], result[idx]["gamma"], result[idx]["q"]) ,c


stats = recorder.get_statistics()[max_iters/2:,:] 
thetas = recorder.get_thetas()[max_iters/2:,:]    

pp.figure(figsize=(10,8))
for j in range(10):
  pp.subplot( 5,2,j+1 )
  pp.plot( stats[:,j], 'k', lw=2 )
  pp.hlines(problem.obs_statistics[j], 0, len(stats[:,j]), color="r", lw=4, alpha=4)
  pp.axis('tight')
pp.suptitle( "Statistics")
  
pp.figure(figsize=(10,8))
for j in range(10):
  pp.subplot( 5,2,j+1 )
  pp.hist( stats[:,j], 50, normed=True, alpha=0.5 )
  ax = pp.axis()
  pp.vlines(problem.obs_statistics[j], 0, ax[3], color="r", lw=4, alpha=4)
  pp.axis('tight')
pp.suptitle( "Posterior Predictive")  

D=6
pp.figure(figsize=(10,8))
for j in range(D):
  pp.subplot( D,1,j+1 )
  pp.plot( thetas[:,j], 'b', lw=2 )
  #pp.hlines(problem.obs_statistics[j], 0, len(stats[:,j]), color="r", lw=4, alpha=4)
  pp.axis('tight')
pp.suptitle( "Parameters")

pp.figure(figsize=(10,8))
for j in range(D):
  pp.subplot( D,2,2*j+1 )
  pp.hist( thetas[:,j], 50, normed=True, alpha=0.5 )
  ax = pp.axis()
  pd = spstats.norm(problem.prior_means[j], problem.prior_stds[j] )
  mn = min( ax[0],problem.prior_means[j]-1*problem.prior_stds[j] )
  mx = min( ax[1],problem.prior_means[j]+1*problem.prior_stds[j] )
  mn=ax[0]
  mx=ax[1]
  pp.plot( np.linspace(mn,mx,100), pd.pdf(np.linspace(mn,mx,100)), "k--" )
  #ax = pp.axis()
  #pp.vlines(problem.obs_statistics[j], 0, ax[3], color="b", lw=4, alpha=4)
  pp.axis('tight')
for j in range(D):
  pp.subplot( D,2,2*j+2 )
  pp.hist( np.exp(thetas[:,j]), 50, normed=True, alpha=0.5 )
  #ax = pp.axis()
  #pp.vlines(problem.obs_statistics[j], 0, ax[3], color="b", lw=4, alpha=4)
  pp.axis('tight')
pp.suptitle( "Left: Posterior of log, Right: Posterior")  

pp.figure()
pp.semilogy( others[1] )  
pp.semilogy( others[0][:,0],'k-',lw=4 )
pp.semilogy( others[0][:,1],'b-',lw=4 )
pp.title("gradient and injected")
pp.show()    
    