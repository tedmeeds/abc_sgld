from abcpy.factories import *
from abcpy.problems.blowfly.blowfly import *
from sa_algorithms import *

import pylab as pp

problem_params = default_params()
problem_params["epsilon"] = 0.05 #10*np.array([1.0,1.0,0.5,0.5,0.5,0.1,0.01,0.5,2.0,1])
problem_params["blowfly_filename"] = "./data/blowfly.txt"
problem = BlowflyProblem( problem_params, force_init = True )

state_params = state_params_factory.scrape_params_from_problem( problem )

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
  def __init__( self, model, state, recorder ):
    self.model = model
    self.state = state
    self.recorder = recorder
    self.tau   = 15
    
    self.moving_average_statistics = None
    self.count = 0
  
  def add_prediction( self, x_statistics ):
    if self.moving_average_statistics is None:
      self.moving_average_statistics = x_statistics.copy()
      self.count = 1
    else:
      self.moving_average_statistics = (float( self.count)/float(self.count+1))*self.moving_average_statistics + (float( 1)/float(self.count+1))*x_statistics
      self.count +=1
    self.error = pow( self.state.observation_statistics - self.moving_average_statistics, 2 ).sum()
    
  def train_cost( self, w, seed = None ):
    #theta = np.hstack( (w,self.tau))
    theta = w.copy()
    theta[-1] = int(np.exp(w[-1]))
    #print theta
    self.state.theta = theta#.reshape( (1,len(theta)))
    self.state.loglikelihood_is_computed = False
    self.model.set_current_state( self.state )
    
    if seed is not None:
      np.random.seed(seed)
    #pdb.set_trace()
    log_posterior = self.model.log_posterior()
    
    return -log_posterior 
    
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
  
  abc_problem = LikelihoodFree( model, state, recorder )
  
  max_iters = 5000
  q         = 1
  c         = 0.2
  alpha     = 0.1
  gamma     = 0.9
  
  #cs = [0.5,0.1,0.2,0.3]
  cs = [np.array([0.5,0.1,0.1,0.5,0.01,0.1])]
  gammas = [0.9999]
  moms = [0.9]
  qs = [5]
  result = []
  for c in cs:
    #alpha = 0.000001 #*3*c/(4)
    alpha  = 0.00002 #*3*c/(4)
    for gamma in gammas:
      for mom in moms:
        for q in qs:
          np.random.seed(10)
          theta_rand = problem.theta_prior_rand()
          w = theta_rand #[:5]
          w[-1] = np.log(w[-1])
          spall_abc_params = {"ml_problem":abc_problem, 
                              "recorder":recorder,
                              "max_iters":max_iters, 
                              "q":q,
                              "c":c,
                              "alpha":problem_params["epsilon"]*alpha, 
                              "gamma":gamma,
                              "mom":mom,
                              "init_seed":20,
                              "verbose_rate":50
                              }
          
          wout, errors = spall_abc( w, spall_abc_params )
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

stats = recorder.get_statistics()    
pp.figure(figsize=(10,8))
for j in range(10):
  pp.subplot( 5,2,j+1 )
  pp.plot( stats[:,j], 'k', lw=2 )
  pp.hlines(problem.obs_statistics[j], 0, len(stats[:,j]), color="r", lw=4, alpha=4)
  pp.axis('tight')
  
pp.show()    
    