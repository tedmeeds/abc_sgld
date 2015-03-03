import numpy as np
import pylab as pp
import scipy as sp
from scipy import stats as spstats
from exp_wrapper import *
from habc_algos import *

def total_variational_distance_errors( problem, theta, times ):
  self=problem
  errs = []
  time_ids = []
  nbr_sims = []
  
  for time_id in times:
    if time_id <= len(theta):
      #errs.append( bin_errors_1d(self.p.coarse_theta_range, self.p.posterior_cdf_bins, thetas[:time_id]) )
      er = bin_errors_1d( self.p.coarse_theta_range, self.p.posterior_cdf_bins, theta[:time_id] )
      
      errs.append( er  )
      time_ids.append(time_id)
      
  errs = np.array(errs)
  time_ids = np.array(time_ids)
  return errs, time_ids
  
    
def view_theta_timeseries( problem, theta_range, samples, algoname, howmany = 1000 ):
  figsize = (16,6)
  alpha=0.75
  pp.rc('text', usetex=True)
  pp.rc('font', family='times')
  f = pp.figure( figsize=figsize )
  sp=pp.subplot(1,1,1)
      
  theta0=theta_range[0]
  thetaN=theta_range[-1]
  
  #pp.hist( samples, problem.p.nbins_coarse, range=problem.p.range,normed = True, alpha = alpha )
  pp.plot( samples[-howmany:], 'k-', lw=2, alpha=0.5)
  pp.plot( samples[-howmany:], 'ko')
  
  pp.ylim(theta0,thetaN)
  #pp.xlabel( "theta")
  #pp.ylabel( "x")
  
  pp.xlabel( r'time' )
  pp.ylabel( r'${\bm \theta}$', rotation='horizontal' )
  
  pp.legend( [algoname], loc=1,fancybox=True,prop={'size':16} )
  
    
  #pp.title( r"Simulation with Common Random Numbers", fontsize=22) 
  set_tick_fonsize( sp, 16 )
  set_title_fonsize( sp, 32 )
  set_label_fonsize( sp, 24 )

def view_posterior( problem, theta_range, samples, algoname, burnin = 1000 ):
  figsize = (6,6)
  alpha=0.75
  pp.rc('text', usetex=True)
  pp.rc('font', family='times')
  f = pp.figure( figsize=figsize )
  sp=pp.subplot(1,1,1)
      
  theta0=theta_range[0]
  thetaN=theta_range[-1]
  
  pp.hist( samples[burnin:], problem.p.nbins_coarse, range=problem.p.range,normed = True, alpha = alpha )
  pp.plot( theta_range, np.exp( problem.loglike_posterior(theta_range)), 'k--', lw=3)
  
  pp.xlim(theta0,thetaN)
  #pp.xlabel( "theta")
  #pp.ylabel( "x")
  
  pp.ylabel( r'$\pi( {\bm \theta } | {\bf y} )$', rotation='vertical' )
  pp.xlabel( r'${\bm \theta}$' )
  
  pp.legend( ["True Posterior",algoname], loc=1,fancybox=True,prop={'size':16} )
  
    
  #pp.title( r"Simulation with Common Random Numbers", fontsize=22) 
  set_tick_fonsize( sp, 16 )
  set_title_fonsize( sp, 32 )
  set_label_fonsize( sp, 24 )
  

burnin = 1000

keep_x        = True 
init_seed     = 4
T             = 50000 + burnin # nbr of samples
verbose_rate  = 1000
C             = 10.0    # injected noise variance parameter
eta           = 0.01 # step size for Hamiltoniam dynamics
#C = eta/2
#h = 0.0005

# params for gradients
d_theta = 0.001  # step size for gradient estimate
S       = 5
grad_params = {}
  
# keep theta within bounds
lower_bounds = np.array([0.001])
upper_bounds = np.array([np.inf])


# ----------------------- #
# common ransom seeds     #
# ----------------------- #
use_omega    = False    # use fixed random seeds for simulations
omega_rate   = 0.1    # probabilty of changing omegas
omega_switch = True    # if true, randomly change omegas
omega_sample = False   # sample omegas instead
assert (omega_switch * omega_sample) ==0, "only get to do one type of omega update"
omega_params = {"use_omega":use_omega, \
                "omega_rate":omega_rate, \
                "omega_switch":omega_switch,\
                "omega_sample":omega_sample}  
if use_omega:
  sticky_str = "omega-rate-0p1"
else:
  sticky_str = "omega-rate-100p0"
  
if __name__ == "__main__": 
  pp.close('all')
  saveit = True
  chain_id = 1
  params = {}
  params["chain_id"]  = chain_id
  params["init_seed"] = init_seed
  params["T"]       = T
  params["S"]       = S
  params["d_theta"] = d_theta
  params["eta"]     = eta
  params["C"]       = C
  params["omega_params"] = omega_params
  params["verbose_rate"] = verbose_rate
  params["grad_params"]  = {"logs":{"true":[],"true_abc":[],"2side_keps":[],"2side_sl":[]},\
                            "record_2side_sl_grad":False, "record_2side_keps_grad":False,"record_true_abc_grad":False,"record_true_grad":False}
  params["lower_bounds"] = lower_bounds
  params["upper_bounds"] = upper_bounds
  params["keep_x"]       = keep_x
  params["mh_correction"] = False
  
  problem_name = "exp3"
  theta0 = np.array( [0.2])
  x0     = None
  times = [10,100,1000,5000,10000,20000,30000,40000,50000]
  
  # initialize the simulator
  theta0 = np.array( [0.2])
  x0     = None
  np.random.seed(4)
  problem = generate_exponential( exp_problem )
  
  seeds = np.arange(8)
  theta_range = np.linspace( 0.01, 0.3, 200 )
  
  problem = generate_exponential( exp_problem )
  params["grad_func"] = problem.two_sided_sl_gradient
  #params["grad_func"] = problem.two_sided_keps_gradient

  # init randomly for MCMC chain
  np.random.seed(init_seed + 1000*chain_id)
  
  algonames = ["SL-MCMC","SG-Langevin", "SG-HMC", "SG-Thermostats"]
  algos = [run_mcmc,run_sgld, run_sghmc,run_thermostats]
  algonames = ["SG-Thermostats"]
  algos = [run_thermostats]
  #
  # algonames = ["SG-Langevin","SG-HMC","SG-Thermostats"]
  # algos     = [run_sgld, run_sghmc,run_thermostats]
  
  #algonames = ["SG-Langevin"]
  #algos     = [run_sgld]
  #algonames = ["SL-MCMC"]
  #algos     = [run_mcmc]
  
  #algonames = ["SG-HMC"]
  #algos     = [run_sghmc]
  #algonames = ["SG-Thermostats"]
  #algos     = [run_thermostats]
    
  for algoname, algo in zip( algonames, algos):
    errors = []
    for chain_id in range(5):
      np.random.seed(init_seed + 1000*chain_id)
      theta0 = problem.p.theta_prior_rand()
      while theta0[0] < 0.01:
        theta0 = problem.p.theta_prior_rand()
      while theta0[0] > 2.0:
        theta0 = problem.p.theta_prior_rand()
    
      print "running chain %d for algo = %s    theta0 = %f"%(chain_id, algoname, theta0[0])
      C=10.0
      if algoname == "SG-HMC":
        C = 100.0
      params["C"] = C
      # if algoname == "SG-Langevin":
      #   params["eta"] = 0.02
      #   params["omega_params"]["omega_rate"] = 0.5
      #   sticky_str = "omega-rate-0p5"
      #   print "  setting eta ",params["eta"]
      # else:
      #   params["eta"] = 0.005
      #   params["omega_params"]["omega_rate"] = 0.01
      #   sticky_str = "omega-rate-0p01"
      #   print "  setting eta ",params["eta"]
        
      run_result = algo( problem, params, theta0, x0 )
    
      view_posterior( problem, theta_range, run_result["THETA"], algoname, burnin=burnin )
      if saveit:
        pp.savefig("../../../abc_sgld_exponential/%s-%s-posterior-hist-%s-chain%d.pdf"%(problem_name, algoname, sticky_str, chain_id), format="pdf", dpi=600,bbox_inches="tight")
      
      view_theta_timeseries( problem, theta_range, run_result["THETA"], algoname, howmany = 1000 )
      if saveit:
        pp.savefig("../../../abc_sgld_exponential/%s-%s-theta-timeseries-%s-chain%d.pdf"%(problem_name, algoname, sticky_str, chain_id), format="pdf", dpi=600,bbox_inches="tight")

      pp.close('all')
      errs,used_times = total_variational_distance_errors( problem, run_result["THETA"][burnin:], times )
      print "eta ", eta
      print "d_theta", d_theta
      print "grad type ", params["grad_func"]
      print "times", used_times
      print "tvd err",errs
      errors.append(errs)
    errors = np.array(errors)
    used_times = np.array(used_times)
    if saveit:
      np.savetxt( "../../../abc_sgld_exponential/%s-%s-tvd-%s.txt"%(problem_name, algoname, sticky_str), errors )
      np.savetxt( "../../../abc_sgld_exponential/%s-%s-tvd-times-%s.txt"%(problem_name, algoname, sticky_str), used_times )
      #np.savetxt( "./images/%s-%s-tvd-%s.txt"%(problem_name, algoname, sticky_str), errors )
      #np.savetxt( "./images/%s-%s-tvd-times-%s.txt"%(problem_name, algoname, sticky_str), used_times )
      
      
    
  pp.show()