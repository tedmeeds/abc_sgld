import numpy as np
import pylab as pp
import scipy as sp
from scipy import stats as spstats
from blowfly_wrapper import *
from habc_algos import *

import cPickle

#def pp_vector( x ):
def view_all_convs( problem, algo_convs, intervals, stats_2_use = [0,1,2,3,4,5,6,7,8]):
  figsize = (16,6)
  alpha=0.75
  pp.rc('text', usetex=True)
  pp.rc('font', family='times')
  f = pp.figure( figsize=figsize )
  sp=pp.subplot(1,1,1)
  
  
  for j in stats_2_use:
    pp.subplot( 3,3,j+1)
    
    leg_names = []
    for name, convs in algo_convs.iteritems():
      pp.loglog( intervals, convs[:,j], lw=2 )
      leg_names.append(name)
    pp.title( problem.p.stats_names[j] )
    pp.xlim( intervals[0],intervals[-1])
  pp.legend( leg_names, loc=1,fancybox=True,prop={'size':16} )
    
  
def quick_convergence_single_algo( y, X, intervals ):
  C = np.zeros( (len(y),len(intervals)))
  
  convs = []
  count = 0.0
  mean  = np.zeros(len(y))
  
  j = 0
  for i in xrange(len(X)):
    mean = count*mean + X[i]
    count+=1
    mean /= count
    
    convs.append( pow( y - mean, 2 )/pow(y,2) )
  
  convs = np.array(convs)
  used_convs = []
  used_times = []
  T = len(X)
  for t in intervals:  
    if t < T:
      used_convs.append( convs[t,:] )
      used_times.append(t)
  return np.array(used_convs),np.array(used_times)
    
def view_results( problem, thetas, all_stats, burnin = 1 ):
  stats = all_stats.mean(1)
  self=problem
  pp.rc('text', usetex=True)
  pp.rc('font', family='times')
  f = pp.figure( figsize=(12,8) )
  sp=pp.subplot(1,1,1)
  
  # plotting params
  nbins       = 20
  alpha       = 0.5
  label_size  = 8
  linewidth   = 3
  linecolor   = "r"
  
  f=pp.figure()
  for i in range(6):
    sp=f.add_subplot(2,10,i+1)
    pp.hist( thetas[:,i], 10, normed=True, alpha = 0.5)
    #pp.title( self.theta_names[i])
    set_label_fonsize( sp, 6 )
    set_tick_fonsize( sp, 6 )
    set_title_fonsize( sp, 8 )
  for i in range(10):
    sp=f.add_subplot(2,10,10+i+1)
    pp.hist( stats[:,i], 10, normed=True, alpha = 0.5)
    ax=pp.axis()
    pp.vlines( self.obs_statistics[i], 0, ax[3], color="r", linewidths=2)
    # if self.obs_statistics[i] < ax[0]:
    #   ax[0] = self.obs_statistics[i]
    # elif self.obs_statistics[i] > ax[1]:
    #   ax[1] = self.obs_statistics[i]
    pp.axis( [ min(ax[0],self.obs_statistics[i]), max(ax[1],self.obs_statistics[i]), ax[2],ax[3]] )
    #pp.title( self.stats_names[i])
    #set_label_fonsize( sp, 6 )
    #set_tick_fonsize( sp, 6 )
    #set_title_fonsize( sp, 8 )
  #pp.suptitle( "top: posterior, bottom: post pred with true")
  
  # f = pp.figure()
  # I = np.random.permutation( len(thetas) )
  # for i in range(16):
  #   sp=pp.subplot(4,4,i+1)
  #   theta = thetas[ I[i],:]
  #   test_obs = self.simulation_function( theta )
  #   test_stats = self.statistics_function( test_obs )
  #   err = np.sum( np.abs( self.obs_statistics - test_stats ) )
  #   pp.title( "%0.2f"%( err ))
  #   pp.plot( self.observations/1000.0 )
  #   pp.plot(test_obs/1000.0)
  #   pp.axis("off")
  #   set_label_fonsize( sp, 6 )
  #   set_tick_fonsize( sp, 6 )
  #   set_title_fonsize( sp, 8 )
  # pp.suptitle( "time-series from random draws of posterior")

def view_X_timeseries( problem, X, algoname, howmany = 1000 ):
  muX = X.mean(1) # average over S
  stdX = X.std(1)
  x_names = problem.p.stats_names
  
  figsize = (16,6)
  alpha=0.75
  pp.rc('text', usetex=True)
  pp.rc('font', family='times')
  f = pp.figure( figsize=figsize )
  sp=pp.subplot(1,1,1)
      
  for i in range(10):
    n = len(muX[-howmany:,i])
    sp=pp.subplot(5,2,i+1)
    pp.plot( muX[-howmany:,i], 'k-', lw=2, alpha=0.5)
    pp.fill_between( np.arange(n), muX[-howmany:,i]-2*stdX[-howmany:,i], muX[-howmany:,i]+2*stdX[-howmany:,i],color='b', alpha=0.5)
    y = problem.y[i]
    yplus = y + 2*problem.p.epsilon[i]
    yminus = y - 2*problem.p.epsilon[i]
    #pp.hlines( y,0,n-1,color='g',linewidth=2)
    pp.fill_between( np.arange(n),yminus,yplus,color='r',alpha=0.25)
    pp.hlines( y,0,n-1,color='r',linewidth=2)
    pp.hlines( yplus,0,n-1,color='r',linewidth=1)
    pp.hlines( yminus,0,n-1,color='r',linewidth=1)
    pp.ylabel( x_names[i], rotation='vertical')
    print "i ",x_names[i], y,yplus,yminus
    #pdb.set_trace()
    # set_tick_fonsize( sx_names[i]p, 16 )
    # set_title_fonsize( sp, 32 )
    # set_label_fonsize( sp, 24 )
  
  #pp.ylim(theta0,thetaN)
  #pp.xlabel( "theta")
  #pp.ylabel( "x")
  
  #pp.xlabel( r'time' )
  #pp.ylabel( r'${\bm \theta}$', rotation='horizontal' )
  
  pp.legend( [algoname], loc=1,fancybox=True,prop={'size':16} )
  
def view_theta_hist( problem, theta_range, samples, algoname, burnin ):
  figsize = (10,12)
  alpha=0.75
  pp.rc('text', usetex=True)
  pp.rc('font', family='times')
  f = pp.figure( figsize=figsize )
  sp=pp.subplot(1,1,1)
      
  #theta0=theta_range[0]
  #thetaN=theta_range[-1]
  theta_names = [r'\text{log} $P$', r'\text{log} ${\delta}$', r'\text{log} ${N_0}$', r'\text{log} ${\sigma_d}$', r'\text{log} ${\sigma_p}$', r'$\tau$']
  #pp.hist( samples, problem.p.nbins_coarse, range=problem.p.range,normed = True, alpha = alpha )
  for i in range(6):
    sp=pp.subplot(3,2,i+1)
    pp.hist( samples[burnin:,i], 20, normed=True, alpha=0.5)
    #pp.plot( samples[-howmany:,i], 'ko')
    pp.xlabel( theta_names[i], rotation='horizontal')
    # set_tick_fonsize( sp, 16 )
    # set_title_fonsize( sp, 32 )
    # set_label_fonsize( sp, 24 )
  
  #pp.ylim(theta0,thetaN)
  #pp.xlabel( "theta")
  #pp.ylabel( "x")
  
  #pp.xlabel( r'time' )
  #pp.ylabel( r'${\bm \theta}$', rotation='horizontal' )
  
  pp.legend( [algoname], loc=1,fancybox=True,prop={'size':16} )
      
def view_theta_timeseries( problem, theta_range, samples, algoname, howmany = 1000 ):
  figsize = (16,6)
  alpha=0.75
  pp.rc('text', usetex=True)
  pp.rc('font', family='times')
  f = pp.figure( figsize=figsize )
  sp=pp.subplot(1,1,1)
      
  #theta0=theta_range[0]
  #thetaN=theta_range[-1]
  theta_names = [r'\text{log} $P$', r'\text{log} ${\delta}$', r'\text{log} ${N_0}$', r'\text{log} ${\sigma_d}$', r'\text{log} ${\sigma_p}$', r'$\tau$']
  #pp.hist( samples, problem.p.nbins_coarse, range=problem.p.range,normed = True, alpha = alpha )
  for i in range(6):
    sp=pp.subplot(3,2,i+1)
    pp.plot( samples[-howmany:,i], 'k-', lw=2, alpha=0.5)
    pp.plot( samples[-howmany:,i], 'ko')
    pp.ylabel( theta_names[i], rotation='vertical')
    # set_tick_fonsize( sp, 16 )
    # set_title_fonsize( sp, 32 )
    # set_label_fonsize( sp, 24 )
  
  #pp.ylim(theta0,thetaN)
  #pp.xlabel( "theta")
  #pp.ylabel( "x")
  
  #pp.xlabel( r'time' )
  #pp.ylabel( r'${\bm \theta}$', rotation='horizontal' )
  
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
  
  pp.hist( samples[1000:], problem.p.nbins_coarse, range=problem.p.range,normed = True, alpha = alpha )
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
  
burnin        = 1000
keep_x        = True 
init_seed     = 4
T             = 10000 + burnin # nbr of samples
verbose_rate  = 50
C             = 10.01    # injected noise variance parameter
eta           = 0.01 # step size for Hamiltoniam dynamics
#h = 0.0005

# params for gradients
d_theta = 0.5  # step size for gradient estimate
S       = 10 
grad_params = {}
  


# ----------------------- #
# common ransom seeds     #
# ----------------------- #
use_omega    = True    # use fixed random seeds for simulations
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
  pp.close('False')
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
  params["grad_params"]["method"] = "spsa"
  params["grad_params"]["R"] = 2
  if use_omega:
    problem_name = "bf-sticky2"
  else:
    problem_name = "bf-no-sticky2"

  mu_log_P         = 2.0
  std_log_P        = 2.0
  mu_log_delta     = -1.0
  std_log_delta    = 2.0
  mu_log_N0        = 5.0
  std_log_N0       = 2.0
  mu_log_sigma_d   = 0.0
  std_log_sigma_d  = 2.0
  mu_log_sigma_p   = 0.0
  std_log_sigma_p  = 2.0
  mu_tau           = 15
  mu_log_tau       = np.log(mu_tau)
  std_log_tau      = 0.5
  
  params["lower_bounds"] = np.array([-4,-4,-1,-5,-5,3.0])
  params["upper_bounds"] = np.array([8,5,11,6,6,30.0])
  
  params["keep_x"]       = keep_x
  
  times = [5,10,100,500,1000,2000,5000,10000,20000,30000,40000,50000]
  
  #theta0 = np.array( [0.2])
  #x0     = None
  theta_range=None
  # initialize the simulator
  np.random.seed(4)
  problem = generate_blowfly( bf_problem )
  theta0 = theta0 = problem.p.theta_prior_rand()
  x0     = None
  
  seeds = np.arange(8)
  #theta_range = np.linspace( 0.01, 0.3, 200 )
  
  #problem = generate_exponential( exp_problem )
  
  params["grad_func"] = problem.two_sided_sl_gradient
  algonames = ["SL-MCMC","SG-Langevin", "SG-Thermostats"]
  algos = [run_mcmc,run_sgld,run_thermostats]
  #algonames = ["SG-Langevin"]
  #algos = [run_sgld]
  algonames = ["SG-Thermostats"]
  algos = [run_thermostats]
  #algonames = ["SL-MCMC"]
  #algos = [run_mcmc]
  #algonames = ["SL-HMC"]
  #algos = [run_sghmc]
  
  results = {}
  for algoname, algo in zip( algonames, algos):
    errors = []
    results[algoname] = {"results":[]}
    for chain_id in range(1):
      np.random.seed(init_seed + 1000*chain_id)
      theta0 = problem.p.theta_prior_rand()
    
      print "running chain %d for algo = %s    theta0 = %f"%(chain_id, algoname, theta0[0])
    
      if algoname == "SG-Langevin":
        params["eta"] = 0.1
      elif algoname == "SG-Thermostats":
        if use_omega:
          params["eta"] = 0.01
        else:
          params["eta"] = 0.01
         
      run_result = algo( problem, params, theta0, x0 )
      results[algoname]["results"].append(run_result)
      intervals = times #np.array([100,250,500,1000,1500,nbr_keep])-1
      algo_convs = {}
      # c_mcmc = quick_convergence_single_algo( problem.y, mcmc["X"].mean(1)[-nbr_keep:], intervals );algo_convs["MCMC"]=c_mcmc;
      used_conv, used_times = quick_convergence_single_algo( problem.y, run_result["X"].mean(1)[burnin:], intervals );
      errors.append( used_conv );
      
      #view_posterior( problem, theta_range, run_result["THETA"], algoname, burnin )
      view_theta_timeseries( problem, theta_range, run_result["THETA"], algoname, howmany = 1000 )
      if saveit:
        pp.savefig("../../../abc_sgld_blowfly/%s-%s-theta-timeseries-hist-%s-chain%d.pdf"%(problem_name, algoname, sticky_str, chain_id), format="pdf", dpi=600,bbox_inches="tight")
        
      view_theta_hist( problem, theta_range, run_result["THETA"], algoname, burnin )
      if saveit:
        pp.savefig("../../../abc_sgld_blowfly/%s-%s-theta-hist-%s-chain%d.pdf"%(problem_name, algoname, sticky_str, chain_id), format="pdf", dpi=600,bbox_inches="tight")
        
      view_X_timeseries( problem, run_result["X"], algoname, howmany = 1000 )
      if saveit:
        pp.savefig("../../../abc_sgld_blowfly/%s-%s-stats-timeseries-hist-%s-chain%d.pdf"%(problem_name, algoname, sticky_str, chain_id), format="pdf", dpi=600,bbox_inches="tight")

    errors = np.array(errors)
    mean_errors = np.squeeze( errors.mean(0) )
    std_errors = np.squeeze( errors.std(0) )

    results[algoname]["params"] = {"eta":eta,"d_theta":d_theta,"S":S,"C":C,"omega_params": omega_params,"grad_params":params["grad_params"]}
    results[algoname]["errors"] = errors
    results[algoname]["mean_errors"] = mean_errors
    results[algoname]["std_errors"] = std_errors
    if saveit:
      cPickle.dump(results, open("../../../abc_sgld_blowfly/%s-%s-results.pkl"%(problem_name, algoname),"w+"))
      np.save( "../../../abc_sgld_blowfly/%s-%s-tvd-errors-%s.npy"%(problem_name, algoname, sticky_str), errors )
      np.savetxt( "../../../abc_sgld_blowfly/%s-%s-tvd-mean-%s.txt"%(problem_name, algoname, sticky_str), mean_errors )
      np.savetxt( "../../../abc_sgld_blowfly/%s-%s-tvd-std-%s.txt"%(problem_name, algoname, sticky_str), std_errors )
      np.savetxt( "../../../abc_sgld_blowfly/%s-%s-tvd-times-%s.txt"%(problem_name, algoname, sticky_str), used_times )
    print "==================================="
    print "algo = %s    theta0 = %f"%(algoname, theta0[0])
    print "==================================="
    print "use_omega", use_omega
    print "eta ", eta
    print "S   ", S
    print "C   ", C
    print "grad func", params["grad_func"]
    #print "conv ", errors
    #print "final err", errors[:,-1]
    print "mean errors:"
    for j in range(mean_errors.shape[1]):
      print "%d  %3.8f  %3.8f  %3.8f"%(j,mean_errors[0,j],mean_errors[1,j],mean_errors[-1,j])
      
    print "==================================="
  
  #
  
  #view_all_convs( problem, algo_convs, intervals)
  pp.show()
