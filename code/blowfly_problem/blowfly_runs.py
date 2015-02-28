import numpy as np
import pylab as pp
import scipy as sp
from scipy import stats as spstats
from blowfly_wrapper import *
from habc_algos import *

#def pp_vector( x ):
  
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
  
def view_theta_hist( problem, theta_range, samples, algoname, howmany = 1000 ):
  figsize = (12,8)
  alpha=0.75
  pp.rc('text', usetex=True)
  pp.rc('font', family='times')
  f = pp.figure( figsize=figsize )
  sp=pp.subplot(1,1,1)
      
  #theta0=theta_range[0]
  #thetaN=theta_range[-1]
  theta_names = [r'\text{log} $P$', r'\text{log} ${\delta}$', r'\text{log} ${N_0}$', r'\text{log} ${\sigma_d}$', r'\text{log} ${\sigma_p}$', r'\text{log} $\tau$']
  #pp.hist( samples, problem.p.nbins_coarse, range=problem.p.range,normed = True, alpha = alpha )
  for i in range(6):
    sp=pp.subplot(3,2,i+1)
    pp.hist( samples[-howmany:,i], 20, normed=True, alpha=0.5)
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
  theta_names = [r'\text{log} $P$', r'\text{log} ${\delta}$', r'\text{log} ${N_0}$', r'\text{log} ${\sigma_d}$', r'\text{log} ${\sigma_p}$', r'\text{log} $\tau$']
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
  # set_tick_fonsize( sp, 16 )
  # set_title_fonsize( sp, 32 )
  # set_label_fonsize( sp, 24 )

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
  
keep_x        = True 
init_seed     = 4
T             = 2000 # nbr of samples
verbose_rate  = 10
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

if __name__ == "__main__": 
  pp.close('all')
  saveit = False
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
  params["grad_params"]["R"] = 1
  

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
  #params["grad_func"] = problem.two_sided_keps_gradient

  # init randomly for MCMC chain
  np.random.seed(init_seed + 1000*chain_id)
  
  # run algorithm
  np.random.seed(init_seed + 1000*chain_id)
  sgnht = run_thermostats( problem, params, theta0, x0 )
  algoname = "SG-Thermostats"
  # view_posterior( problem, theta_range, sgnht["THETA"], algoname, burnin=1000 )
  # # if saveit:
  # #   pp.savefig("bf-%s-posterior_hist.pdf"%(algoname), format="pdf", dpi=600,bbox_inches="tight")
  # #   pp.savefig("../../papers/uai-2015/images/bf-%s-posterior_hist.pdf"%(algoname), format="pdf",dpi=600,bbox_inches="tight")
  view_theta_timeseries( problem, theta_range, sgnht["THETA"], algoname, howmany = 1000 )
  view_theta_hist( problem, theta_range, sgnht["THETA"], algoname, howmany = 1000 )
  view_X_timeseries( problem, sgnht["X"], algoname, howmany = 1000 )
  # # if saveit:
  # #   pp.savefig("bf-%s-theta-timeseries.pdf"%(algoname), format="pdf", dpi=600,bbox_inches="tight")
  # #   pp.savefig("../../papers/uai-2015/images/bf-%s-theta-timeseries.pdf"%(algoname), format="pdf", dpi=600,bbox_inches="tight")
  # #
  algoname = "SG-HMC"
  np.random.seed(init_seed + 1000*chain_id)
  sghmc = run_sghmc( problem, params, theta0, x0 )
  view_theta_timeseries( problem, theta_range, sghmc["THETA"], algoname, howmany = 1000 )
  view_theta_hist( problem, theta_range, sghmc["THETA"], algoname, howmany = 1000 )
  view_X_timeseries( problem, sghmc["X"], algoname, howmany = 1000 )
  
  #
  np.random.seed(init_seed + 1000*chain_id)
  sgld = run_sgld( problem, params, theta0, x0 )
  algoname = "SG-Langevin"
  # view_posterior( problem, theta_range, sgld["THETA"], algoname, burnin=1000 )
  # if saveit:
  #   pp.savefig("exp-%s-posterior_hist.pdf"%(algoname), format="pdf", dpi=600,bbox_inches="tight")
  #   pp.savefig("../../papers/uai-2015/images/exp-%s-posterior_hist.pdf"%(algoname), format="pdf", dpi=600,bbox_inches="tight")
  view_theta_timeseries( problem, theta_range, sgld["THETA"], algoname, howmany = 1000 )
  view_theta_hist( problem, theta_range, sgld["THETA"], algoname, howmany = 1000 )
  view_X_timeseries( problem, sgld["X"], algoname, howmany = 1000 )
  # if saveit:
  #   pp.savefig("bf-%s-theta-timeseries.pdf"%(algoname), format="pdf", dpi=600,bbox_inches="tight")
  #   pp.savefig("../../papers/uai-2015/images/bf-%s-theta-timeseries.pdf"%(algoname), format="pdf", dpi=600,bbox_inches="tight")
  #
  #
  np.random.seed(init_seed + 1000*chain_id)
  mcmc = run_mcmc( problem, params, theta0, x0 )
  algoname = "ABC-MCMC"
  #view_posterior( problem, theta_range, mcmc["THETA"], algoname, burnin=1000 )
  #if saveit:
  #  pp.savefig("exp-%s-posterior_hist.pdf"%(algoname), format="pdf", dpi=600,bbox_inches="tight")
  #  pp.savefig("../../papers/uai-2015/images/exp-%s-posterior_hist.pdf"%(algoname), format="pdf", dpi=600,bbox_inches="tight")
  view_theta_timeseries( problem, theta_range, mcmc["THETA"], algoname, howmany = 1000 )
  view_theta_hist( problem, theta_range, mcmc["THETA"], algoname, howmany = 1000 )
  view_X_timeseries( problem, mcmc["X"], algoname, howmany = 1000 )

  if saveit:
    pp.savefig("exp-%s-theta-timeseries.pdf"%(algoname), format="pdf", dpi=600,bbox_inches="tight")
    pp.savefig("../../papers/uai-2015/images/exp-%s-theta-timeseries.pdf"%(algoname), format="pdf", dpi=600,bbox_inches="tight")
  #  #
  # results = {}
  # results["mcmc"] = mcmc
  # results["sgld"] = sgld
  # results["sgnht"] = sgnht
  # pp.figure(1)
  # pp.clf()
  # pp.plot( mcmc["THETA"][-1000:])
  # pp.plot( sgld["THETA"][-1000:])
  # pp.plot( sghmc["THETA"][-1000:])
  # pp.plot( sgnht["THETA"][-1000:])
  # pp.legend( ["MCMC","SGLD","SGHMC","SGNHT"])
  # view results of single chain
  #problem.view_single_chain( sgnht )
    
  pp.show()