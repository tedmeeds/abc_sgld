import numpy as np
import pylab as pp
import scipy as sp
from scipy import stats as spstats
from exp_wrapper import *
from abc_sgld.code.working_code import *

keep_x        = True 
init_seed     = 1
T             = 10000 # nbr of samples
verbose_rate  = 1000
C             = 1.01    # injected noise variance parameter
eta           = 0.01 # step size for Hamiltoniam dynamics
#h = 0.0005

# params for gradients
d_theta = 0.01  # step size for gradient estimate
S       = 5 
grad_params = {}
  
# keep theta within bounds
lower_bounds = np.array([0.001])
upper_bounds = np.array([np.inf])


# ----------------------- #
# common ransom seeds     #
# ----------------------- #
use_omega    = False    # use fixed random seeds for simulations
omega_rate   = 1.1    # probabilty of changing omegas
omega_switch = False    # if true, randomly change omegas
omega_sample = True   # sample omegas instead
assert (omega_switch * omega_sample) ==0, "only get to do one type of omega update"
omega_params = {"use_omega":use_omega, \
                "omega_rate":omega_rate, \
                "omega_switch":omega_switch,\
                "omega_sample":omega_sample}  


def view_grads(G,names,lines):
  Ss = G.keys()
  xr = np.linspace(-100,100,1000)
  pp.rc('text', usetex=True)
  pp.rc('font', family='times')
  f = pp.figure( figsize=(12,8) )
  lgs = []
  sp=pp.subplot(1,1,1)
  for line,S in zip(lines,Ss):
    g=G[S]
    g1 = spstats.kde.gaussian_kde(g[:,1])
    g2 = spstats.kde.gaussian_kde(g[:,2])
    
    #pp.hist( g[:,1], 50, normed=True, histtype='step', color="b",linewidth=2)
    #pp.hist( g[:,2], 50, normed=True, histtype='step', color="g",linewidth=2)
    
    pp.plot( xr, g1(xr), 'r'+line, lw=4)
    pp.plot( xr, g2(xr), 'b'+line, lw=4)
    lgs.append( "%s S=%d"%(names[0],S))
    lgs.append( "%s S=%d"%(names[1],S))
  pp.grid('on')
  pp.legend(lgs,loc=1,fancybox=True,prop={'size':18} )
  pp.xlim(xr[0],xr[-1])
  
  pp.ylabel( r'$p(\nabla\hat{U}({\bm \theta_{\text{MAP}}}))$', rotation='vertical' )
  pp.xlabel( r'$\nabla\hat{U}({\bm \theta_{\text{MAP}})$' )
  pp.title( r"Variance of Gradient Under Different Methods", fontsize=22) 
  
  set_tick_fonsize( sp, 16 )
  set_title_fonsize( sp, 32 )
  set_label_fonsize( sp, 24 )
  
  pp.show()
  
  
def make_figure(  problem, seeds, theta_range ):
  figsize = (12,8)

  pp.rc('text', usetex=True)
  pp.rc('font', family='times')
  f = pp.figure( figsize=figsize )
  sp=pp.subplot(1,1,1)
      
  theta0=theta_range[0]
  thetaN=theta_range[-1]
  
  #pp.plot( THETA[-5000:], X[-5000:], 'b.-', alpha=0.5 )
  pp.hlines( problem.p.obs_statistics, theta0, thetaN,lw=4,alpha=0.5)
  
  #pp.fill_between( [theta0, thetaN], problem.y-problem.p.epsilon, problem.y+problem.p.epsilon, alpha=0.5,color='r')
  pp.fill_between( [theta0, thetaN], problem.y-2*problem.p.epsilon, problem.y+2*problem.p.epsilon, alpha=0.25,color='r')

  mn = 1.0/theta_range
  sd = np.sqrt( 1.0/(problem.p.N*theta_range**2) )
  #pp.fill_between( theta_range, mn-sd, mn+sd, alpha=0.5,color='g')
  pp.fill_between( theta_range, mn-2*sd, mn+2*sd, alpha=0.25,color='k')

  X = np.zeros( (len(seeds),len(theta_range)))
  i = 0
  for s in seeds:
    
    j = 0
    for theta in theta_range:
      X[i,j] = problem.simulate( np.array([theta]), seed=s)[0][0]
      j+=1
    pp.plot(theta_range, X[i,:], 'k--',lw=2)
    i += 1
    
  nbr_pts = 75
  for n in range(nbr_pts):
    i = np.random.randint(len(seeds))
    j = np.random.randint(len(theta_range))
    pp.plot( [theta_range[j]], [X[i,j]], 'bo', ms=15, alpha=0.75)

  pp.ylim(0,20)
  pp.xlim(theta0,thetaN)
  #pp.xlabel( "theta")
  #pp.ylabel( "x")
  
  pp.ylabel( r'${\bf x}$', rotation='horizontal' )
  pp.xlabel( r'${\bm \theta}$' )
  
  #pp.legend( [nm], loc=1,fancybox=True,prop={'size':12} )
    
  pp.title( r"Simulation with Common Random Numbers", fontsize=22) 
  set_tick_fonsize( sp, 16 )
  set_title_fonsize( sp, 32 )
  set_label_fonsize( sp, 24 )
  
  

if __name__ == "__main__": 
  pp.close('all')
  save_it = True
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
                            "record_2side_sl_grad":False, "record_2side_keps_grad":False,"record_true_abc_grad":True,"record_true_grad":False}
  params["lower_bounds"] = lower_bounds
  params["upper_bounds"] = upper_bounds
  params["keep_x"]       = keep_x
  
  theta0 = np.array( [0.2])
  x0     = None
  np.random.seed(4)
  problem = generate_exponential( exp_problem )
  
  theta_map = problem.p.theta_posterior
  
  G={}
  names = [r'Kernel-${\bm \epsilon}$', r'Synthetic Likelihood']
  Ss = [5,50]
  lines = ["-","--"]
  eps = "0p37"
  for S in Ss:
    G[S] = np.loadtxt( "grads_at_map_S%d_eps%s_10K_2.txt"%(S,eps))
    
  view_grads( G, names, lines )
  # pp.figure(1)
  # pp.clf()
  # #pp.hist( G[:,0], normed=True, alpha=0.5 )
  # pp.subplot( 1,2,1); pp.hist( G[:,1], 50, normed=True, alpha=0.5 )
  # pp.subplot( 1,2,2); pp.hist( G[:,2], 50, normed=True, alpha=0.5 )
  # pp.legend( ["true abc", "sl", "keps"])
  #
  # pp.figure(2)
  # pp.clf()
  # #pp.hist( G[:,0], normed=True, alpha=0.5 )
  # pp.hist( G[:,1], 50, normed=True, alpha=0.5 )
  # pp.hist( G[:,2], 50, normed=True, alpha=0.5 )
  # pp.legend( ["keps","sl"])

    
  if save_it:
    pp.savefig("exp_varg_figure.pdf", format="pdf", dpi=600,bbox_inches="tight")
    #pp.savefig("../../papers/uai-2015/images/exp_varg_figure.pdf", format="pdf", dpi=600,bbox_inches="tight")
  pp.show()