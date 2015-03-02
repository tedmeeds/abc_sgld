import numpy as np
import scipy as sp
import pylab as pp
import pdb

from scipy import special

from abcpy.helpers import *
#from target_densities import *
#from models import *
#from gp import *
#from viewers import *
#from acquisitions import *
#from surrogate_mcmc import *

sqrt2 = np.sqrt( 2.0 )
sqrt_2pi = np.sqrt( 2.0 * np.pi )
log_sqrt_2pi = np.sqrt(sqrt_2pi)

cdf_gaussian = normcdf
log_pdf_gaussian = gaussian_logpdf



demo_range = np.linspace( -15, 15, 200 )
nbr_demo = len(demo_range)
use_noise_in_model = True


def q( w_old ):
  return w_old + 0.05*np.random.randn(3)

def gaussian_2randn(x, dim = 3):
  return x + np.random.randn(dim)*0.5

def mu( x ):
  p=0.01
  return -log_sqrt_2pi + 0.5*np.log(p)-0.5*p*((x)**2  ).sum()

def experiment( T, N, x0, params, true_target, nbr_bins ):
  disc = np.zeros(N)
  bin_edges   = true_target.make_bin_edges(nbr_bins)
  true_cdf = true_target.cdf_bins(bin_edges)
  for n in range(N):
    X = surrogate_mcmc( T, x0, params )

    pp.hist( X, bin_edges, alpha=0.5)
    est_cdf = pp.hist( X, bin_edges, alpha=0.5)[0]/float(T)
    print "total mass", est_cdf.sum()
    disc[n] = np.abs( true_cdf - est_cdf ).sum() + 1-est_cdf.sum()
    print n+1, disc[n]

  return disc

def std_mcmc( T, w0, model, problem, q ):
  W = [w0]
  wt = w0.copy()
  loglike_t = model.log_pdf( wt )

  LL = [loglike_t]
  for t in range(T):
    wp = q( wt )

    loglike_p = model.log_pdf( wp )

    u = np.random.rand()

    if np.log(u) <  loglike_p - loglike_t:
      wt = wp.copy()
      loglike_t = loglike_p

    LL.append( loglike_t )
    W.append( wt.copy() )

  W = np.array(W)
  LL = np.array(LL)
  return W, LL

def show_w( T, model, W, LL, gp = None, alpha=0.25 ):
  gridsize = max(20,int(np.sqrt(T/2)/2))
  gridsize=20
  scatter_ax=[-2,2,-2,2]
  pp.figure()
  pp.subplot( 3,2,1)
  model.show_all( W[T/2:][:] )
  pp.subplot( 3,2,2)
  pp.plot(W)
  pp.subplot( 3,3,4)
  #pp.plot(W[T/2:,0],W[T/2:,1],'o', alpha=alpha)
  pp.hexbin(W[T/2:,0],W[T/2:,1],extent=scatter_ax, gridsize=gridsize)
  pp.xlabel( "w0");pp.ylabel( "w1")
  pp.subplot( 3,3,5)
  #pp.plot(W[T/2:,0],W[T/2:,2],'o', alpha=alpha)
  pp.hexbin(W[T/2:,0],W[T/2:,2],extent=scatter_ax,gridsize=gridsize)
  pp.xlabel( "w0");pp.ylabel( "b")
  pp.subplot( 3,3,6)
  #pp.plot(W[T/2:,1],W[T/2:,2],'o', alpha=alpha)
  pp.hexbin(W[T/2:,1],W[T/2:,2],extent=scatter_ax,gridsize=gridsize)
  pp.xlabel( "w1");pp.ylabel( "b")
  pp.subplot( 3,3,7)
  pp.plot(W[T/2:,0],LL[T/2:],'o', alpha=alpha)
  pp.xlabel( "w0");
  pp.subplot( 3,3,8)
  pp.plot(W[T/2:,1],LL[T/2:],'o', alpha=alpha)
  pp.xlabel( "w1");
  pp.subplot( 3,3,9)
  pp.plot(W[T/2:,2],LL[T/2:],'o', alpha=alpha)
  pp.xlabel( "b");

  if gp is not None:
    X=[]
    y=[]
    if gp is not None:
      X = gp.Xtrain
      y = gp.ytrain

    pp.subplot( 3,2,1)
    for w in X:
      model.draw_w( w, 'm--', 0.5 )
    pp.subplot( 3,3,4)
    ax = pp.axis()
    pp.plot(X[:,0],X[:,1],'w+', alpha=1, mew=2,ms=10)
    pp.axis(ax)
    pp.subplot( 3,3,5)
    ax = pp.axis()
    pp.plot(X[:,0],X[:,2],'w+', alpha=1, mew=2,ms=10)
    pp.axis(ax)
    pp.subplot( 3,3,6)
    ax = pp.axis()
    pp.plot(X[:,1],X[:,2],'w+', alpha=1, mew=2,ms=10)
    pp.axis(ax)
    pp.subplot( 3,3,7)
    pp.plot(X[:,0],y,'ro', alpha=1)
    pp.subplot( 3,3,8)
    pp.plot(X[:,1],y,'ro', alpha=1)
    pp.subplot( 3,3,9)
    pp.plot(X[:,2],y,'ro', alpha=1)

if __name__ == "__main__":
  T = 5000
  N=5
  nbr_bins = 40
  dim = 3

  true_target_params = {}
  true_target_params["mu1"] = -1.0*np.ones( dim, dtype = float )
  true_target_params["mu2"] = 1.0*np.ones( dim, dtype = float )
  true_target_params["s1"]  = 1.0*np.ones( dim, dtype = float )
  true_target_params["s2"]  = 1.0*np.ones( dim, dtype = float )
  true_target_params["p1"]  = 0.5
  true_target_params["nbr_bins"] = nbr_bins

  kernel_prior  = {}
  kernel_params = np.array([50.0, 5.0, 5.0, 5.0])
  kernel_prior["signalA"] = 1.0
  kernel_prior["signalB"] = 0.5
  kernel_prior["lengthA"] = 2.5
  kernel_prior["lengthB"] = 0.5
  kernel = RBF( kernel_params, kernel_prior )

  # mean function
  mu_prior = {}
  mu_prior["dataCenterPriorMu"]        = np.zeros( (1,dim))
  mu_prior["dataCenterPriorPrecision"] = 0.5 # very imprecise center
  mu_prior["precPriorAlpha"] = 1.1 # precisions on data are gamma
  mu_prior["precPriorBeta"]  = 0.1   #


  # init Gaussian process
  kernel = RBF( kernel_params, kernel_prior )
  #noise_model = StandardNoiseModel( {"alpha":10.1, "beta":0.1} )
  noise_model = FixedNoiseModel( {"std_dev": 0.001} )
  mu = LogGaussianMean( mu_prior )
  gp_object = GaussianProcess( dim, kernel, noise_model, mu )
  gp_object.init_by_expectation()


  surrogate        = GaussianProcessSurrogate( gp_object )
  acceptance_model = LognormalMetropolisAcceptanceModel( corrected=True)
  #acceptance_model.corrected = False
  acquisition      = ExpectedAcceptanceImprovement( {"gp":    gp_object,\
                                                     "model": acceptance_model,\
                                                    "DIRECT": True })
  acquisition      = ProposalAcceptanceImprovement( {"gp":    gp_object,\
                                                     "model": acceptance_model,\
                                                    "DIRECT": False })

  np.random.seed(22)
  problem = TwoBlobClass( 0.0, 1.0, 1.0, 10, 2 )
  true_target_params["priorMu"] = 0
  true_target_params["priorVar"] = 1.0
  true_target_params["X"] = problem.X
  true_target_params["y"] = problem.y
  true_target      = DemoLogisticRegression( true_target_params )
  #bin_edges   = true_target.make_bin_edges(nbr_bins)
  #true_cdf    = true_target.cdf_bins( bin_edges )
  #f_x         = m.log_pdf

  w0 = np.random.randn(3)
  W,LL = std_mcmc( T, w0, true_target, problem, gaussian_2randn )

  approximate_target = ApproximateTarget( surrogate, acceptance_model, \
                                          acquisition, true_target.log_pdf, \
                                          true_target, corrected=acceptance_model.corrected)

  params = {}
  nbr_hmc_steps = 20
  hmc_epsilon = 0.01

  params["q_rand"]                  = gp_hmc
  params["q_params"]                = [gp_object, nbr_hmc_steps, hmc_epsilon]

  #params["f_x"]                     = true_target.log_pdf
  # params["q_rand"]                  = gaussian_2randn
#   params["q_params"]                = dim
  params["u_stream"]                = np.random.rand(T)
  params["max_nbr_reduction_steps"] = 1
  params["approximate_target"]      = approximate_target
  params["epsilon"]                 = 0.05
  x0  = 0.0*np.ones( (1,dim), dtype = float )
  f0 = true_target.log_pdf(x0)
  surrogate.add_data( x0, f0 )
  true_fp = surrogate.gp.get_free_params()

  minimize( true_fp, gp_grad_params_objective, gp_grad_p, [surrogate.gp], maxnumlinesearch=2 )
  # f0 = true_target.log_pdf(2.0*np.ones( (1,dim), dtype = float ))
#   surrogate.add_data( 2.0*np.ones( (1,dim), dtype = float ), f0 )
#   X = surrogate_mcmc( T, x0, params )
#   f0 = true_target.log_pdf(1.0*np.ones( dim, dtype = float ))
#   surrogate.add_data( 1.0*np.ones( dim, dtype = float ), f0 )
#   f0 = true_target.log_pdf(-1.0*np.ones( dim, dtype = float ))
#   surrogate.add_data( -1.0*np.ones( dim, dtype = float ), f0 )

  #disc = experiment( T, N, x0, params, true_target, nbr_bins )
  X = surrogate_mcmc( T, x0, params )
  X = X.squeeze()
  pp.figure()
  pp.subplot(1,2,1)
  pp.plot(X)
  pp.subplot(1,2,2)
  pp.plot(X[:,0],X[:,1],'o')
  pp.plot( surrogate.gp.Xtrain[:,0], surrogate.gp.Xtrain[:,1], 'ro' )
  #true_target.view( nbr_bins = nbr_bins, count = T )
  #pp.hist( X, bin_edges, alpha=0.5)

  #pp.plot(disc)
  #print disc.mean(),disc.std()

  model = true_target

  show_w( T, model, W, LL )
  pp.suptitle( "std mcmc")
  LLX = []
  for x in X:
    LLX.append( model.log_pdf(x))
  LLX = np.array(LLX)

  show_w( T, model, X, LLX, surrogate.gp )
  pp.suptitle( "gp mcmc")

  XX=np.mgrid[-2:2:10j,-2:2:10j,-2:2:10j]

  XXX = XX.reshape( (1000,3) )
  mu_xxx, cov_xxx = gp_object.full_posterior( XXX )
  #mu_xx = mu_xxx.reshape( (3,10,10,10 ))

  pp.figure()
  pp.subplot( 1,3,1 )
  pp.plot( XXX[:,0], mu_xxx, '.' )
  pp.subplot( 1,3,2 )
  pp.plot(  XXX[:,1], mu_xxx, '.' )
  pp.subplot( 1,3,3 )
  pp.plot( XXX[:,2], mu_xxx, '.' )


  pp.show()
