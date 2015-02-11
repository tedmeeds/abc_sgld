import numpy as np
import pylab as pp

from sghmc_problems import *
from sgnht_problems import *
from sg import *

def sample_density( samples, bins ):
  bw = bins[1]-bins[0]
  cnts, bins, patches = pp.hist( samples, bins )
  pdf = cnts / (bw*float(len(samples)))
  return pdf
    
def run_sghmc_with_problem( problem, T, h, A, Bhat, current_q, current_p = None, current_xi = None ):
  # extract problem functions
  U  = problem.U
  K  = problem.K
  dU = problem.dU
  
  # initialize
  if current_p is None:
    p = np.random.randn(2)
  else:
    p = current_p
  current_p = None
  q = current_q
  
  if current_xi is None:
    xi = A
  else:
    xi = current_xi
    
  P  = [p]
  Q  = [q]
  XI = [xi]
  for t in range(T):
    
    
    q = q + p*h
    
    grad_U = dU( q )
    p = p - A*p*h - grad_U*h + np.sqrt(2.0*(A-Bhat)*h)*np.random.randn()
    
    
    # update xi if it wasn't initialize
    if current_xi is None:
      xi = xi + (0.5*np.sum( p**2 ) - 1.0)*h
  
    P.append( p )
    Q.append( q )
    XI.append( xi )
    
  return np.squeeze(np.array(Q)), np.squeeze(np.array(P)), np.squeeze(np.array(XI))
  
def run_sgnht_with_problem( problem, T, h, A, current_q, current_p = None, current_xi = None ):
  # extract problem functions
  U  = problem.U
  K  = problem.K
  dU = problem.dU
  
  # initialize
  if current_p is None:
    p = np.random.randn(2)
  else:
    p = current_p
  current_p = None
  q = current_q
  
  if current_xi is None:
    xi = A
  else:
    xi = current_xi
    
  P  = [p]
  Q  = [q]
  XI = [xi]
  for t in range(T):
    grad_U = dU( q )
    p = p - xi*p*h - grad_U*h + np.sqrt(2.0*A*h)*np.random.randn()
    q = q + p*h
    
    # update xi if it wasn't initialize
    if current_xi is None:
      xi = xi + (0.5*np.sum( p**2 ) - 1.0)*h
  
    P.append( p )
    Q.append( q )
    XI.append( xi )
    
  return np.squeeze(np.array(Q)), np.squeeze(np.array(P)), np.squeeze(np.array(XI))
  

if __name__ == "__main__":
  pp.close('all')
  T             = 10**6
  L             = 2
  epsilon       = 0.01
  h             = epsilon
  B             = 1.0
  A             = 1.0
  C             = A
  Bhat          = 0.0
  xi            = None #10.0
  
  current_p     = None
  current_q     = np.array([0.0,1.0])
  mh_correction = True
  step_method   = leapfrog_step
  
  #injected_gradient_noise_std = 4.0
  
  # pp.figure(5)
  # pp.clf()
  # mxs = []
  # for t in range(100):
  #   np.random.seed(t)
  #   problem = generate_1d_gaussian_var_unknown(Ntilde = 100)
  #   p = problem.gamma_post.pdf( np.linspace( 0.1,2.0,100) )
  #   mx = max(p)
  #   mxs.append(mx)
  #   pp.plot( np.linspace( 0.1,2.0,100), problem.gamma_post.pdf( np.linspace( 0.1,2.0,100) ), lw=1 )
  #
  # mxs = np.array(mxs)
  # pp.show()
  # assert False
  np.random.seed(74) # seed set to find a posterior for gamma similar to paper SGNHT
  problem = generate_1d_gaussian_var_unknown(Ntilde = 10)
  
  cnts, bins, patches = pp.hist( problem.mu_post_samples, problem.bins_mu )
  mu_post_probs = cnts / float(len(problem.mu_post_samples))
  
  
  Qsghmc,Psghmc, XIsghmc = run_sghmc_with_problem( problem, T, h, C, Bhat, current_q, current_p = current_p, current_xi = xi  )
  Qsgnht,Psgnht, XIsgnht = run_sgnht_with_problem( problem, T, h, A, current_q, current_p = current_p, current_xi = xi  )
  
  #Qleap_no_mh,Pleap_no_mh = run_hmc_with_problem( T, problem, leapfrog_step, False, current_q, epsilon, L  )
  #Qmeul,Pmeul = run_hmc_with_problem( T, problem, mod_euler_step, mh_correction, current_q, epsilon, L  )
  #Qeul,Peul = run_hmc_with_problem( T, problem, euler_step, mh_correction, current_q, epsilon, L  )
  
  #assert False
  bins = problem.bins_mu
  bw   = bins[1]- bins[0]
  cnts, bins, patches = pp.hist( Qsgnht[:,0], problem.bins_mu )
  sgnht_mu_density = cnts / ( bw * float(len(Qsgnht[:,0]) ))
  cnts, bins, patches = pp.hist( Qsghmc[:,0], problem.bins_mu )
  sghmc_mu_density = cnts / ( bw * float(len(Qsghmc[:,0]) ))
  
  bins = problem.bins_gamma
  bw   = bins[1]- bins[0]
  cnts, bins, patches = pp.hist( Qsgnht[:,1], problem.bins_gamma )
  sgnht_gamma_density = cnts / ( bw * float(len(Qsgnht[:,1]) ))
  cnts, bins, patches = pp.hist( Qsghmc[:,1], problem.bins_gamma )
  sghmc_gamma_density = cnts / ( bw * float(len(Qsghmc[:,1]) ))
  
  pp.figure(1)
  pp.clf()
  
  
  
  #pp.plot( Qleap[:,0], Qleap[:,1], '.')
  #pp.plot( problem, problem["true_posterior"](problem["theta_range"]), 'k--', lw=3)
  pp.subplot(1,2,1)
  bins = problem.bins_mu
  bw   = bins[1]- bins[0]
  mu_post_density = mu_post_probs/bw
  pp.plot( bins[:-1]+0.5*bw, mu_post_density, 'k--', lw=3)
  pp.plot( bins[:-1]+0.5*bw, sgnht_mu_density, 'b-', lw=3)
  pp.plot( bins[:-1]+0.5*bw, sghmc_mu_density, 'g-', lw=3)
  
  #pp.hist( Qsgnht[:,0], 20,histtype='step', normed=True, alpha=0.75, linewidth=3 )
  #pp.hist( Qsghmc[:,0], 20,histtype='step', normed=True, alpha=0.75, linewidth=3 )
  ax = pp.axis()
  pp.axis( [problem.bins_mu[0],problem.bins_mu[-1],0,ax[3]])
  pp.legend( ["True","SG-NHT","SG-HMC"])
  pp.title( "mu")
  
  pp.subplot(1,2,2)
  bins = problem.bins_gamma
  bw   = bins[1]- bins[0]
  gamma_post_density = problem.gamma_post.pdf( bins[:-1]+0.5*bw )
  pp.plot( bins[:-1]+0.5*bw, problem.gamma_post.pdf( bins[:-1]+0.5*bw ), 'k--', lw=3 )
  pp.plot( bins[:-1]+0.5*bw, sgnht_gamma_density, 'b-', lw=3)
  pp.plot( bins[:-1]+0.5*bw, sghmc_gamma_density, 'g-', lw=3)
  
  #pp.hist( Qsgnht[:,1], 20,histtype='step', normed=True, alpha=0.75, linewidth=3 )
  #pp.hist( Qsghmc[:,1], 20,histtype='step', normed=True, alpha=0.75, linewidth=3 )
  ax = pp.axis()
  pp.axis( [problem.bins_gamma[0],problem.bins_gamma[-1],0,ax[3]])
  pp.legend( ["True","SG-NHT","SG-HMC"])
  pp.title( "gamma")
  pp.suptitle( "A = %d  h=%0.4f"%(A,epsilon))
  # pp.figure(2)
  # pp.clf()
  #
  # pp.subplot(1,1,1)
  # pp.plot( XIsgnht, alpha=0.75, linewidth=3 )
  # pp.plot( XIsgnht.cumsum()/(np.arange(T+1)+1.0), alpha=0.75, linewidth=3 )
  # #pp.plot( XIsghmc,  alpha=0.75,linewidth=3 )
  # pp.legend( ["SG-NHT","SG-NHT -- cummean"])
  # pp.title( "XI")
  
  # p=problem
  # b=p["bins"]
  # td = p["bin_density"]
  # sd = p["sample_density"]
  
  # print "NHT"
  # autocorr_times = [1,2,10,50,300]
  # for t in autocorr_times:
  #   x = Qsgnht[::t]
  #   sample_pdf = sd( x, b )
  #   print np.sum( np.sqrt( (td-sample_pdf)**2) )
  #
  # print "HMC"
  # for t in autocorr_times:
  #   x = Qsghmc[::t]
  #   sample_pdf = sd( x, b )
  #   print np.sum( np.sqrt( (td-sample_pdf)**2) )
  pp.figure(3)
    
  times = np.array([1,10,100,1000,5000,10000,50000,100000])
  errors_mu    = []
  errors_gamma = []
  for t in times:

    if len( Qsgnht ) >= t:
      bw = problem.bins_mu[1]-problem.bins_mu[0]
      td = bw*mu_post_density
      at = bw*sample_density( Qsgnht[:t,0], problem.bins_mu )
      bt = bw*sample_density( Qsghmc[:t,0], problem.bins_mu )
      errors_mu.append([0.5*np.sum( np.abs(td-at) ),0.5*np.sum( np.abs(td-bt) )])
      
      bw = problem.bins_gamma[1]-problem.bins_gamma[0]
      td = bw*gamma_post_density
      at = bw*sample_density( Qsgnht[:t,1], problem.bins_gamma )
      bt = bw*sample_density( Qsghmc[:t,1], problem.bins_gamma )
      errors_gamma.append([0.5*np.sum( np.abs(td-at) ),0.5*np.sum( np.abs(td-bt) )])

  errors_mu    = np.array(errors_mu)
  errors_gamma = np.array(errors_gamma)
  nt = len(errors_mu)

  
  pp.clf()
  pp.subplot(1,2,1)
  pp.semilogx( times[:nt], errors_mu, 'o-', lw=3 )
  pp.title("mu error")
  pp.legend(["SG-NHT", "SG-HMC"])
  
  pp.subplot(1,2,2)
  pp.semilogx( times[:nt], errors_gamma, 's-', lw=3 )
  pp.title("gamma error")
  pp.legend(["SG-NHT", "SG-HMC"])
  
  pp.suptitle("Error in dist A = %d  h=%0.4f"%(A,epsilon))
  pp.show()
  
  