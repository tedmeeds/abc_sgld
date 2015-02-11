import numpy as np
import pylab as pp

from sghmc_problems import *
from sgnht_problems import *
from sg import *

def run_sghmc_with_problem( problem, T, h, A, Bhat, current_q, current_p = None, current_xi = None ):
  # extract problem functions
  U  = problem["U"]
  K  = problem["K"]
  dU = problem["dU"]
  
  # initialize
  if current_p is None:
    p = np.random.randn(1)
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
      xi = xi + (p**2 - 1.0)*h
  
    P.append( p )
    Q.append( q )
    XI.append( xi )
    
  return np.squeeze(np.array(Q)), np.squeeze(np.array(P)), np.squeeze(np.array(XI))
  
def run_sgnht_with_problem( problem, T, h, A, current_q, current_p = None, current_xi = None ):
  # extract problem functions
  U  = problem["U"]
  K  = problem["K"]
  dU = problem["dU"]
  
  # initialize
  if current_p is None:
    p = np.random.randn(1)
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
      xi = xi + (p**2 - 1.0)*h
  
    P.append( p )
    Q.append( q )
    XI.append( xi )
    
  return np.squeeze(np.array(Q)), np.squeeze(np.array(P)), np.squeeze(np.array(XI))
  

if __name__ == "__main__":
  pp.close('all')
  T             = 10**5
  L             = 2
  epsilon       = 0.01
  h             = epsilon
  B             = 1.0
  A             = 10.0
  C             = A
  Bhat          = 0.0
  xi            = None #10.0
  
  current_p     = None
  current_q     = 0.0
  mh_correction = True
  step_method   = leapfrog_step
  
  #injected_gradient_noise_std = 4.0
  
  problem = generate_1d_gaussian_mu_unknown()
  

  Qsghmc,Psghmc, XIsghmc = run_sghmc_with_problem( problem, T, h, C, Bhat, current_q, current_p = current_p, current_xi = xi  )
  Qsgnht,Psgnht, XIsgnht = run_sgnht_with_problem( problem, T, h, A, current_q, current_p = current_p, current_xi = xi  )
  
  #Qleap_no_mh,Pleap_no_mh = run_hmc_with_problem( T, problem, leapfrog_step, False, current_q, epsilon, L  )
  #Qmeul,Pmeul = run_hmc_with_problem( T, problem, mod_euler_step, mh_correction, current_q, epsilon, L  )
  #Qeul,Peul = run_hmc_with_problem( T, problem, euler_step, mh_correction, current_q, epsilon, L  )
  
  pp.figure(1)
  pp.clf()
  
  #pp.plot( Qleap[:,0], Qleap[:,1], '.')
  pp.plot( problem["theta_range"], problem["true_posterior"](problem["theta_range"]), 'k--', lw=3)
  pp.hist( Qsgnht, 20,histtype='step', normed=True, alpha=0.75, linewidth=3 )
  pp.hist( Qsghmc, 20,histtype='step', normed=True, alpha=0.75, linewidth=3 )
  ax = pp.axis()
  pp.axis( [-0.6,1,0,ax[3]])
  pp.legend( ["True","SG-NHT","SG-HMC"])
  
  pp.figure(2)
  pp.clf()
  
  pp.subplot(1,1,1)
  pp.plot( XIsgnht, alpha=0.75, linewidth=3 )
  pp.plot( XIsgnht.cumsum()/(np.arange(T+1)+1.0), alpha=0.75, linewidth=3 )
  #pp.plot( XIsghmc,  alpha=0.75,linewidth=3 )
  pp.legend( ["SG-NHT","SG-NHT -- cummean"])
  pp.title( "XI")
  #
  # pp.subplot(2,1,2)
  # pp.plot( XIsghmc, alpha=0.75, linewidth=3 )
  # pp.plot( XIsghmc.cumsum()/(np.arange(T+1)+1.0), alpha=0.75, linewidth=3 )
  # #pp.plot( XIsghmc,  alpha=0.75,linewidth=3 )
  # pp.legend( ["SG-NHT","SG-HMC -- cummean"])
  pp.title( "XI")
  
  p=problem
  b=p["bins"]
  td = p["bin_density"]
  sd = p["sample_density"]
  
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

    
  times = np.array([1,10,100,1000,10000,100000])
  errors = []
  for t in times:
    
    if len( Qsgnht ) >= t:
      at = sd( Qsgnht[:t], b )
      bt = sd( Qsghmc[:t], b )
      errors.append([0.5*np.sum( np.abs(td-at) ),0.5*np.sum( np.abs(td-bt) )])
  
  errors = np.array(errors)
  nt = len(errors)
  
  pp.figure(3)
  pp.clf()
  pp.semilogx( times[:nt], errors, 'o-' )
  
  # pp.figure(4)
  # pp.clf()
  # pp.subplot(3,1,1)
  # pp.plot( b[:-1]+0.5*(b[1]-b[0]),td, 'k--',lw=1,alpha=0.75 )
  # pp.plot( b[:-1]+0.5*(b[1]-b[0]),sample_pdf_nht_1, 'g-',lw=1,alpha=0.75 )
  # pp.plot( b[:-1]+0.5*(b[1]-b[0]),sample_pdf_hmc_1, 'r-',lw=1,alpha=0.75 )
  # pp.subplot(3,1,2)
  # pp.plot( b[:-1]+0.5*(b[1]-b[0]),td, 'k--',lw=1,alpha=0.75 )
  # pp.plot( b[:-1]+0.5*(b[1]-b[0]),sample_pdf_nht_2, 'g-',lw=1,alpha=0.75 )
  # pp.plot( b[:-1]+0.5*(b[1]-b[0]),sample_pdf_hmc_2, 'r-',lw=1,alpha=0.75 )
  # pp.subplot(3,1,3)
  # pp.plot( b[:-1]+0.5*(b[1]-b[0]),td, 'k--',lw=1,alpha=0.75 )
  # pp.plot( b[:-1]+0.5*(b[1]-b[0]),sample_pdf_nht_3, 'g-',lw=1,alpha=0.75 )
  # pp.plot( b[:-1]+0.5*(b[1]-b[0]),sample_pdf_hmc_3, 'r-',lw=1,alpha=0.75 )
  
  # print "@ 100 error hmc = ", np.sqrt( np.mean( (td-sample_pdf_hmc_1)**2 ) )
  # print "@ 1000 error hmc = ", np.sqrt( np.mean( (td-sample_pdf_hmc_2)**2 ) )
  # print "@ 10000 error hmc = ", np.sqrt( np.mean( (td-sample_pdf_hmc_3)**2 ) )
  #
  # print "@ 100 error nht = ", np.sqrt( np.mean( (td-sample_pdf_nht_1)**2 ) )
  # print "@ 1000 error nht = ", np.sqrt( np.mean( (td-sample_pdf_nht_2)**2 ) )
  # print "@ 10000 error nht = ", np.sqrt( np.mean( (td-sample_pdf_nht_3)**2 ) )
  
  #pp.legend(["True","NHT","HMC"])
  # pp.subplot(2,2,2)
#   pp.plot( Psgnht, alpha=0.75, linewidth=3 )
#   pp.plot( Psghmc,  alpha=0.75,linewidth=3 )
#   pp.legend( ["SG-NHT","SG-HMC"])
#   pp.title( "momentum")
#
#   pp.subplot(2,2,3)
#   pp.plot( Qsgnht, alpha=0.75, linewidth=3 )
#   pp.plot( Qsghmc,  alpha=0.75,linewidth=3 )
#   pp.legend( ["SG-NHT","SG-HMC"])
#   pp.title( "position")
#
#   pp.subplot(2,2,4)
#   pp.plot( problem["K"](Psgnht), alpha=0.75, linewidth=3 )
#   pp.plot( problem["K"](Psghmc),  alpha=0.75,linewidth=3 )
#   pp.legend( ["SG-NHT","SG-HMC"])
#   pp.title( "K(p)")
  
  
  # pp.figure(3)
  # pp.clf()
  # pp.subplot(3,1,1)
  # pp.plot( problem["U"](Qsgnht).cumsum()/( np.arange(T+1)+1.0), alpha=0.75, linewidth=3 )
  # pp.plot( problem["U"](Qsghmc).cumsum()/( np.arange(T+1)+1.0),  alpha=0.75,linewidth=3 )
  # pp.legend( ["SG-NHT","SG-HMC"])
  # pp.title( "U(q)")
  # pp.subplot(3,1,2)
  # pp.plot(problem["K"](Psgnht).cumsum()/( np.arange(T+1)+1.0), alpha=0.75, linewidth=3)
  # pp.plot(problem["K"](Psghmc).cumsum()/( np.arange(T+1)+1.0), alpha=0.75, linewidth=3)
  # pp.hlines( 0.5, 0,T+1)
  # #pp.plot( problem["K"](Psgnht), alpha=0.75, linewidth=3 )
  # #pp.plot( problem["K"](Psghmc),  alpha=0.75,linewidth=3 )
  # pp.legend( ["cum SG-NHT","cum SG-HMC"])
  # pp.title( "K(p)")
  # pp.subplot(3,1,3)
  # pp.plot( problem["H"](Qsgnht,Psgnht).cumsum()/( np.arange(T+1)+1.0), alpha=0.75, linewidth=3 )
  # pp.plot( problem["H"](Qsghmc,Psghmc).cumsum()/( np.arange(T+1)+1.0),  alpha=0.75,linewidth=3 )
  # pp.legend( ["SG-NHT","SG-HMC"])
  # pp.title( "H(q,p)")
  
  # pp.figure(4)
  # pp.clf()
  # pp.subplot(2,1,1)
  # pp.plot( problem["U"](Qsgnht), problem["K"](Psgnht), "o", alpha=0.25 )
  # pp.plot( problem["U"](Qsghmc), problem["K"](Psghmc),"o", alpha=0.25 )
  # pp.xlabel("U(q)")
  # pp.xlabel("K(p)")
  # pp.legend( ["SG-NHT","SG-HMC"])
  # pp.subplot(2,1,2)
  # pp.plot( Qsgnht, Psgnht, "o", alpha=0.25 )
  # pp.plot( Qsghmc, Psghmc, "o", alpha=0.25 )
  # pp.xlabel("q")
  # pp.xlabel("p")
  # pp.legend( ["SG-NHT","SG-HMC"])
  #pp.plot( problem["theta_range"](), problem["true_posterior"](problem["theta_range"]()),  'k--', lw=4)
  #pp.hist( Qleap, 20,histtype='step', normed=True, alpha=0.5, linewidth=4 )
  #pp.hist( Qleap_no_mh, 20,histtype='step', normed=True, alpha=0.5, linewidth=4 )
  #pp.hist( Qmeul, 50,histtype='step',normed=True, alpha=0.5, linewidth=4 )
  #pp.hist( Qeul, 50,histtype='step',normed=True, alpha=0.5, linewidth=4 )
  #pp.legend( ["True","Leap + mh","Leap no mh"])
  #pp.legend( ["True","Leap","mEuler"])
  #pp.legend( ["True","Leap","mEuler","euler"])
  pp.show()
  
  