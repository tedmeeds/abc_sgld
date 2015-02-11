import numpy as np
import pylab as pp

from sghmc_problems import *
from sg import *


    
    
if __name__ == "__main__":
  pp.close('all')
  T             = 5
  L             = 20
  epsilon       = 0.1
  current_q     = np.random.randn()
  mh_correction = True
  step_method   = leapfrog_step
  
  injected_gradient_noise_std = 2.0
  
  problem = generate_sghmc_fig1_problem(noise=injected_gradient_noise_std)
  problem["other_params"]={"C":6,"V":injected_gradient_noise_std**2}

  Qleap,Pleap = run_hmc_with_problem( T, problem, leapfrog_step, mh_correction, current_q, epsilon, L  )
  Qleap_no_mh,Pleap_no_mh = run_hmc_with_problem( T, problem, leapfrog_step, False, current_q, epsilon, L  )
  Qsghmc,Psghmc = run_hmc_with_problem( T, problem, sghmc_step, False, current_q, epsilon, L  )
  Qsgnht,Psgnht = run_hmc_with_problem( T, problem, sgnht_step, False, current_q, epsilon, L  )
  #Qmeul,Pmeul = run_hmc_with_problem( T, problem, mod_euler_step, mh_correction, current_q, epsilon, L  )
  #Qeul,Peul = run_hmc_with_problem( T, problem, euler_step, mh_correction, current_q, epsilon, L  )
  
  pp.figure(1)
  pp.clf()
  pp.plot( problem["theta_range"](), problem["true_posterior"](problem["theta_range"]()),  'k--', lw=4)
  pp.hist( Qleap, 20,histtype='step', normed=True, alpha=0.75, linewidth=3 )
  pp.hist( Qleap_no_mh, 20,histtype='step', normed=True, alpha=0.75, linewidth=3 )
  pp.hist( Qsghmc, 20,histtype='step', normed=True, alpha=0.75, linewidth=3 )
  pp.hist( Qsgnht, 20,histtype='step', normed=True, alpha=0.75, linewidth=3 )
  #pp.hist( Qmeul, 50,histtype='step',normed=True, alpha=0.5, linewidth=4 )
  #pp.hist( Qeul, 50,histtype='step',normed=True, alpha=0.5, linewidth=4 )
  pp.legend( ["True","Leap + mh","Leap no mh","sghmc","sgnht"])
  #pp.legend( ["True","Leap","mEuler"])
  #pp.legend( ["True","Leap","mEuler","euler"])
  pp.show()
  
  