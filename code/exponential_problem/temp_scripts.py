from abcpy.problems.blowfly.blowfly    import BlowflyProblem   as Problem
from abcpy.problems.blowfly.blowfly    import default_params   as load_default_params
from abcpy.plotting import *

import pdb
import numpy as np
import pylab as pp

# exponential distributed observations with Gamma(alpha,beta) prior over lambda
problem_params = load_default_params()
problem_params["q_factor"] = 0.1
problem = Problem( problem_params, force_init = True )

if __name__ == "__main__":
  figsize = (16,12)
  #dpi = 800
  
  
  
  
  params = load_default_params()
  p = Problem( problem_params, force_init = True )
  
  # log_P       = theta[0]
  # log_delta   = theta[1]
  # log_N0      = theta[2]
  # log_sigma_d = theta[3]
  # log_sigma_p = theta[4]
  # tau         = theta[5]
  
  bin_ranges = [(-10,3),(-10,3),(2,32)]
  
  bin_ranges = [(-3,7),(-6,6),(2,10)]
  theta_names = [r'$\log P$', r'$\log \delta$', r'$\log N_0$', r'$\log \sigma_d$',r'\log \sigma_p',r'$\tau']
  #theta_names = ["A","B","C","D","E","F"]
  
  colors = ["b","g","r","b","g","r"]
  theta_ids =   [0,1,2]
  #theta_ids =   [3,4,5]
  #names = [r"REJ $\epsilon 5$",r"p-SL S50",r"GPS $\xi 0.3$"]
  names = ["REJ","SL","GPS"]
  #names = ["REJ 10","REJ 7.5","REJ 5"]
  #names = [r"m-SL 2 $\xi$ 0.1",r"ASL 2 $\xi$ 0.2",r"ASL 2 $\xi$ 0.3"]
  #names = [r"m-SL 2",r"m-SL 10",r"m-SL 50"]
  
  thetas1 = np.loadtxt("./uai2014/runs/blowfly/rejection/eps5p0_repeat1_thetas.txt")
  thetas2 = np.loadtxt("./uai2014/runs/blowfly_eps/sl_pseudo_s10/just_gaussian_repeat2_thetas.txt")
  thetas3 = np.loadtxt("./uai2014/runs/blowfly/gps2/xi0p4_eps0p5_ninit50_repeat1_thetas.txt")
  # 
  # sims1 = np.loadtxt("./uai2014/runs/blowfly/rejection/eps5p0_repeat1_sims.txt")
  # sims2 = np.loadtxt("./uai2014/runs/blowfly_eps/sl_pseudo_s10/just_gaussian_repeat2_sims.txt")
  # sims3 = np.loadtxt("./uai2014/runs/blowfly/gps2/xi0p4_eps0p5_ninit50_repeat1_sims.txt")

  # thetas1 = np.loadtxt("./uai2014/runs/blowfly/rejection/eps10p0_repeat1_thetas.txt")
  # thetas2 = np.loadtxt("./uai2014/runs/blowfly/rejection/eps7p5_repeat1_thetas.txt")
  # thetas3 = np.loadtxt("./uai2014/runs/blowfly/rejection/eps5p0_repeat1_thetas.txt")
  # 
  # sims1 = np.loadtxt("./uai2014/runs/blowfly/rejection/eps10p0_repeat1_sims.txt")
  # sims2 = np.loadtxt("./uai2014/runs/blowfly/rejection/eps7p5_repeat1_sims.txt")
  # sims3 = np.loadtxt("./uai2014/runs/blowfly/rejection/eps5p0_repeat1_sims.txt")
  # thetas1 = np.loadtxt("./uai2014/runs/blowfly_eps/asl_pseudo_s2_ds5/xi0p1_just_gaussian_repeat2_thetas.txt")
  # thetas2 = np.loadtxt("./uai2014/runs/blowfly_eps/asl_pseudo_s5_ds5/xi0p1_just_gaussian_repeat2_thetas.txt")
  # thetas3 = np.loadtxt("./uai2014/runs/blowfly_eps/asl_pseudo_s10_ds5/xi0p1_just_gaussian_repeat2_thetas.txt")
  # thetas11 = np.loadtxt("./uai2014/runs/blowfly_eps/asl_pseudo_s2_ds5/xi0p1_just_gaussian_repeat3_thetas.txt")
  # thetas22 = np.loadtxt("./uai2014/runs/blowfly_eps/asl_pseudo_s5_ds5/xi0p1_just_gaussian_repeat3_thetas.txt")
  # thetas33 = np.loadtxt("./uai2014/runs/blowfly_eps/asl_pseudo_s10_ds5/xi0p1_just_gaussian_repeat3_thetas.txt")
  #   
  # thetas1 = np.loadtxt("./uai2014/runs/blowfly_eps/sl_pseudo_s2/just_gaussian_repeat2_thetas.txt")
  # thetas2 = np.loadtxt("./uai2014/runs/blowfly_eps/sl_pseudo_s10/just_gaussian_repeat2_thetas.txt")
  # thetas3 = np.loadtxt("./uai2014/runs/blowfly_eps/sl_pseudo_s50/just_gaussian_repeat2_thetas.txt")
  # thetas11 = np.loadtxt("./uai2014/runs/blowfly_eps/sl_pseudo_s2/just_gaussian_repeat1_thetas.txt")
  # thetas22 = np.loadtxt("./uai2014/runs/blowfly_eps/sl_pseudo_s10/just_gaussian_repeat2_thetas.txt")
  # thetas33 = np.loadtxt("./uai2014/runs/blowfly_eps/sl_pseudo_s50/just_gaussian_repeat3_thetas.txt")

  # thetas1 = np.loadtxt("./uai2014/runs/blowfly_eps/sl_marginal_s2/just_gaussian_repeat2_thetas.txt")
  # thetas2 = np.loadtxt("./uai2014/runs/blowfly_eps/sl_marginal_s10/just_gaussian_repeat2_thetas.txt")
  # thetas3 = np.loadtxt("./uai2014/runs/blowfly_eps/sl_marginal_s50/just_gaussian_repeat2_thetas.txt")
  # thetas11 = np.loadtxt("./uai2014/runs/blowfly_eps/sl_marginal_s2/just_gaussian_repeat1_thetas.txt")
  # thetas22 = np.loadtxt("./uai2014/runs/blowfly_eps/sl_marginal_s10/just_gaussian_repeat1_thetas.txt")
  # thetas33 = np.loadtxt("./uai2014/runs/blowfly_eps/sl_marginal_s50/just_gaussian_repeat1_thetas.txt")  
  # 
  #sims1 = np.loadtxt("./uai2014/runs/blowfly/rejection/eps10p0_repeat1_sims.txt")
  #sims2 = np.loadtxt("./uai2014/runs/blowfly/rejection/eps7p5_repeat1_sims.txt")
  #sims3 = np.loadtxt("./uai2014/runs/blowfly/rejection/eps5p0_repeat1_sims.txt")
  # thetas1 = np.vstack( (thetas1,thetas11))
  # thetas2 = np.vstack( (thetas2,thetas22))
  # thetas3 = np.vstack( (thetas3,thetas33))
  thetas =[thetas1,thetas2,thetas3]

  pp.rc('text', usetex=True)
  pp.rc('font', family='serif')
  f = pp.figure( figsize=figsize )
  pp.close("all")
  spid = 1
  burnin =1000
  row_id = 0
  col_id = 0
  for th,nm in zip( thetas, names):
    
    for theta_id, br in zip( theta_ids, bin_ranges):
      #sp = f.add_subplot( 3, 3, spid )
      sp=pp.subplot(3,3,spid)
      pp.hist( th[burnin:,theta_id], 30, range=br, normed=True, alpha=0.75, color=colors[col_id])
      mu_theta = th[burnin:,theta_id].mean()
      std_theta = th[burnin:,theta_id].std()
      
      #ax = pp.axis()
      #pp.vlines(mu_theta,0,ax[3] )
      
      #   
      # pp.plot( y / 1000.0, "k-", lw=3, alpha=1 )
      # pp.plot( x / 1000.0, "r-", lw=3, alpha=0.75 )
      #   
      # pp.legend(["observation","generated"],loc=1,fancybox=True,prop={'size':18})
      # 
      # pp.ylabel( "N/1000")
      # pp.xlabel("Time")
      # pp.title("Blowfly Problem")
      
      pp.xlim( br )
      if theta_id == theta_ids[0]:
        pp.ylabel( r'$p( \theta | y )$' )
        #sp.yaxis.label.set_rotation(0)
      if theta_id == theta_ids[1]:
        pp.legend( [nm], loc=1,fancybox=True,prop={'size':12} )
        
      
      if nm == names[-1]:
        pp.xlabel( theta_names[theta_id] )
      
        
      set_tick_fonsize( sp, 16 )
      set_title_fonsize( sp, 36 )
      set_label_fonsize( sp, 24 )
      spid+=1
    col_id+=1
  pp.suptitle( r"Blowfly Problem: posterior samples $p( \theta | y )$", fontsize=24)  
  #print "observed stats:", p.obs_statistics
  #print "generated stats:", p.statistics_function(x)
  #pp.savefig( "blowfly_problem.png", format="png", dpi=600,bbox_inches="tight")
  pp.savefig("/Users/uvapostdoc/Dropbox/tedmeeds-share-with-max/uai2014-GPSABC/blowfly_theta_given_y.pdf", format="pdf", dpi=600) #,bbox_inches="tight")
  #pp.savefig("/Users/uvapostdoc/Dropbox/tedmeeds-share-with-max/uai2014-GPSABC/blowfly_theta_given_y_pSL_extra_stats.pdf", format="pdf", dpi=600) #,bbox_inches="tight")
  pp.show()