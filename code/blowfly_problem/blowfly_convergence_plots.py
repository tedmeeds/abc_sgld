from abcpy.problems.blowfly.blowfly    import BlowflyProblem   as Problem
from abcpy.problems.blowfly.blowfly    import default_params   as load_default_params
from abcpy.plotting import *

import pdb
import numpy as np
import pylab as pp

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
    
    convs.append( pow( y - mean, 2 )/pow(y_star,2) )
    
  convs = np.array(convs)
  C = convs[intervals,:]
  return C

def compute_convergence( p, y_star, th, sm, sts, burnin, intervals, ignore_stats = False ):
  t = th[burnin:,:]
  s = sm[burnin:]
  
  have_stats = False
  if len(sts) == len(th) and ignore_stats == False:
    sts = sts[burnin:,:]
    have_stats = True
    print "Have stats!"
  else:
    print "Bummer, no stats!"
    
  cs = np.cumsum(s)
  
  N = len(t)
  
  idx = 0
  times = []
  convs = []
  sims = []
  statss = []
  
  count = 0.0
  mean  = np.zeros(len(y_star))
  while idx < N:
    
    theta = t[idx]
    
    if have_stats:
      stats = sts[idx,:]
    else:
      outs = p.simulation_function(theta)
      stats = p.statistics_function(outs)
    
    mean = count*mean + stats
    count+=1
    mean /= count
    
    convs.append( pow( y_star - mean, 2 )/pow(y_star,2) )
    sims.append( cs[idx] )
    times.append( idx)
    statss.append(stats)
    
    idx += 1
  
  convs = np.array(convs)
  sims  = np.array(sims)
  times = np.array(times)
  statss = np.array(statss)
  
  ids = []
  for i in intervals:
    if i == -1:
      ids.append(-1)
    else:
      j = pp.find( times == i-1 )
      if len(j)>0:
        ids.append(j[0])
  convs = convs[ids,:]
  sims  = sims[ids]
  times = times[ids]
  
  return convs, times, sims, statss
  
def compute_all( p, y_star, thetas1, sims1, stats1, burnin, intervals, ignore_stats = False ):
  CONV = []
  TIMES = []
  SMS = []
  STATS = []
  for th,sm, sts in zip( thetas1, sims1, stats1 ):
    print "running"
    conv, times, sms,stats = compute_convergence( p, y_star, th, sm, sts, burnin, intervals, ignore_stats )
    #assert False 
    #conv, times, sms = compute_convergence( y_star, th, sm, burnin, every )
    CONV.append(conv)
    TIMES.append(times)
    SMS.append(sms)
    STATS.append(stats)
  CONV = np.array(CONV)
  TIMES = np.array(TIMES)
  SMS = np.array(SMS)
  STATS = np.array(STATS)
  return CONV, TIMES, SMS, STATS

def load_all( name, repeats ):
  thetas1 = []
  sims1   = []
  stats1  = []
  for i in range(repeats):
    print "loading "
    thetas1.append( np.loadtxt("%s%d_thetas.txt"%(name,i+1)) )
    sims1.append( np.loadtxt("%s%d_sims.txt"%(name,i+1)) )
    try:
      #stats1.append([])
      stats1.append( np.loadtxt("%s%d_stats.txt"%(name,i+1)) )
    except:
      stats1.append([])
      
  return thetas1, sims1, stats1

def add_to_plots(CONV,TIMES,SMS, stats, colr, name ):
  MU_TIMES = TIMES.mean(0)
  MU_SIMS  = SMS.mean(0)
  
  f = pp.figure( 1, figsize=(12,8) ) 
  spid = 1
  for stat_id in stats:
    sp = f.add_subplot(3,3,spid)
    #m = CONV[:,:,stat_id].mean(0)
    m = CONV[:,stat_id].mean(0)
    #u_stdev = CONV[:,:,stat_id].mean(0) + 2*CONV[:,:,stat_id].std(0)
    #l_stdev = CONV[:,:,stat_id].mean(0)
    pp.loglog( MU_TIMES, m, colr, lw=3, alpha=0.75 )
    set_title_fonsize( sp, 8 )
    set_label_fonsize( sp, 8 )
    
    spid += 1
  
  f = pp.figure( 2, figsize=figsize ) 
  spid = 1
  for stat_id in stats:
    sp = f.add_subplot(3,3,spid)
    #m = CONV[:,:,stat_id].mean(0)
    m = CONV[:,stat_id].mean(0)
    #u_stdev = CONV[:,:,stat_id].mean(0) + 2*CONV[:,:,stat_id].std(0)
    #l_stdev = CONV[:,:,stat_id].mean(0)
    pp.loglog( MU_SIMS, m, colr, lw=3, alpha=0.75 )
    set_title_fonsize( sp, 8 )
    set_label_fonsize( sp, 8 )
    
    spid += 1
          
if __name__ == "__main__":
  
  params = load_default_params()
  p = Problem( params, force_init = True )
  
  figsize = (9,6)
  thetas1,sims1,stats1 = load_all( "./uai2014/runs/blowfly/gps2/xi0p3_eps0p0_ninit50_repeat", 3 )
  thetas2,sims2,stats2 = load_all( "./uai2014/runs/blowfly_eps/asl_pseudo_s5_ds5/xi0p1_just_gaussian_repeat", 5 )

  thetas3,sims3,stats3 = load_all( "./uai2014/runs/blowfly/rejection/eps5p0_repeat", 5 )
  thetas4,sims4,stats4 = load_all( "./uai2014/runs/blowfly_eps/sl_pseudo_s10/just_gaussian_repeat", 5 )
  thetas5,sims5,stats5 = load_all( "./uai2014/runs/blowfly_eps/sl_pseudo_s50/just_gaussian_repeat", 5 )
  thetas6,sims6,stats6 = load_all( "./uai2014/runs/blowfly_eps/asl_pseudo_s5_ds5/xi0p3_just_gaussian_repeat", 5 )

  # explicit fix for GPS --first 50 do not count as samples, just simulations
  # for t,s,st in sims4:
  #   s[1:] = 10.0
    

  # explicit fix for SL -- miscounted sims
  for s in sims4:
    s[1:] = 10.0
    
  burnin = 100
  intervals = np.array([1,10,100,1000,-1])
  
  y_star = p.obs_statistics
  
  C1,T1,S1,ST1 = compute_all( p, y_star, thetas1, sims1, stats1, burnin, intervals )
  C2,T2,S2,ST2 = compute_all( p, y_star, thetas2, sims2, stats2, burnin, intervals, ignore_stats=True )
  C3,T3,S3,ST3 = compute_all( p, y_star, thetas3, sims3, stats3, burnin, intervals )
  C4,T4,S4,ST4 = compute_all( p, y_star, thetas4, sims4, stats4, burnin, intervals, ignore_stats=True )
  C5,T5,S5,ST5 = compute_all( p, y_star, thetas5, sims5, stats5, burnin, intervals, ignore_stats=True )
  C6,T6,S6,ST6 = compute_all( p, y_star, thetas6, sims6, stats6, burnin, intervals, ignore_stats=True )

  pp.close("all")
  add_to_plots( C1,T1,S1, [0,1,2,3,4,5,6,7,8], "b", "GPS" )
  add_to_plots( C2,T2,S2, [0,1,2,3,4,5,6,7,8], "r--", "ASL 0.1" )
  add_to_plots( C3,T3,S3, [0,1,2,3,4,5,6,7,8], "g", "REJ" )
  add_to_plots( C4,T4,S4, [0,1,2,3,4,5,6,7,8], "k", "SL10" )
  add_to_plots( C5,T5,S5, [0,1,2,3,4,5,6,7,8], "k--", "SL50" )
  add_to_plots( C6,T6,S6, [0,1,2,3,4,5,6,7,8], "r", "ASL 0.3" )
  
  f1 = pp.figure(1)
  
  pp.suptitle("Blowfly: convergence per sample")
  
  f2 = pp.figure(2)
  pp.suptitle("Blowfly: convergence per simulation")
  
  # pp.close("all")
#   f = pp.figure( figsize=figsize )
#   
#   sp = f.add_subplot(111)
#   
#   
#   pp.legend(["observation","generated"],loc=1,fancybox=True,prop={'size':18})
# 
#   pp.ylabel( "N/1000")
#   pp.xlabel("Time")
#   pp.title("Blowfly Problem")
#   set_tick_fonsize( sp, 12 )
#   set_title_fonsize( sp, 14 )
#   set_label_fonsize( sp, 14 )
#     
#   print "observed stats:", p.obs_statistics
#   print "generated stats:", p.statistics_function(x)
  #pp.savefig( "blowfly_problem.png", format="png", dpi=600,bbox_inches="tight")
  #pp.savefig( "/Users/tmeeds/Dropbox/tedmeeds-share-with-max/uai2014-GPSABC/blowfly_problem.png", format="png", dpi=600,bbox_inches="tight")
  pp.show()