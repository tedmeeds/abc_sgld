import numpy as np
import pylab as pp
import scipy as sp
from numpy import newaxis
import pdb

def spsa_gradient( f, w, c, N, batchsize, q = 1, ids = None, batchreplace = 1.0, mask = None, l1 = 0, l2=0, g_var=None):
  # f: the objective function, e.g. loglikelihood
  # w: parameters at this time step, len p
  # c: step size, constant for all dimensions
  # N: total nbr data vectors
  # batchsize: nbr of function evals to use in f
  # q: nbr of repeats for the gradient
  # batchreplace: percent of ids to replace
  
  c_adjust = c*np.ones(len(w))
  #if g_var is not None:
  #  c_adjust = len(w)*c*(np.sqrt(g_var)+1e-3)/(np.sqrt(g_var).sum()+len(w)*1e-3)
  #  #c_adjust = len(w)*c*g_var/(g_var.sum())
  
  #pdb.set_trace()
  p = len(w)
  
  g = np.zeros( p )
  
  if ids is None:
    ids = np.random.permutation( N )[:batchsize]
  else:
    nbr_replace = int(batchreplace*batchsize)
    new_ids = np.random.permutation( N )[:nbr_replace]
    ids_in_batch = np.random.permutation( batchsize )
    for i, idx in zip( range(nbr_replace), ids_in_batch[:nbr_replace]):
      ids[idx] = new_ids[i]
      
  G = np.zeros( (q,p))
  for j in range(q):
    # TODO: could put different batch ids here
    # ids=new_ids
    
    mask = 2*np.random.binomial(1,0.5,p)-1
  
    f_plus  = f( w + c_adjust*mask, ids = ids, l1=l1, l2=l2 )
    f_minus = f( w - c_adjust*mask, ids = ids, l1=l1, l2=l2 )
  
    g += (f_plus-f_minus)/(2*c_adjust*mask)
    G[j] = (f_plus-f_minus)/(2*c_adjust*mask)
  
  g/=q  
  
  #V = pow( G - g, 2).mean(0)   
  
  #g/=V
  #pdb.set_trace()
  return g, ids
  
  # GM = (1-mom)*GM + mom*G
  #
  # self.W = self.W + lr*GM
 #
 # historical_grad += g^2
 # adjusted_grad = grad / (fudge_factor + sqrt(historical_grad))
 # w = w - master_stepsize*adjusted_grad
  
def spall( w, params ):
  problem   = params["ml_problem"]
  max_iters = params["max_iters"]
  q         = params["q"] # the nbr of repeats for spsa
  c0         = params["c"]
  N         = params["N"]
  batchsize = params["batchsize"]
  alpha0     = params["alpha"]
  gamma     = params["gamma"]
  mom       = params["mom"]
  batchreplace = params["batchreplace"]
  l1       = params["l1"]
  l2       = params["l2"]
  
  #assert alpha0 < c0, "must have alpha < c"
  
  c=c0
  alpha=alpha0
  train_error = problem.train_error( w )
  test_error = problem.test_error( w )
  ids = None
  errors = [[train_error,test_error]]
  g_var = np.ones(len(w))
  for t in xrange(max_iters):
    g_hat, ids = spsa_gradient( problem.train_cost, w, c, N, batchsize, q=q, ids=None, batchreplace=batchreplace, l1=l1,l2=l2, g_var=g_var )
    
    if t==0:
      g_mom = g_hat.copy()
    else:
      g_mom = mom*g_mom + (1-mom)*g_hat
      
    if t==0:
      g_var = pow( g_hat, 2 )
    else:
      g_var = mom*g_var + (1-mom)*pow( g_hat, 2 )
    
    #g_hat = g_mom
    adj_grad = g_hat / (0 + np.sqrt(g_var) )  
    w = w + alpha*g_hat #g_mom
    #w = w + alpha*adj_grad
    
    alpha *= gamma
    c     *= gamma
    #alpha=c
    if np.mod(t+1,100)==0:
      train_error = problem.train_error( w )
      test_error = problem.test_error( w )
      print "%4d train %0.4f test %0.4f  alpha %0.4f  c %0.4f  gamma %0.4f"%(t+1, train_error, test_error,alpha,c,gamma)
      errors.append( [train_error,test_error])
    
  return w, np.array(errors)
    