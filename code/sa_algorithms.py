import numpy as np
import pylab as pp
import scipy as sp
from numpy import newaxis
import pdb

def sgradient( g, w, N, batchsize, q = 1, ids = None, l1 = 0, l2=0):
  # g: the gradient of the objective function, e.g. grad of loglikelihood
  # w: parameters at this time step, len p
  # N: total nbr data vectors
  # batchsize: nbr of function evals to use in f
  
  p = len(w)
  
  
  if ids is None:
    ids = np.random.permutation( N )[:batchsize]
  
  grads = g( w, ids )    
  
  return grads, ids

def spsa_gradient_with_hessian( f, w, c, c_tilde, N, batchsize, q = 1, ids = None, batchreplace = 1.0, mask = None, l1 = 0, l2=0, g_var=None):
  # f: the objective function, e.g. loglikelihood
  # w: parameters at this time step, len p
  # c: step size, constant for all dimensions
  # N: total nbr data vectors
  # batchsize: nbr of function evals to use in f
  # q: nbr of repeats for the gradient
  # batchreplace: percent of ids to replace
  
  c_adjust = c*np.ones(len(w))
  
  #pdb.set_trace()
  p = len(w)
  
  g = np.zeros( p )
  h = np.zeros( p )
  
  if ids is None:
    ids = np.random.permutation( N )[:batchsize]
      
  G = np.zeros( (q,p))
  for j in range(q):
    # TODO: could put different batch ids here
    # ids=new_ids
    
    # TODO: uncomment for perturbed c
    # if np.random.randn()<0:
    #   c_adjust *= 1 + np.random.rand()
    # else:
    #   c_adjust /= 1 + np.random.rand()
        
    mask = 2*np.random.binomial(1,0.5,p)-1
    mask_tilde = 2*np.random.binomial(1,0.5,p)-1
  
    f_plus  = f( w + c_adjust*mask, ids = ids, l1=l1, l2=l2 )
    f_minus = f( w - c_adjust*mask, ids = ids, l1=l1, l2=l2 )
    
    f_plus_1  = f( w + c_adjust*mask + c_tilde*mask_tilde, ids = ids, l1=l1, l2=l2 )
    f_minus_1 = f( w - c_adjust*mask + c_tilde*mask_tilde, ids = ids, l1=l1, l2=l2 )
  
    g += (f_plus-f_minus)/(2*c_adjust*mask)
    
    # ones-sided grads
    g1plus  = (f_plus-f_plus_1)/(c_tilde*mask_tilde)
    g1minus = (f_minus-f_minus_1)/(c_tilde*mask_tilde)
    
    delta_g = (g1plus - g1minus)/(2*c_adjust*mask)
    
    h += delta_g
    
    #pdb.set_trace()
    G[j] = (f_plus-f_minus)/(2*c_adjust*mask)
  
  g /= q  
  h /= q
  
  #V = pow( G - g, 2).mean(0)   
  
  #g/=V
  #pdb.set_trace()
  return g, h, ids

def spsa_abc_gradient( f, w, c, q = 1, seed = None, mask = None, hessian=False, c_tilde=None ):
  # f: the objective function, e.g. loglikelihood
  # w: parameters at this time step, len p
  # c: step size, constant for all dimensions
  # q: nbr of repeats for the gradient
  
  c_adjust = c#*np.ones(len(w))
  p = len(w)
  
  g = np.zeros( p )
  if hessian:
    h = np.zeros( (p,p))
  G = np.zeros( (q,p))
  for j in range(q):
    # TODO: uncomment for perturbed c
    # if np.random.randn()<0:
    #   c_adjust *= 1 + np.random.rand()
    # else:
    #   c_adjust /= 1 + np.random.rand()
        
    mask = 2*np.random.binomial(1,0.5,p)-1
  
    #pdb.set_trace()
    f_plus  = f( w + c_adjust*mask, seed = seed )
    f_minus = f( w - c_adjust*mask, seed = seed )
  
    g += (f_plus-f_minus)/(2*c_adjust*mask)
    #print g
    G[j] = (f_plus-f_minus)/(2*c_adjust*mask)
    
    if np.isinf( f_plus ) or np.isinf( f_minus ):
      pdb.set_trace()
    
    if hessian:
      mask_tilde = 2*np.random.binomial(1,0.5,p)-1
      
      # ones-sided grads
      f_plus_1  = f( w + c_adjust*mask + c_tilde*mask_tilde, seed = seed )
      f_minus_1 = f( w - c_adjust*mask + c_tilde*mask_tilde, seed = seed )
      
      g1plus  = (f_plus-f_plus_1)/(c_tilde*mask_tilde)
      g1minus = (f_minus-f_minus_1)/(c_tilde*mask_tilde)
    
      delta_g = (g1plus - g1minus)
      #/(2*c_adjust*mask)
    
      h1 = delta_g.reshape( (1,p) ) / (c_adjust*mask).reshape( (p,1) )
      h += 0.5*(h1+h1.T)
      #pdb.set_trace()
      
    
  
  g/=q  
  #print g
  if hessian:
    h/=q  
  else:
    h=None
  
  if seed is not None:
    new_seed = seed + 1
  else:
    new_seed = None
  return g, h, new_seed
    
def spsa_gradient( f, w, c, N, batchsize, q = 1, ids = None, batchreplace = 1.0, mask = None, l1 = 0, l2=0, g_var=None):
  # f: the objective function, e.g. loglikelihood
  # w: parameters at this time step, len p
  # c: step size, constant for all dimensions
  # N: total nbr data vectors
  # batchsize: nbr of function evals to use in f
  # q: nbr of repeats for the gradient
  # batchreplace: percent of ids to replace
  
  c_adjust = c #*np.ones(len(w))
  #if g_var is not None:
  #  c_adjust = len(w)*c*(np.sqrt(g_var)+1e-3)/(np.sqrt(g_var).sum()+len(w)*1e-3)
  #  #c_adjust = len(w)*c*g_var/(g_var.sum())
  
  #pdb.set_trace()
  p = len(w)
  
  g = np.zeros( p )
  
  if ids is None:
    ids = np.random.permutation( N )[:batchsize]
  # else:
  #   nbr_replace = int(batchreplace*batchsize)
  #   new_ids = np.random.permutation( N )[:nbr_replace]
  #   ids_in_batch = np.random.permutation( batchsize )
  #   for i, idx in zip( range(nbr_replace), ids_in_batch[:nbr_replace]):
  #     ids[idx] = new_ids[i]
      
  G = np.zeros( (q,p))
  for j in range(q):
    # TODO: could put different batch ids here
    # ids=new_ids
    
    # TODO: uncomment for perturbed c
    # if np.random.randn()<0:
    #   c_adjust *= 1 + np.random.rand()
    # else:
    #   c_adjust /= 1 + np.random.rand()
        
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
  verbose_rate = 100
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
    #adj_grad = g_hat / (0 + np.sqrt(g_var) )  
    #w = w - alpha*g_hat #g_mom
    w = w - alpha*g_mom
    #w = w - alpha*adj_grad
    
    alpha *= gamma
    c     *= gamma
    #alpha=c
    if np.mod(t+1,verbose_rate)==0:
      train_error = problem.train_error( w )
      test_error = problem.test_error( w )
      print "%4d error %0.4f cost %0.4f  alpha %0.4f"%(t+1, train_error, test_error,alpha)
      errors.append( [train_error,test_error])
    
  return w, np.array(errors)

def spall_abc( w, params ):
  problem   = params["ml_problem"]
  recorder  = params["recorder"]
  max_iters = params["max_iters"]
  q         = params["q"] # the nbr of repeats for spsa
  c0         = params["c"]
  alpha0     = params["alpha"]
  gamma     = params["gamma"]
  mom       = params["mom"]
  init_seed = params["init_seed"]
  verbose_rate = params["verbose_rate"]
  #assert alpha0 < c0, "must have alpha < c"
  hessian=params["hessian"]
  h_delta = params["h_delta"]
  
  p = len(w)
  if hessian:
    h_bar         = np.zeros( (p,p))
    h_bar_bar     = np.zeros( (p,p) )
    h_bar_bar_inv = np.zeros( (p,p) )
  
  c=c0
  alpha=alpha0
  init_cost = problem.train_cost( w )
  train_error = problem.train_error( w )
  test_error = problem.test_error( w )
  ids = None
  errors = [[train_error,test_error]]
  g_var = np.ones(len(w))
  seed = init_seed
  for t in xrange(max_iters):
    c_tilde = 1.15*c
    g_hat, h_hat, seed = spsa_abc_gradient( problem.train_cost, w, c, q=q, seed=seed, hessian= False, c_tilde=c_tilde )
    # if t < 50:
    #   g_hat, h_hat, seed = spsa_abc_gradient( problem.train_cost, w, c, q=q, seed=seed, hessian= False, c_tilde=c_tilde )
    # else:
    #   g_hat, h_hat, seed = spsa_abc_gradient( problem.train_cost, w, c, q=q, seed=seed, hessian= hessian, c_tilde=c_tilde )
    
    g_hat += problem.grad_prior( w )
    
    if t==0:
      g_mom = g_hat.copy()
    else:
      g_mom = mom*g_mom + (1-mom)*g_hat
    
    if t==0:
      g_var = pow( g_hat, 2 )
    else:
      #g_var = mom*g_var + (1-mom)*pow( (g_hat), 2 )
      g_var = g_var + pow( (g_hat), 2 )
      
    if hessian:
      pass
      # r = float(t)/float(t+1)
      # if t-50 > 0:
      #   r =0.25*mom
      #
      # else:
      #   r=0
      # if h_hat is not None:
      #   h_bar         = r*h_bar + (1-r)*h_hat
      #   h_bar_diag    = np.diag( np.diag(h_bar))
      #   h_bar_bar     = pow(np.dot(h_bar,h_bar.T),0.5) +h_delta*np.eye(p)
      #   #h_bar_bar     = h_bar + h_delta*np.eye(p)
      #   #pdb.set_trace()
      #   h_bar_bar_inv = h_delta*np.linalg.inv( h_bar_bar  )
      #
      # if t < 50:
      #   #print "using gradient"
      #   w = w - alpha*g_mom
      # else:
      #   #print "using hessian"
      #   w = w - alpha*np.dot( h_bar_bar_inv, g_mom )
      #pdb.set_trace()
    else:  
      #w = w - alpha*g_mom
      #w = w - 1000*alpha*g_mom/(1e-3 + np.sqrt(g_var) )  
      #w = w - g_hat/(1e-3 + np.sqrt(g_var) ) 
      w = w - g_mom/(1e-3 + np.sqrt(g_var) )  
      #w = w - 0.5*alpha*g_hat/(1e-3 + np.sqrt(g_var) )  + np.sqrt(alpha)*np.random.randn(p)
      #pdb.set_trace()
    #w = w - alpha*adj_grad
    
    problem.model.current.response_groups[0].epsilon *= gamma
    
    alpha *= gamma
    c     *= gamma
    
    train_error = problem.train_error( w )
    test_error = problem.test_error( w )
    errors.append( [train_error,test_error])
    if np.mod(t+1,verbose_rate)==0:
      print "%4d train %0.4f test %0.4f  alpha %g  "%(t+1, train_error, test_error,alpha), problem.model.current.theta
    
  return w, np.array(errors)
    
def spall_with_hessian( w, params ):
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
  diag_add =  params["diag_add"]
  #assert alpha0 < c0, "must have alpha < c"
  
  c=c0
  alpha=alpha0
  train_error = problem.train_error( w )
  test_error = problem.test_error( w )
  ids = None
  errors = [[train_error,test_error]]
  g_var = np.ones(len(w))
  
  p=len(w)
  H_bar = np.zeros( p )
  
  
  for t in xrange(max_iters):
    c_tilde = 0.5*c
    
    g_hat, h_hat, ids = spsa_gradient_with_hessian( problem.train_cost, w, c, c_tilde, N, batchsize, q=q, ids=None, batchreplace=batchreplace, l1=l1,l2=l2, g_var=g_var )
    
    if t==0:
      H_bar = h_hat
    else:
      H_bar = float(t/(t+1))*H_bar + float(1/(t+1))*h_hat
    
    H_double_bar = H_bar + diag_add
    Hinv = 1.0 / H_double_bar
    
    if t==0:
      g_mom = g_hat.copy()
    else:
      g_mom = mom*g_mom + (1-mom)*g_hat
      
    if t==0:
      g_var = pow( g_hat, 2 )
    else:
      g_var = mom*g_var + (1-mom)*pow( g_hat, 2 )
    
    #g_hat = g_mom
    #adj_grad = g_hat / (0 + np.sqrt(g_var) )  
    #w = w - alpha*Hinv*g_hat  #g_mom
    w = w - alpha*g_mom
    #w = w - alpha*adj_grad
    
    alpha *= gamma
    c     *= gamma
    #alpha=c
    if np.mod(t+1,100)==0:
      train_error = problem.train_error( w )
      test_error = problem.test_error( w )
      print "%4d train %0.4f test %0.4f  alpha %0.4f  c %0.4f  gamma %0.4f"%(t+1, train_error, test_error,alpha,c,gamma)
      errors.append( [train_error,test_error])
    
  return w, np.array(errors)

    
    
def sgd( w, params ):
  problem   = params["ml_problem"]
  max_iters = params["max_iters"]
  q         = params["q"] # the nbr of repeats for spsa
  c0         = params["c"]
  N         = params["N"]
  batchsize = params["batchsize"]
  alpha0     = params["alpha"]
  gamma     = params["gamma"]
  mom       = params["mom"]
  #batchreplace = params["batchreplace"]
  l1       = params["l1"]
  l2       = params["l2"]
  bcompare = params["compare"]
  #assert alpha0 < c0, "must have alpha < c"
  
  c=c0
  alpha=alpha0
  train_error = problem.train_error( w )
  test_error = problem.test_error( w )
  ids = None
  errors = [[train_error,test_error]]
  g_var = np.ones(len(w))
  for t in xrange(max_iters):
    g_hat, ids = sgradient( problem.gradient, w, N, batchsize, ids=None )
    
    if bcompare:
      spsa_g_hat, ids = spsa_gradient( problem.train_cost, w, c, N, batchsize, q=q, ids=ids )
      
      rads = np.arccos( np.dot(g_hat,spsa_g_hat)/(np.linalg.norm(spsa_g_hat)*np.linalg.norm(g_hat)))
      degrees = np.rad2deg(rads)
      
      print "angle in degrees : ", degrees
      
    if t==0:
      g_mom = g_hat.copy()
    else:
      g_mom = mom*g_mom + (1-mom)*g_hat
      
    if t==0:
      g_var = pow( g_hat, 2 )
    else:
      g_var = mom*g_var + (1-mom)*pow( g_hat, 2 )
    
    #g_hat = g_mom
    #adj_grad = g_mom / (1e-3 + np.sqrt(g_var) )  
    w = w - alpha*g_mom
    #w = w - alpha*adj_grad
    
    alpha *= gamma
    #alpha=c
    if np.mod(t+1,100)==0:
      train_error = problem.train_error( w )
      test_error = problem.test_error( w )
      print "%4d train %0.4f test %0.4f  alpha %0.4f  gamma %0.4f"%(t+1, train_error, test_error,alpha,gamma)
      errors.append( [train_error,test_error])
    
  return w, np.array(errors)