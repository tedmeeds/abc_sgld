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

# def spsa_gradient_with_hessian( f, w, c, c_tilde, N, batchsize, q = 1, ids = None, batchreplace = 1.0, mask = None, l1 = 0, l2=0, g_var=None):
#   # f: the objective function, e.g. loglikelihood
#   # w: parameters at this time step, len p
#   # c: step size, constant for all dimensions
#   # N: total nbr data vectors
#   # batchsize: nbr of function evals to use in f
#   # q: nbr of repeats for the gradient
#   # batchreplace: percent of ids to replace
#
#   c_adjust = c*np.ones(len(w))
#
#   #pdb.set_trace()
#   p = len(w)
#
#   g = np.zeros( p )
#   h = np.zeros( p )
#
#   if ids is None:
#     ids = np.random.permutation( N )[:batchsize]
#
#   G = np.zeros( (q,p))
#   for j in range(q):
#     # TODO: could put different batch ids here
#     # ids=new_ids
#
#     # TODO: uncomment for perturbed c
#     # if np.random.randn()<0:
#     #   c_adjust *= 1 + np.random.rand()
#     # else:
#     #   c_adjust /= 1 + np.random.rand()
#
#     mask = 2*np.random.binomial(1,0.5,p)-1
#     mask_tilde = 2*np.random.binomial(1,0.5,p)-1
#
#     f_plus  = f( w + c_adjust*mask, ids = ids, l1=l1, l2=l2 )
#     f_minus = f( w - c_adjust*mask, ids = ids, l1=l1, l2=l2 )
#
#     f_plus_1  = f( w + c_adjust*mask + c_tilde*mask_tilde, ids = ids, l1=l1, l2=l2 )
#     f_minus_1 = f( w - c_adjust*mask + c_tilde*mask_tilde, ids = ids, l1=l1, l2=l2 )
#
#     g += (f_plus-f_minus)/(2*c_adjust*mask)
#
#     # ones-sided grads
#     g1plus  = (f_plus-f_plus_1)/(c_tilde*mask_tilde)
#     g1minus = (f_minus-f_minus_1)/(c_tilde*mask_tilde)
#
#     delta_g = (g1plus - g1minus)/(2*c_adjust*mask)
#
#     h += delta_g
#
#     #pdb.set_trace()
#     G[j] = (f_plus-f_minus)/(2*c_adjust*mask)
#
#   g /= q
#   h /= q
#
#   #V = pow( G - g, 2).mean(0)
#
#   #g/=V
#   #pdb.set_trace()
#   return g, h, ids

def finite_difference_abc_gradient( f, w, c, seed = None ):
  # f: the objective function, e.g. loglikelihood
  # w: parameters at this time step, len p
  # c: step size, constant for all dimensions
  # q: nbr of repeats for the gradient

  p = len(w)
  g = np.zeros( p )
  
  for i in range(p):
    
    mask = np.zeros(p)
    mask[i] = 1.0
    
    #pdb.set_trace()
    f_plus  = f( w + c*mask, seed = seed )
    f_minus = f( w - c*mask, seed = seed )

    g[i] = (f_plus-f_minus)/(2*c)

    #print "   ",i,"f's  ", f_plus, f_minus, f_minus-f_plus, 2*c, g[i]
    if np.isinf( f_plus ) or np.isinf( f_minus ) or np.isnan( f_plus ) or np.isnan( f_minus ):
      pdb.set_trace()

  h = None
  if seed is not None:
    new_seed = seed + 1
  else:
    new_seed = None
  return g, new_seed


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
  
  masks = 2*np.random.binomial(1,0.5,(q,p))-1
  
  for j in range(q):
    # TODO: uncomment for perturbed c
    #if np.random.randn()<0:
    #  c_adjust *= 1 + 0.25*np.random.rand()
    #else:
    #  c_adjust /= 1 + 0.25*np.random.rand()

    mask = masks[j]

    #pdb.set_trace()
    f_plus  = f( w + c_adjust*mask, seed = seed )
    f_minus = f( w - c_adjust*mask, seed = seed )

    #seed += 1
    #print f_plus, f_minus, seed
    g += (f_plus-f_minus)/(2*c_adjust*mask)
    #print g
    G[j] = (f_plus-f_minus)/(2*c_adjust*mask)

    if np.isinf( f_plus ) or np.isinf( f_minus ):
      pdb.set_trace()


    # if hessian:
    #   mask_tilde = 2*np.random.binomial(1,0.5,p)-1
    #
    #   # ones-sided grads
    #   f_plus_1  = f( w + c_adjust*mask + c_tilde*mask_tilde, seed = seed )
    #   f_minus_1 = f( w - c_adjust*mask + c_tilde*mask_tilde, seed = seed )
    #
    #   g1plus  = (f_plus-f_plus_1)/(c_tilde*mask_tilde)
    #   g1minus = (f_minus-f_minus_1)/(c_tilde*mask_tilde)
    #
    #   delta_g = (g1plus - g1minus)
    #   #/(2*c_adjust*mask)
    #
    #   h1 = delta_g.reshape( (1,p) ) / (c_adjust*mask).reshape( (p,1) )
    #   h += 0.5*(h1+h1.T)
    #   #pdb.set_trace()
      

  g/=q  
  
  if q > 1:
    Vw = np.sum( (G - g)**2, 0 )/(q-1)
    nVw = np.sum(Vw)/q
    ng  = np.dot(g.T,g)
  #pdb.set_trace()

    
    #pdb.set_trace()
    # print "norm 1 Vw/q        = ", nVw
    # print "   norm g          = ", ng
    # print "   ratio (theta^2) = ", nVw / ng
    # print "   theta           = ", np.sqrt( nVw / ng )
    # sgld_eps = nVw/4.0
    # print "   sgld            = ", sgld_eps
    
  #print g
  if hessian:
    h/=q
  else:
    h=None

  if seed is not None:
    new_seed = seed + 1
  else:
    new_seed = None
  return g, h, new_seed, Vw
    
def spsa_gradient( f, w, c, N, batchsize, q = 1, ids = None):
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
    
    f_plus  = f( w + c_adjust*mask, ids = ids )
    f_minus = f( w - c_adjust*mask, ids = ids )
  
    g += (f_plus-f_minus)/(2*c_adjust*mask)
    G[j] = (f_plus-f_minus)/(2*c_adjust*mask)

  g/=q

  return g, None,ids

def spall( w, params ):
  problem   = params["ml_problem"]
  max_iters = params["max_iters"]
  q         = params["q"] # the nbr of repeats for spsa
  c0         = params["c"]
  N         = params["N"]
  batchsize = params["batchsize"]
  alpha0     = params["alpha"]

  gamma_alpha     = params["gamma_alpha"]
  gamma_c     = params["gamma_c"]
  gamma_eps     = params["gamma_eps"]
  mom_beta1       = params["mom_beta1"]
  mom_beta2       = params["mom_beta2"]
  update_method   = params["update_method"]
  
  mom_beta1       = params["mom_beta1"]
  mom_beta2       = params["mom_beta2"]
  
  batchreplace    = params["batchreplace"]
  verbose_rate    = 100
  update_method   = params["update_method"]
  #assert alpha0 < c0, "must have alpha < c"

  c=c0
  alpha=alpha0
  train_error = problem.train_error( w )
  test_error = problem.test_error( w )
  ids = None
  errors = [[train_error,test_error]]
  
  g_squared = np.ones(len(w))
  
  for t in xrange(max_iters):
    g_hat, h_hat,ids = spsa_gradient( problem.train_cost, w, c, N, batchsize, q=q, ids=None )
    
    g_hat += problem.grad_prior( w )
    
    if t==0:
      g_mom = g_hat.copy()
    else:
      g_mom = mom_beta1*g_mom + (1-mom_beta1)*g_hat
      
    if t==0:
      g_squared = pow( g_hat, 2 )
    else:
      #g_var = mom*g_var + (1-mom)*pow( g_hat, 2 )
      if update_method == "adagrad":
        g_squared = g_squared + g_hat**2
      elif update_method == "rmsprop" or update_method == "adam":
        g_squared = mom_beta2*g_squared + (1-mom_beta2)*g_hat**2
    
    #print "before ",w
    #print "    g_mom: ", g_mom
    #print "    g_sqr: ",g_squared
    if update_method == "grad":
      w = w - alpha*g_mom
      
    elif update_method == "adagrad" or update_method == "rmsprop":
      w = w - alpha*g_mom / (1e-3 + np.sqrt(g_squared) ) 
      
    elif update_method == "adam":
      gamma_adam = np.sqrt( 1.0-(1-mom_beta2)**(t+1)) / ( 1.0-(1-mom_beta1)**(t+1))
      #pdb.set_trace()
      w = w - alpha*g_mom*gamma_adam / (1e-3 + np.sqrt(g_squared) )     
    
    w = problem.fix_w( w )
    alpha *= gamma_alpha
    c     *= gamma_c

    if np.mod(t+1,verbose_rate)==0:
      train_error = problem.train_error( w )
      test_error = problem.test_error( w )
      print "%4d error %0.4f cost %0.4f  alpha %0.4f"%(t+1, train_error, test_error,alpha)
      errors.append( [train_error,test_error])

  return w, np.array(errors)

def spall_sgld( w, params ):
  problem   = params["ml_problem"]
  max_iters = params["max_iters"]
  q         = params["q"] # the nbr of repeats for spsa
  c0         = params["c"]
  N         = params["N"]
  batchsize = params["batchsize"]
  alpha0     = params["alpha"]

  gamma_alpha     = params["gamma_alpha"]
  gamma_c     = params["gamma_c"]
  gamma_eps     = params["gamma_eps"]
  mom_beta1       = params["mom_beta1"]
  mom_beta2       = params["mom_beta2"]
  update_method   = params["update_method"]
  
  mom_beta1       = params["mom_beta1"]
  mom_beta2       = params["mom_beta2"]
  
  batchreplace    = params["batchreplace"]
  verbose_rate    = 100
  update_method   = params["update_method"]
  #assert alpha0 < c0, "must have alpha < c"

  c=c0
  alpha=alpha0
  train_error = problem.train_error( w )
  test_error = problem.test_error( w )
  ids = None
  errors = [[train_error,test_error]]
  
  g_squared = np.ones(len(w))
  
  for t in xrange(max_iters):
    g_hat, h_hat,ids = spsa_gradient( problem.train_cost, w, c, N, batchsize, q=q, ids=None )
    
    g_hat += problem.grad_prior( w )
    
    if t==0:
      g_mom = g_hat.copy()
    else:
      g_mom = mom_beta1*g_mom + (1-mom_beta1)*g_hat
      
    if t==0:
      g_squared = pow( g_hat, 2 )
    else:
      #g_var = mom*g_var + (1-mom)*pow( g_hat, 2 )
      if update_method == "adagrad":
        g_squared = g_squared + g_hat**2
      else:
        g_squared = mom_beta2*g_squared + (1-mom_beta2)*g_hat**2
      #elif update_method == "rmsprop" or update_method == "adam":
      #  g_squared = mom_beta2*g_squared + (1-mom_beta2)*g_hat**2
    
    print "before ",w
    print "    g_mom: ", g_mom
    print "    g_sqr: ",g_squared
    if update_method == "grad":
      w = w - 0.5*alpha*g_mom - np.sqrt(alpha+0.5*alpha*g_mom*g_squared)*np.random.randn( len(w) )
      #print w
      
    elif update_method == "adagrad" or update_method == "rmsprop":
      w = w - 0.5*alpha*g_mom / (1e-3 + np.sqrt(g_squared) ) - np.sqrt(alpha)*np.random.randn( len(w) )
      
    elif update_method == "adam":
      gamma_adam = np.sqrt( 1.0-(1-mom_beta2)**(t+1)) / ( 1.0-(1-mom_beta1)**(t+1))
      #pdb.set_trace()
      w = w - 0.5*alpha*g_mom*gamma_adam / (1e-3 + np.sqrt(g_squared) ) - np.sqrt(alpha)*np.random.randn( len(w) )
    
    w = problem.fix_w( w )
    
    alpha *= gamma_alpha
    c     *= gamma_c

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
  #gamma     = params["gamma"]
  gamma_alpha     = params["gamma_alpha"]
  gamma_c     = params["gamma_c"]
  gamma_eps     = params["gamma_eps"]
  mom_beta1       = params["mom_beta1"]
  mom_beta2       = params["mom_beta2"]
  update_method   = params["update_method"]
  init_seed = params["init_seed"]
  verbose_rate = params["verbose_rate"]
  sgld_alpha = params["sgld_alpha"]
  
  max_steps = params["max_steps"]

  q_rate  =params["q_rate"]
  q0=q
  p = len(w)
  # if hessian:
  #   h_bar         = np.zeros( (p,p))
  #   h_bar_bar     = np.zeros( (p,p) )
  #   h_bar_bar_inv = np.zeros( (p,p) )
  
  c=c0
  alpha=alpha0
  init_cost = problem.train_cost( w )
  train_error = problem.train_error( w )
  test_error = problem.test_error( w )
  ids = None
  errors = [[train_error,test_error]]
  g_squared = np.zeros(len(w))
  seed = init_seed
  
  others = []
  param_noise   = []
  injected_noise = []
  
  for t in xrange(max_iters):
    c_tilde = 1.15*c
    
    g_hat, h_hat, seed, Vj = spsa_abc_gradient( problem.train_cost, w, c, q=q, seed=seed, hessian= False, c_tilde=c_tilde )
    
    q = int(q0*pow( q_rate, t ) )
    #print "q=",q
    g_hat += problem.grad_prior( w )

    if t==0:
      g_mom = g_hat.copy()
    else:
      g_mom = mom_beta1*g_mom + (1-mom_beta1)*g_hat
      #g_mom = mom*g_mom + (1-mom)*g_hat

    if t==0:
      g_squared = pow( g_hat, 2 )
    else:
      if update_method == "adagrad":
        g_squared = g_squared + g_hat**2
      elif update_method == "rmsprop" or update_method == "adam":
        #g_squared = mom_beta2*g_squared + (1-mom_beta2)*(g_hat-g_mom)**2
        g_squared = mom_beta2*g_squared + (1-mom_beta2)*g_hat**2
    
    #print "before ",w
    #print "    g_mom: ", g_mom
    #print "    g_sqr: ",g_squared
    if update_method == "grad":
      sgld_criteria = alpha*Vj/(4*q)
      I = pp.find( sgld_criteria > sgld_alpha )
      if len(I)>0:
        vio = np.sum( sgld_criteria[I] - sgld_alpha )
      else:
        vio=0
        
      loglik_noise = alpha*alpha*Vj/(4*q)
      #injected_noise = sgld_epsilon
      
      param_noise.append( loglik_noise )
      injected_noise.append( alpha*np.ones(len(loglik_noise)) )
      others.append([loglik_noise.mean(),alpha,len(I),vio,np.sum( sgld_criteria - sgld_alpha ), np.sum(Vj)/(q*np.dot(g_hat.T,g_hat))])
      
      nrm = np.linalg.norm( alpha*g_mom)
      if nrm > max_steps.mean():
        print "reducing nrm"
        g_mom = alpha*max_steps.mean()*g_mom/(nrm)
        #pdb.set_trace()
      # for j in range(len(g_mom)):
      #   if np.abs( alpha*g_mom[j] ) > max_steps[j]:
      #     print "reducing %d"%(j)
      #     g_mom[j] = max_steps[j]/alpha
      w = w - alpha*g_mom
      
    elif update_method == "adagrad" or update_method == "rmsprop":
      w = w - alpha*g_mom / (1e-3 + np.sqrt(g_squared) ) 
      
    elif update_method == "adam":
      gamma_adam = np.sqrt( 1.0-(1-mom_beta2)**(t+1)) / ( 1.0-(1-mom_beta1)**(t+1))
      adam_alpha = alpha*gamma_adam/(1e-3 + np.sqrt(g_squared) )
      adam_epsilon = 2*adam_alpha
      
      #print "ADAM alpha: ",  adam_alpha
      #sgld_criteria = np.dot( adam_epsilon.T, Vj)/(4*q)
      sgld_criteria = adam_epsilon * Vj/float(4*q)
      #sgld_epsilon = 2.0*alpha*g_mom*gamma_adam / (1e-3 + np.sqrt(g_squared) )
      
      I = pp.find( sgld_criteria > sgld_alpha )
      if len(I)>0:
        vio = np.sum( sgld_criteria[I] - sgld_alpha )
      else:
        vio=0
        
      loglik_noise = adam_epsilon*adam_epsilon*Vj/(4*q)
      #injected_noise = sgld_epsilon
      
      param_noise.append( loglik_noise )
      injected_noise.append( adam_epsilon )
      others.append([loglik_noise.mean(),adam_epsilon.mean(),len(I),vio,np.sum( sgld_criteria - sgld_alpha ), np.sum(Vj)/(q*np.dot(g_hat.T,g_hat))])
      if np.mod(t+1,verbose_rate)==0:
        print "ADAM alpha: ",  alpha*gamma_adam/(1e-3 + np.sqrt(g_squared) ) 
      
      w = w - alpha*g_mom*gamma_adam / (1e-3 + np.sqrt(g_squared) ) 
    
    w = problem.fix_w( w )
    #print "after ",w
    
    # if t < 200:
    #   alpha *= gamma_alpha
    # else:
    #   alpha *= 0.995
    alpha *= gamma_alpha
    c     *= gamma_c
    
    problem.model.current.response_groups[0].epsilon *= gamma_eps
    
    train_error = problem.train_error( w )
    test_error = problem.test_error( w )
    errors.append( [train_error,test_error])
    if np.mod(t+1,verbose_rate)==0:
      print "%4d train %0.4f test %0.4f  alpha %g  "%(t+1, train_error, test_error,alpha), problem.model.current.theta

  return w, np.array(errors), ( np.array(others), np.array(param_noise), np.array(injected_noise) )

def spall_abc_sgld( w, params ):
  problem   = params["ml_problem"]
  recorder  = params["recorder"]
  max_iters = params["max_iters"]
  q         = params["q"] # the nbr of repeats for spsa
  c0         = params["c"]
  alpha0     = params["alpha"]
  #gamma     = params["gamma"]
  gamma_alpha     = params["gamma_alpha"]
  gamma_c     = params["gamma_c"]
  gamma_eps     = params["gamma_eps"]
  mom_beta1       = params["mom_beta1"]
  mom_beta2       = params["mom_beta2"]
  update_method   = params["update_method"]
  init_seed = params["init_seed"]
  verbose_rate = params["verbose_rate"]
  
  sgld_alpha = params["sgld_alpha"]
  max_steps = params["max_steps"]

  q_rate  =params["q_rate"]
  q0=q
  p = len(w)
  # if hessian:
  #   h_bar         = np.zeros( (p,p))
  #   h_bar_bar     = np.zeros( (p,p) )
  #   h_bar_bar_inv = np.zeros( (p,p) )
  
  c=c0
  alpha=alpha0
  init_cost = problem.train_cost( w )
  train_error = problem.train_error( w )
  test_error = problem.test_error( w )
  ids = None
  errors = [[train_error,test_error]]
  
  g_squared = np.zeros(len(w))
  seed = init_seed
  
  others = []
  param_noise   = []
  injected_noise = []
  
  for t in xrange(max_iters):
    c_tilde = 1.15*c
    
    g_hat, seed = finite_difference_abc_gradient( problem.train_cost, w, c, seed=seed )
    
    q = int(q0*pow( q_rate, t ) )
    
    g_hat += problem.grad_prior( w )

    if t==0:
      g_mom = g_hat.copy()
    else:
      g_mom = mom_beta1*g_mom + (1-mom_beta1)*g_hat
      #g_mom = mom*g_mom + (1-mom)*g_hat

    if t==0:
      g_squared = pow( g_hat, 2 )
    else:
      if update_method == "adagrad":
        g_squared = g_squared + g_hat**2
      elif update_method == "rmsprop" or update_method == "adam":
        g_squared = mom_beta2*g_squared + (1-mom_beta2)*g_hat**2
      else:
        g_squared = mom_beta2*g_squared + (1-mom_beta2)*(g_hat-g_mom)**2
    
    #sgld_epsilon = 2.0*alpha
    
    # print "before ",w
    # print "    g_mom: ", g_mom
    # print "    g_sqr: ",g_squared
    if update_method == "grad":
      
      
      sgld_criteria = alpha*Vj/(4*q)
      I = pp.find( sgld_criteria > sgld_alpha )
      if len(I)>0:
        vio = np.sum( sgld_criteria[I] - sgld_alpha )
      else:
        vio=0
        
      loglik_noise = alpha*alpha*Vj/(4*q)
      #injected_noise = sgld_epsilon
      
      param_noise.append( loglik_noise )
      injected_noise.append( alpha*np.ones(len(loglik_noise)) )
      others.append([loglik_noise.mean(),alpha,len(I),vio,np.sum( sgld_criteria - sgld_alpha ), np.sum(Vj)/(q*np.dot(g_hat.T,g_hat))])
      
      
      
      
      if np.mod(t+1,verbose_rate)==0:
        print "sgld criteria = ", alpha*Vj/(4*q), "nbr violators = ", len(I), vio, "byrd criteria = ", np.sum(Vj)/(q*np.dot(g_hat.T,g_hat))  
      
      #w = w - 0.5*alpha*g_mom - np.sqrt(alpha)*np.random.randn( len(w) )
      w = w - 0.5*alpha*g_mom + np.sqrt(alpha+0.25*alpha*alpha*g_squared)*np.random.randn( len(w) )
      
      #pdb.set_trace()
            
    elif update_method == "adagrad" or update_method == "rmsprop":
      w = w - 0.5*alpha*g_mom / (1e-3 + np.sqrt(g_squared) ) - np.sqrt(alpha)*np.random.randn( len(w) )
      
    elif update_method == "adam":
      #print "BROKEN!"
      gamma_adam = np.sqrt( 1.0-(1-mom_beta2)**(t+1)) / ( 1.0-(1-mom_beta1)**(t+1))
      adam_alpha = alpha*gamma_adam/(1e-3 + np.sqrt(g_squared) )
      adam_epsilon = 2*adam_alpha
      
      #print "ADAM alpha: ",  adam_alpha
      #sgld_criteria = np.dot( adam_epsilon.T, Vj)/(4*q)
      sgld_criteria = adam_epsilon * Vj/float(4*q)
      #sgld_epsilon = 2.0*alpha*g_mom*gamma_adam / (1e-3 + np.sqrt(g_squared) )
      
      I = pp.find( sgld_criteria > sgld_alpha )
      if len(I)>0:
        vio = np.sum( sgld_criteria[I] - sgld_alpha )
      else:
        vio=0
        
      loglik_noise = adam_epsilon*adam_epsilon*Vj/(4*q)
      #injected_noise = sgld_epsilon
      
      param_noise.append( loglik_noise )
      injected_noise.append( adam_epsilon )
      others.append([loglik_noise.mean(),adam_epsilon.mean(),len(I),vio,np.sum( sgld_criteria - sgld_alpha ), np.sum(Vj)/(q*np.dot(g_hat.T,g_hat))])
      
      
      
      #I = pp.find( sgld_criteria > sgld_alpha )
      if np.mod(t+1,verbose_rate)==0:
        print "ADAM alpha: ",  adam_alpha
        print "sgld criteria = ", sgld_criteria, "nbr violators = ", len(I)
        print "seed ",seed
      
      
      #pdb.set_trace()
      #w = w - 0.5*adam_epsilon*g_mom - np.sqrt(adam_epsilon)*np.random.randn( len(w) )
    
      w = w - 0.5*adam_epsilon*g_mom - np.sqrt(adam_epsilon)*np.random.randn( len(w) )
    
    w = problem.fix_w( w )
    
    #print "after ",w
    alpha *= gamma_alpha
    alpha = max(alpha, alpha0/10)
    c     *= gamma_c
    
    problem.model.current.response_groups[0].epsilon *= gamma_eps
    
    train_error = problem.train_error( w )
    test_error = problem.test_error( w )
    errors.append( [train_error,test_error])
    if np.mod(t+1,verbose_rate)==0:
      print "%4d train %0.4f test %0.4f  alpha %g  "%(t+1, train_error, test_error,alpha), problem.model.current.theta

  return w, np.array(errors), ( np.array(others), np.array(param_noise), np.array(injected_noise) )

def spall_abc_sgnht( w, params ):
  problem   = params["ml_problem"]
  recorder  = params["recorder"]
  max_iters = params["max_iters"]
  q         = params["q"] # the nbr of repeats for spsa
  c0         = params["c"]
  alpha0     = params["alpha"]
  #gamma     = params["gamma"]
  gamma_alpha     = params["gamma_alpha"]
  gamma_c     = params["gamma_c"]
  gamma_eps     = params["gamma_eps"]
  mom_beta1       = params["mom_beta1"]
  mom_beta2       = params["mom_beta2"]
  update_method   = params["update_method"]
  init_seed = params["init_seed"]
  verbose_rate = params["verbose_rate"]
  
  A = params["A"]
  
  sgld_alpha = params["sgld_alpha"]
  max_steps = params["max_steps"]

  q_rate  =params["q_rate"]
  q0=q
  p = len(w)
  # if hessian:
  #   h_bar         = np.zeros( (p,p))
  #   h_bar_bar     = np.zeros( (p,p) )
  #   h_bar_bar_inv = np.zeros( (p,p) )
  
  c=c0
  alpha=alpha0
  init_cost = problem.train_cost( w )
  train_error = problem.train_error( w )
  test_error = problem.test_error( w )
  ids = None
  errors = [[train_error,test_error]]
  
  g_squared = np.zeros(len(w))
  seed = init_seed
  
  others = []
  param_noise   = []
  injected_noise = []
  
  xi = A
  h  = alpha0
  nw = len(w)
  p  = np.random.randn( nw )
  
  u_stream = np.zeros( max_iters, dtype=int )
  u_stream2 = np.zeros( max_iters, dtype=int )
  tau = params["seed_tau"]
  c   = 0
  #np.random.seed(4)
  for t in xrange(max_iters):
    if c+1 == tau:
      u_stream[t] = np.random.randint(5*max_iters)
      c = 0
    else:
      u_stream[t] = u_stream[t-1]
      c += 1
    u_stream2[t] = np.random.randint(10*max_iters)
    
  c=c0  
  for t in xrange(max_iters):
    #print "t = ",t
    g_hat, seed = finite_difference_abc_gradient( problem.train_cost, w, c, seed=u_stream[t] )
    np.random.seed( u_stream2[t] )
    
    g_hat += problem.grad_prior( w )
    grad_U = -g_hat
    #print "grad_U: ",grad_U
    p = p - xi*p*h - grad_U*h + np.sqrt(2.0*A*h)*np.random.randn(nw)
    #print "p: ",p
    w = w + p*h
    w = problem.fix_w( w )
    
    #print "w: ",w
    xi = xi + h*(np.dot( p.T, p )/float(nw) - 1.0)
      
    if np.mod(t+1,verbose_rate)==0:
      print "xi: ",  xi
      
    
    
    #print "after ",w
    alpha *= gamma_alpha
    alpha = max(alpha, alpha0/10)
    c     *= gamma_c
    
    problem.model.current.response_groups[0].epsilon *= gamma_eps
    
    train_error = problem.train_error( w )
    test_error = problem.test_error( w )
    errors.append( [train_error,test_error])
    if np.mod(t+1,verbose_rate)==0:
      print "%4d train %0.4f test %0.4f  alpha %g  "%(t+1, train_error, test_error,alpha), problem.model.current.theta

  return w, np.array(errors), ( np.array(others), np.array(param_noise), np.array(injected_noise) )
  
def spall_abc_ld( w, params ):
  problem   = params["ml_problem"]
  recorder  = params["recorder"]
  max_iters = params["max_iters"]
  q         = params["q"] # the nbr of repeats for spsa
  c0         = params["c"]
  alpha0     = params["alpha"]
  #gamma     = params["gamma"]
  gamma_alpha     = params["gamma_alpha"]
  gamma_c     = params["gamma_c"]
  gamma_eps     = params["gamma_eps"]
  mom_beta1       = params["mom_beta1"]
  mom_beta2       = params["mom_beta2"]
  update_method   = params["update_method"]
  init_seed = params["init_seed"]
  verbose_rate = params["verbose_rate"]
  
  sgld_alpha = params["sgld_alpha"]
  max_steps = params["max_steps"]

  q_rate  =params["q_rate"]
  q0=q
  p = len(w)
  # if hessian:
  #   h_bar         = np.zeros( (p,p))
  #   h_bar_bar     = np.zeros( (p,p) )
  #   h_bar_bar_inv = np.zeros( (p,p) )
  
  c=c0
  alpha=alpha0
  init_cost = problem.train_cost( w )
  train_error = problem.train_error( w )
  test_error = problem.test_error( w )
  ids = None
  errors = [[train_error,test_error]]
  
  g_squared = np.zeros(len(w))
  seed = init_seed
  
  others = []
  param_noise   = []
  injected_noise = []
  
  
  
  for t in xrange(max_iters):
    seed = t
    
    current_loglik = problem.train_cost( w, seed=seed )
    
    
    g_hat, seed = finite_difference_abc_gradient( problem.train_cost, w, c, seed=seed )
    seed = t
    
    g_hat += problem.grad_prior( w )

    if t==0:
      g_mom = g_hat.copy()
    else:
      g_mom = mom_beta1*g_mom + (1-mom_beta1)*g_hat

    if t==0:
      g_squared = pow( g_hat, 2 )
    else:
      if update_method == "adagrad":
        g_squared = g_squared + g_hat**2
      elif update_method == "rmsprop" or update_method == "adam":
        g_squared = mom_beta2*g_squared + (1-mom_beta2)*g_hat**2
      else:
        g_squared = mom_beta2*g_squared + (1-mom_beta2)*(g_hat-g_mom)**2
    
    if update_method == "grad":
      delta_w = 0.5*alpha*alpha*g_mom + alpha*np.random.randn( len(w) )
      
            
    elif update_method == "adagrad" or update_method == "rmsprop":
      delta_w = 0.5*alpha*alpha*g_mom / (1e-3 + np.sqrt(g_squared) ) - alpha*np.random.randn( len(w) )
      
      
    elif update_method == "adam":
      #print "BROKEN!"
      gamma_adam = np.sqrt( 1.0-(1-mom_beta2)**(t+1)) / ( 1.0-(1-mom_beta1)**(t+1))
      adam_alpha = alpha*gamma_adam/(1e-3 + np.sqrt(g_squared) )
      adam_epsilon = 2*adam_alpha
      
      #print "ADAM alpha: ",  adam_alpha
      #sgld_criteria = np.dot( adam_epsilon.T, Vj)/(4*q)
      #sgld_criteria = adam_epsilon * Vj/float(4*q)
      #sgld_epsilon = 2.0*alpha*g_mom*gamma_adam / (1e-3 + np.sqrt(g_squared) )
      
      #I = pp.find( sgld_criteria > sgld_alpha )
      #if len(I)>0:
      #  vio = np.sum( sgld_criteria[I] - sgld_alpha )
      #else:
      #  vio=0
        
      #loglik_noise = adam_epsilon*adam_epsilon*Vj/(4*q)
      #injected_noise = sgld_epsilon
      
      #param_noise.append( loglik_noise )
      #injected_noise.append( adam_epsilon )
      #others.append([loglik_noise.mean(),adam_epsilon.mean(),len(I),vio,np.sum( sgld_criteria - sgld_alpha ), np.sum(Vj)/(q*np.dot(g_hat.T,g_hat))])
      
      
      
      #I = pp.find( sgld_criteria > sgld_alpha )
      #if np.mod(t+1,verbose_rate)==0:
      #  print "ADAM alpha: ",  adam_alpha
      #  print "sgld criteria = ", sgld_criteria, "nbr violators = ", len(I)
      #  print "seed ",seed
      
      
      #pdb.set_trace()
      #w = w - 0.5*adam_epsilon*g_mom - np.sqrt(adam_epsilon)*np.random.randn( len(w) )
      delta_w = 0.5*adam_epsilon*g_mom - np.sqrt(adam_epsilon)*np.random.randn( len(w) )
      #w = w - 0.5*adam_epsilon*g_mom - np.sqrt(adam_epsilon)*np.random.randn( len(w) )
    
    proposed_w = w - delta_w
    
    proposed_loglik = problem.train_cost( proposed_w, seed=seed )
    if np.mod(t+1,verbose_rate)==0:
      print "proposed = %0.1f  current = %0.1f  dif = %0.1f"%(proposed_loglik, current_loglik,proposed_loglik - current_loglik)
    if np.log(np.random.rand()) < proposed_loglik - current_loglik:
      current_loglik = proposed_loglik
      w = proposed_w
    
    w = problem.fix_w( w )
    
    #print "after ",w
    alpha *= gamma_alpha
    alpha = max(alpha, alpha0/10)
    c     *= gamma_c
    
    problem.model.current.response_groups[0].epsilon *= gamma_eps
    
    train_error = problem.train_error( w )
    test_error = problem.test_error( w )
    errors.append( [train_error,test_error])
    if np.mod(t+1,verbose_rate)==0:
      print "%4d train %0.4f test %0.4f  alpha %g  "%(t+1, train_error, test_error,alpha), problem.model.current.theta

  return w, np.array(errors), ( np.array(others), np.array(param_noise), np.array(injected_noise) )

  
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
    
    g_hat, h_hat, ids = spsa_gradient_with_hessian( problem.train_cost, w, c, c_tilde, N, batchsize, q=q, ids=None, batchreplace=batchreplace, g_var=g_var )
    
    g_hat += problem.grad_prior( w )
    
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
    
    g_hat += problem.grad_prior( w )
    
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
