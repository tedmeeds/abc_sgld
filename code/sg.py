import numpy as np
import pylab as pp
import scipy as sp
import pdb

def euler_step( U, grad_U, K, epsilon, L, current_q, p, mh_correction, other_params = None ):
  q = current_q
  if p is None:
    p = np.random.randn()
    
  current_p = p
  
  P = [p]
  Q = [q]
  
  old_p = p
  old_q = q
  # half step for p
  #p = p - epsilon*grad_U(q)/2.0
  
  for i in range(L):
    p = old_p - epsilon*grad_U(q)
    
    # full step for position
    q = q + epsilon*old_p
    
    old_p = p
    #old_q = q
    
    Q.append(q)
    P.append(p)
  
  # not needed, but negate momentum to preserve reversibility
  p = -p
  
  current_U = U( current_q )
  current_K = K( current_p )
  proposed_U = U( q )
  proposed_K = K( p )
  
  if mh_correction:
    if np.random.rand() < np.exp( current_U - proposed_U + current_K - proposed_K ):
      return q, p,np.array(Q), np.array(P)
    else:
      return current_q, current_p, np.array(Q), np.array(P)
  else:
    return q, p, np.array(Q), np.array(P)
    
def mod_euler_step( U, grad_U, K, epsilon, L, current_q, p, mh_correction, other_params = None ):
  q = current_q
  if p is None:
    p = np.random.randn()
    
  current_p = p
  
  P = [p]
  Q = [q]
  
  # half step for p
  #p = p - epsilon*grad_U(q)/2.0
  
  for i in range(L):
    p = p - epsilon*grad_U(q)
    
    # full step for position
    q = q + epsilon*p
    
    Q.append(q)
    P.append(p)
  
  # not needed, but negate momentum to preserve reversibility
  p = -p
  
  current_U = U( current_q )
  current_K = K( current_p )
  proposed_U = U( q )
  proposed_K = K( p )
  
  if mh_correction:
    if np.random.rand() < np.exp( current_U - proposed_U + current_K - proposed_K ):
      return q, p,np.array(Q), np.array(P)
    else:
      return current_q, current_p, np.array(Q), np.array(P)
  else:
    return q, p, np.array(Q), np.array(P)

def langevin_step( U, grad_U, K, epsilon, L, current_q, p, mh_correction, other_params = None ):
  q = current_q
  if p is None:
    p = np.random.randn()
     
  current_p = p
  
  P = [p]
  Q = [q]
  
  # half step for p
  # p = p - epsilon*grad_U(q)/2.0
  # q = q + epsilon*p
  # sub in p update: q = q + epsilon*(p - epsilon*grad_U(q)/2.0)
  # expand: q = q + epsilon*p - epsilon*epsilon*grad_U(q)/2.0
  # sub in normal(0,1) for p
  #q = q + epsilon*np.random.randn() - epsilon*epsilon*grad_U(q)/2.0 
  
  # but... to use with mh correction, use p
  grad_U_at_q = grad_U(q)  # save grad in case of noise
  p = p - epsilon*grad_U_at_q/2.0
  q = q + epsilon*p
  
  # now propose new p
  p = p - epsilon*grad_U_at_q/2.0 - epsilon*grad_U(q)/2.0

  current_U = U( current_q )
  current_K = K( current_p )
  proposed_U = U( q )
  proposed_K = K( p )
  
  if mh_correction:
    if np.random.rand() < np.exp( current_U - proposed_U + current_K - proposed_K  ):
      return q, p,np.array(Q), np.array(P)
    else:
      return current_q, current_p, np.array(Q), np.array(P)
  else:
    return q, p, np.array(Q), np.array(P)

def sghmc_step( U, grad_U, K, epsilon, L, current_q, p, mh_correction, other_params = None ):
  
  C = other_params["C"]
  V = other_params["V"]
  
  B = 0.5*V*epsilon
  D = np.sqrt( 2*(C-B)*epsilon )
  
  q = current_q
  if p is None:
    if q.__class__ == np.array:
      p = np.random.randn( len(q) )
    else:
      p = np.random.randn()
      
  current_p = p
  
  P = [p]
  Q = [q]
  
  # half step for p
  p = p - epsilon*grad_U(q)/2.0 - p*C*epsilon/2.0 - D*np.random.randn()/2.0
  
  for i in range(L):
    
    # full step for position
    q = q + epsilon*p
    Q.append(q)
    
    # full step momentum except for last step
    if i+1  < L:
      # collect p at full steps, not half steps
      P.append(p - epsilon*grad_U(q)/2.0- p*C*epsilon/2.0 - D*np.random.randn()/2.0)
      p = p - epsilon*grad_U(q)- p*C*epsilon - D*np.random.randn()
    
  # half step for momentum    
  p = p - epsilon*grad_U(q)/2.0- p*C*epsilon/2.0 - D*np.random.randn()/2.0
  P.append(p)
  
  # not needed, but negate momentum to preserve reversibility
  p = -p
  
  current_U = U( current_q )
  current_K = K( current_p )
  proposed_U = U( q )
  proposed_K = K( p )
  
  if mh_correction:
    if np.random.rand() < np.exp( current_U - proposed_U + current_K - proposed_K ):
      return q, p,np.array(Q), np.array(P)
    else:
      return current_q, current_p, np.array(Q), np.array(P)
  else:
    return q, p, np.array(Q), np.array(P)


def sgnht_step( U, grad_U, K, epsilon, L, current_q, p, mh_correction, other_params = None ):
  
  C = other_params["C"]
  V = other_params["V"]
  
  #B = 0.5*V*epsilon
  #D = np.sqrt( 2*(C-B)*epsilon )
  
  A = V
  eta = V
  
  q = current_q
  if p is None:
    if q.__class__ == np.array:
      p = np.random.randn( len(q) )
    else:
      p = np.random.randn()
      
  current_p = p
  
  P = [p]
  Q = [q]
  
  # half step for p
  p = p - epsilon*grad_U(q)/2.0 - p*eta*epsilon/2.0 - np.sqrt(2*A)*np.random.randn()/2.0
  
  for i in range(L):
    
    # full step for position
    q = q + epsilon*p
    Q.append(q)
    
    eta = eta + (p*p-1.0)*epsilon
    # full step momentum except for last step
    if i+1  < L:
      # collect p at full steps, not half steps
      P.append(p - epsilon*grad_U(q)/2.0- p*eta*epsilon/2.0 - np.sqrt(2*A)*np.random.randn()/2.0)
      p = p - epsilon*grad_U(q)- p*eta*epsilon - np.sqrt(2*A)*np.random.randn()
      
    print p,q
    
  # half step for momentum    
  eta = eta + (p*p-1.0)*epsilon
  p = p - epsilon*grad_U(q)/2.0- p*eta*epsilon/2.0 - np.sqrt(2*A)*np.random.randn()/2.0
  P.append(p)
  
  # not needed, but negate momentum to preserve reversibility
  p = -p
  
  current_U = U( current_q )
  current_K = K( current_p )
  proposed_U = U( q )
  proposed_K = K( p )
  
  if mh_correction:
    if np.random.rand() < np.exp( current_U - proposed_U + current_K - proposed_K ):
      return q, p,np.array(Q), np.array(P)
    else:
      return current_q, current_p, np.array(Q), np.array(P)
  else:
    return q, p, np.array(Q), np.array(P)

        
def leapfrog_step( U, grad_U, K, epsilon, L, current_q, p, mh_correction, other_params=None ):
  q = current_q
  if p is None:
    if q.__class__ == np.array:
      p = np.random.randn( len(q) )
    else:
      p = np.random.randn()
      
  current_p = p
  
  P = [p]
  Q = [q]
  
  # half step for p
  p = p - epsilon*grad_U(q)/2.0
  
  for i in range(L):
    
    # full step for position
    q = q + epsilon*p
    Q.append(q)
    
    # full step momentum except for last step
    if i+1  < L:
      # collect p at full steps, not half steps
      P.append(p - epsilon*grad_U(q)/2.0)
      p = p - epsilon*grad_U(q)
    
  # half step for momentum    
  p = p - epsilon*grad_U(q)/2.0
  P.append(p)
  
  # not needed, but negate momentum to preserve reversibility
  p = -p
  
  current_U = U( current_q )
  current_K = K( current_p )
  proposed_U = U( q )
  proposed_K = K( p )
  
  if mh_correction:
    if np.random.rand() < np.exp( current_U - proposed_U + current_K - proposed_K ):
      return q, p,np.array(Q), np.array(P)
    else:
      return current_q, current_p, np.array(Q), np.array(P)
  else:
    return q, p, np.array(Q), np.array(P)


def HMC( T, current_q, epsilon, L, U, grad_U, K, step_method, mh_correction, other_params = None ):
  
  current_p = None
  
  P = [None]
  Q = [current_q]
  
  for t in range(T):
    q, p, Qtraj, Ptraj = step_method( U, grad_U, K, epsilon, L, current_q, current_p, mh_correction,  other_params=other_params )
    
    P.append( p )
    Q.append( q )
    
    current_q = q
    
  return np.array(Q), np.array(P)
  
def run_hmc_with_problem( T, problem, step_method, mh_correction, current_q, epsilon, L  ):
  return HMC( T, current_q, epsilon, L, problem["U"], problem["dU"], problem["K"], step_method, mh_correction, problem["other_params"] ) 
  