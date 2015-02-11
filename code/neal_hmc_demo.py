import numpy as np
import pylab as pp

def U(q):
  return 0.5*q*q
  
def K(p):
  return 0.5*p*p

def hamiltonian( q, p ):
  return U(q) + K(p)
  
def grad_U( q ):
  return q
  
def grad_K( p ):
  return p

def Euler_step( U, grad_U, epsilon, L, current_q, p = None ):
  q = current_q
  if p is None:
    p = np.random.randn()
  else:
    p = 1 
    
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
  
  if np.random.rand() < np.exp( current_U - proposed_U + current_K - proposed_K ):
    return q, p,np.array(Q), np.array(P)
  else:
    return current_q, current_p,np.array(Q), np.array(P)

def mod_Euler_step( U, grad_U, epsilon, L, current_q, p = None ):
  q = current_q
  if p is None:
    p = np.random.randn()
  else:
    p = 1 
    
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
  
  if np.random.rand() < np.exp( current_U - proposed_U + current_K - proposed_K ):
    return q, p,np.array(Q), np.array(P)
  else:
    return current_q, current_p,np.array(Q), np.array(P)    
    
def HMC_step( U, grad_U, epsilon, L, current_q, p = None ):
  q = current_q
  if p is None:
    p = np.random.randn()
  else:
    p = 1 
    
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
  
  if np.random.rand() < np.exp( current_U - proposed_U + current_K - proposed_K ):
    return q, p,np.array(Q), np.array(P)
  else:
    return current_q, current_p,np.array(Q), np.array(P)
    

def HMC( T, current_p, current_q, epsilon, L, U, grad_U, method = HMC_step ):
  
  P = [current_p]
  Q = [current_q]
  
  PP = []
  QQ = []
  for t in range(T):
    q,p,Qs,Ps= method( U, grad_U, epsilon, L, current_q, current_p )
    
    PP.append(Ps)
    QQ.append(Qs)
    P.append( p )
    Q.append( q )
    current_q = q
    
  return np.array(Q), np.array(P), np.array(QQ), np.array(PP)
      
if __name__ == "__main__":
  print "================================================"
  print "   Reproduces plots from Radford Neal's         "
  print "   HMC chapter in MCMC Handbook (Figure 1)      "
  print "================================================"
  pp.close('all')
  L = 20 # leapfrog steps
  q = 0  # initial position
  p = 1  # set to None for actual sampling
  T = 1  # nbr samples
  # r = 1
  # a = np.pi/2.0
  
  # true trajectory
  t=np.linspace(0,2*np.pi,100)
  
  epsilon1 = 0.3
  epsilon2 = 1.2
  
  # -----------------------------
  # eps = 0.3, L=20 leapfrog HMC
  Q,P,QQ,PP = HMC( T, p, q, epsilon1, L, U, grad_U )
  
  
  pp.figure(1,figsize=(12,10))
  pp.clf()
  
  pp.subplot(2,2,3)
  pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  pp.plot( QQ.T, PP.T, 'o-')
  #pp.plot( Q, P, 'ko', ms=15)
  pp.axis([-2,2,-2,2])
  pp.xlabel( 'position (q)')
  pp.ylabel( 'momentum (p)')
  pp.title('Leapfrog Method, stepsize 0.3')
  
  # eps = 1.2, L=20 leapfrog HMC
  Q,P,QQ,PP = HMC( T, p, q, epsilon2, L, U, grad_U )
  
  pp.subplot(2,2,4)
  pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  pp.plot( QQ.T, PP.T, 'o-')
  #pp.plot( Q, P, 'ko', ms=15)
  pp.axis([-2,2,-2,2])
  pp.xlabel( 'position (q)')
  pp.ylabel( 'momentum (p)')
  pp.title('Leapfrog Method, stepsize 1.2')
  
  Q,P,QQ,PP = HMC( T, p, q, epsilon1, L, U, grad_U, method = Euler_step )
  pp.subplot(2,2,1)
  pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  pp.plot( QQ.T, PP.T, 'o-')
  #pp.plot( Q, P, 'ko', ms=15)
  pp.axis([-2.2,2.2,-2.2,2.2])
  pp.xlabel( 'position (q)')
  pp.ylabel( 'momentum (p)')
  pp.title('Eulers Method, stepsize 0.3')
  
  Q,P,QQ,PP = HMC( T, p, q, epsilon1, L, U, grad_U, method = mod_Euler_step )
  pp.subplot(2,2,2)
  pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  pp.plot( QQ.T, PP.T, 'o-')
  #pp.plot( Q, P, 'ko', ms=15)
  pp.axis([-2.2,2.2,-2.2,2.2])
  pp.xlabel( 'position (q)')
  pp.ylabel( 'momentum (p)')
  pp.title('Modified Eulers Method, stepsize 0.3')
  pp.show()
  