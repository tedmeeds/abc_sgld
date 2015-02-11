import numpy as np
import pylab as pp
import pdb

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

def Hamiltonian_dynamics_with_noised_gradient(U, grad_U, epsilon, L, current_q, p = None, noise_in_u=0):
  # %% Hamiltonian dynamics with noised gradient
  # x = xstart;
  # p = pstart;
  q = current_q
  if p is None:
    p = np.random.randn()
  nstep=L
  niter=1
  
  P = [p]
  Q = [q]
  
  # xs = zeros( nstep, 1 );
  # ys = zeros( nstep, 1 );
  # % do leapfrog
  # for i = 1 : nstep
  #     for j = 1: niter
  print "p half-step"
  p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )/2.0
  #p = p - epsilon*( grad_U( q ) + noise_in_u*np.random.randn() ) / 2.0
  for i in range(L):
    #for j in range(nstep):
      print "q full-step"
      q = q + epsilon*p
      Q.append(q)
      
      if i+1 < L:
        print "p full-step"
        P.append( p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )/2.0)
        p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )
    #if i+1 < L:
    #  p = p - epsilon*( grad_U( q ) + noise_in_u*np.random.randn() ) / 2.0
  p = p - epsilon*( grad_U( q ) + noise_in_u*np.random.randn() ) / 2.0   
  P.append(p)    
      
  return q, p,None,np.array(Q), np.array(P),None
  # end
  # plot( xs, ys, 'rv', 'MarkerSize', 3 );
  # hold on;
  
def Hamiltonian_dynamics_with_noised_gradient_bad(U, grad_U, epsilon, L, current_q, p = None, noise_in_u=0):
  # %% Hamiltonian dynamics with noised gradient
  # x = xstart;
  # p = pstart;
  q = current_q
  if p is None:
    p = np.random.randn()
  nstep=L
  niter=50
  
  P = [p]
  Q = [q]
  
  # xs = zeros( nstep, 1 );
  # ys = zeros( nstep, 1 );
  # % do leapfrog
  # for i = 1 : nstep
  #     for j = 1: niter
  for i in range(L):
    for j in range(nstep):
  #         p = p - gradU( x ) * dt / 2;
      
      p = p - epsilon*( grad_U( q ) + noise_in_u*np.random.randn() ) / 2.0
      print "p after full-step", p
      #print "i=%d j=%d q = %0.1f  p = %0.3f "%(i,j,q,p)
  #         x = x + p./m * dt;
  
      
      q = q + epsilon*p
      Q.append(q)
      print "q full-step",q
      #print i,j,q,p
  #         p = p - gradU( x ) * dt / 2;
      #print "p half-step"
      p = p - epsilon*( grad_U( q ) + noise_in_u*np.random.randn() ) / 2.0
      P.append(p)
      #print i,j,q,p
  #     end
  #     xs(i) = x;
  #     ys(i) = p;
    
    
    
  return q, p,None,np.array(Q), np.array(P),None
  # end
  # plot( xs, ys, 'rv', 'MarkerSize', 3 );
  # hold on;


def Euler_step( U, grad_U, epsilon, L, current_q, p = None, noise_in_u=0 ):
  q = current_q
  if p is None:
    p = np.random.randn()
  # else:
  #   p = 1
    
  current_p = p
  
  P = [p]
  Q = [q]
  
  old_p = p
  old_q = q
  # half step for p
  #p = p - epsilon*grad_U(q)/2.0
  
  for i in range(L):
    # full step for position
    q = q + epsilon*old_p
    
    p = old_p - epsilon*( grad_U(old_q)+noise_in_u*np.random.randn() )
    
    
    old_p = p
    old_q = q
    
    Q.append(q)
    P.append(p)
  
  # not needed, but negate momentum to preserve reversibility
  p = -p
  
  current_U = U( current_q )
  current_K = K( current_p )
  proposed_U = U( q )
  proposed_K = K( p )
  
  if np.random.rand() < np.exp( current_U - proposed_U + current_K - proposed_K ):
    return q, p,None,np.array(Q), np.array(P),None
  else:
    return current_q, current_p,None,np.array(Q), np.array(P),None

def mod_Euler_step( U, grad_U, epsilon, L, current_q, p = None, noise_in_u=0 ):
  q = current_q
  if p is None:
    p = np.random.randn()
  # else:
  #   p = 1 
    
  current_p = p
  
  P = [p]
  Q = [q]
  
  # half step for p
  #p = p - epsilon*grad_U(q)/2.0
  
  for i in range(L):
    # full step for position
    q = q + epsilon*p
    
    p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )
    
    
    Q.append(q)
    P.append(p)
  
  # not needed, but negate momentum to preserve reversibility
  #p = -p
  
  current_U = U( current_q )
  current_K = K( current_p )
  proposed_U = U( q )
  proposed_K = K( p )
  
  if np.random.rand() < np.exp( current_U - proposed_U + current_K - proposed_K ):
    return q, p,None,np.array(Q), np.array(P),None
  else:
    return current_q, current_p,None,np.array(Q), np.array(P),None    
        
def leapfrog_HMC_step( U, grad_U, epsilon, L, current_q, p = None, noise_in_u=0 ):
  q = current_q
  if p is None:
    p = np.random.randn()
  # else:
  #   p = 1
    
  current_p = p
  
  P = [p]
  Q = [q]
  
  # half step for p
  p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )/2.0
  
  for i in range(L):
    
    # full step for position
    q = q + epsilon*p
    Q.append(q)
    
    # full step momentum except for last step
    if i+1  < L:
      # collect p at full steps, not half steps
      P.append(p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )/2.0)
      p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )
    
  # half step for momentum    
  p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )/2.0
  P.append(p)
  
  # not needed, but negate momentum to preserve reversibility
  #p = -p
  
  current_U = U( current_q )
  current_K = K( current_p )
  proposed_U = U( q )
  proposed_K = K( p )
  
  if np.random.rand() < np.exp( current_U - proposed_U + current_K - proposed_K ):
    return q, p,None,np.array(Q), np.array(P),None
  else:
    return current_q, current_p,None,np.array(Q), np.array(P),None

def mod_Euler_friction_SGHMC_step( U, grad_U, epsilon, L, current_q, p = None, noise_in_u=0 ):
  #B = 0.5*epsilon*noise_in_u**2 
  #p = p - gradU( x ) * dt  - p * C * dt  + randn(1)*D;
  #        x = x + p./m * dt;
  dt = epsilon
  sigma = noise_in_u
          
  #C=1
  #Bhat =  0.5 * sigma**2 * dt;
  #D = 0*np.sqrt( 2 * (C-Bhat) * dt );
  
  V = sigma**2
  
  
  B = 0.5*V*epsilon
  C = B
  D = 0*np.sqrt( 2*(C-B)*epsilon )
  
  #pdb.set_trace()
  q = current_q
  if p is None:
    p = np.random.randn()
    
  def noisy_grad( q ):
    return grad_U( q )+ noise_in_u*np.random.randn()
    
  current_p = p
  
  P = [p]
  Q = [q]
  print p,q
  # half step for p
  p = p - epsilon*noisy_grad(q)/2.0 - p*C*epsilon/2.0 - D*np.random.randn()/2.0
  
  for i in range(L):
    
    # full step for position
    q = q + epsilon*p
    Q.append(q)
    
    # full step momentum except for last step
    if i+1  < L:
      # collect p at full steps, not half steps
      P.append(p - epsilon*noisy_grad(q)/2.0- p*C*epsilon/2.0 - D*np.random.randn()/2.0)
      p = p - epsilon*noisy_grad(q)- p*C*epsilon - D*np.random.randn()
    
  # half step for momentum    
  p = p - epsilon*noisy_grad(q)/2.0- p*C*epsilon/2.0 - D*np.random.randn()/2.0
  P.append(p)    
  
  # not needed, but negate momentum to preserve reversibility
  #p = -p
  
  current_U = U( current_q ); current_K = K( current_p );proposed_U = U( q ) ;proposed_K = K( p )
  
  if np.random.rand() < np.exp( current_U - proposed_U + current_K - proposed_K ):
    return q, p,None,np.array(Q), np.array(P),None
  else:
    return current_q, current_p,None,np.array(Q), np.array(P),None
    

def leapfrog_friction_SGHMC_step( U, grad_U, epsilon, L, current_q, p = None, noise_in_u=0 ):
  
  B = 0.5 *epsilon*noise_in_u**2 
  
  q = current_q
  if p is None:
    p = np.random.randn()
    
  current_p = p
  
  P = [p]
  Q = [q]
  
  # half step for p
  p = p - epsilon*( grad_U(q)+np.sqrt(2.0*B)*np.random.randn() + B*p)/2.0
  
  for i in range(L):
    
    # full step for position
    q = q + epsilon*p
    Q.append(q)
    
    # full step momentum except for last step
    if i+1  < L:
      # collect p at full steps, not half steps
      P.append(p - epsilon*( grad_U(q)+np.sqrt(2.0*B)*np.random.randn() + B*p))
      p = p - epsilon*( grad_U(q)+np.sqrt(2.0*B)*np.random.randn() + B*p)
    
  # half step for momentum    
  p = p - epsilon*( grad_U(q)+np.sqrt(2.0*B)*np.random.randn() + B*p)/2.0
  P.append(p)
  
  # not needed, but negate momentum to preserve reversibility
  #p = -p
  
  current_U = U( current_q ); current_K = K( current_p );proposed_U = U( q ) ;proposed_K = K( p )
  
  if np.random.rand() < np.exp( current_U - proposed_U + current_K - proposed_K ):
    return q, p,None,np.array(Q), np.array(P),None
  else:
    return current_q, current_p,None,np.array(Q), np.array(P),None

def SGNHT_step( U, grad_U, epsilon, L, current_q, p = None, noise_in_u=0 ):
  
  # from SGHMC paper
  # %% Second order Langevin dynamics with noised gradient
  # x = xstart;
  # p = pstart;
  # xs = zeros( nstep, 1 );
  # ys = zeros( nstep, 1 );
  # Bhat =  0.5 * sigma^2 * dt;
  # D = sqrt( 2 * (C-Bhat) * dt );
  #
  # % do leapfrog
  # for i = 1 : nstep
  #     for j = 1: niter
  #         p = p - gradU( x ) * dt  - p * C * dt  + randn(1)*D;
  #         x = x + p./m * dt;
  #     end
  #     xs(i) = x;
  #     ys(i) = p;
  # end
  # plot( xs, ys, 'gs', 'MarkerSize', 3 );
  # hold on;
  
  C=3
  h = epsilon
  Bhat = 0.5*epsilon*noise_in_u**2 
  D = np.sqrt(2.0*epsilon*(C-Bhat))
  #A = B #/epsilon
  eta = C
  
  q = current_q
  if p is None:
    p = np.random.randn()
    
  
  len_p = 1.0
  current_p = p
  
  P = [p]
  Q = [q]
  E = [eta]
  
  for i in range(L):
    p = p - h*eta*p - h*(grad_U(q)+noise_in_u*np.random.randn()) + D*np.random.randn()
    
    # full step for position
    q = q + h*p
    Q.append(q)
    
    eta = eta + (  p*p - 1.0 )*h
    E.append(eta)
    
    
    P.append(p)
    
    
    
    
    
  # not needed, but negate momentum to preserve reversibility
  #p = -p
  
  current_U = U( current_q ); current_K = K( current_p );proposed_U = U( q ) ;proposed_K = K( p )
  
  return q,p,eta,np.array(Q), np.array(P),np.array(E)
  
  if np.random.rand() < np.exp( current_U - proposed_U + current_K - proposed_K ):
    return q, p, np.array(Q), np.array(P)
  else:
    return current_
            
def naive_SGHMC_step( U, grad_U, epsilon, L, current_q, p = None, noise_in_u=0 ):
  q = current_q
  if p is None:
    p = np.random.randn()
  # else:
  #   p = 1
    
  current_p = p
  
  P = [p]
  Q = [q]
  
  # half step for p
  p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )/2.0
  
  for i in range(L):
    
    # full step for position
    q = q + epsilon*p
    Q.append(q)
    
    # full step momentum except for last step
    #if i+1  < L:
    # collect p at full steps, not half steps
    P.append(p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )/2.0)
    p = p - epsilon*( grad_U(q) + noise_in_u*np.random.randn() )
    
  # half step for momentum    
  p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )/2.0
  #P.append(p)
  
  # not needed, but negate momentum to preserve reversibility
  #p = -p
  
  current_U = U( current_q ); current_K = K( current_p );proposed_U = U( q ) ;proposed_K = K( p )
  
  if np.random.rand() < np.exp( current_U - proposed_U + current_K - proposed_K ):
    return q, p,None,np.array(Q), np.array(P),None
  else:
    return current_q, current_p,None,np.array(Q), np.array(P),None
    

    
def test_HMC_step( U, grad_U, epsilon, L, current_q, p = None, noise_in_u=0 ):
  mom = 0.9
  
  q = current_q
  if p is None:
    p = np.random.randn()
  else:
    p = 1 
    
  current_p = p
  
  P = [p]
  Q = [q]
  
  # half step for p
  g_inst = ( grad_U(q)+noise_in_u*np.random.randn() )
  gm = g_inst
  vm = g_inst**2
  p = p - epsilon*gm/2.0
  
  for i in range(L):
    
    # full step for position
    q = q + epsilon*p + epsilon*vm*np.random.randn()
    Q.append(q)
    
    # full step momentum except for last step
    if i+1  < L:
      # collect p at full steps, not half steps
      g_inst = grad_U(q)+noise_in_u*np.random.randn()
      gm = mom*gm + (1-mom)*g_inst
      vm = mom*vm + (1-mom)*g_inst**2
      P.append(p - epsilon*gm/2.0)
      p = p - epsilon*gm 
    
  # half step for momentum    
  g_inst = grad_U(q)+noise_in_u*np.random.randn()
  gm = mom*gm + (1-mom)*g_inst
  vm = mom*vm + (1-mom)*g_inst**2
  p = p - epsilon*gm/2.0
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
    
    
def HMC( T, current_p, current_q, epsilon, L, U, grad_U, method, noise_in_u = 0, seed = None ):
  if seed is not None:
    np.random.seed(seed)
    
  print "method = ", method
  print "injected noise = ", noise_in_u
  print "current_p = ", current_p
  print "current_q = ", current_q
  P = [current_p]
  Q = [current_q]
  E = [noise_in_u]
  PP = []
  QQ = []
  EE = []
  for t in range(T):
    
    q,p,eta,Qs,Ps,Etas= method( U, grad_U, epsilon, L, current_q, p=current_p, noise_in_u=noise_in_u )
    
    PP.append(Ps)
    QQ.append(Qs)
    EE.append(Etas)
    P.append( p )
    Q.append( q )
    E.append( eta )
    current_q = q
    
  return np.array(Q), np.array(P), np.array(E), np.array(QQ), np.array(PP), np.array(EE)
      
if __name__ == "__main__":
  print "================================================"
  print "   Reproduces plots from Radford Neal's         "
  print "   HMC chapter in MCMC Handbook (Figure 1)      "
  print "================================================"
  pp.close('all')
  L = 150 # leapfrog steps
  q = 1  # initial position
  p = 0  # set to None for actual sampling
  T = 1  # nbr samples
  seed = 2
  noise_in_u = 0.5
  # r = 1
  # a = np.pi/2.0
  
  axis_to_use = [-8,8,-8,8]
  axis_to_use = [-2.2,2.2,-2.2,2.2]
  # true trajectory
  t=np.linspace(0,2*np.pi,100)
  
  epsilon1 = 0.3
  #epsilon2 = 1.2
  pp.figure(1,figsize=(16,10))
  pp.clf()
  
  # -----------------------------

  
  # # eps = 1.2, L=20 leapfrog HMC
  # Q,P,QQ,PP = HMC( T, p, q, epsilon2, L, U, grad_U, noise_in_u = noise_in_u, seed = seed )
  #
  # pp.subplot(2,4,4)
  # pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  # pp.plot( QQ.T, PP.T, 'o-')
  # #pp.plot( Q, P, 'ko', ms=15)
  # pp.axis(axis_to_use)
  # pp.xlabel( 'position (q)')
  # pp.ylabel( 'momentum (p)')
  # pp.title('Leapfrog Method, stepsize %0.1f'%(epsilon2))
  
  Q,P,E,QQ,PP,EE = HMC( T, p, q, epsilon1, L, U, grad_U, method = Euler_step, noise_in_u = noise_in_u, seed = seed )
  pp.subplot(2,4,1)
  pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  pp.plot( QQ.T, PP.T, 'o-')
  pp.plot( Q, P, 'ko', ms=15)
  pp.axis(axis_to_use)
  pp.xlabel( 'position (q)')
  pp.ylabel( 'momentum (p)')
  pp.title('Eulers Method, eps %0.1f'%(epsilon1))
  
  Q,P,E,QQ,PP,EE = HMC( T, p, q, epsilon1, L, U, grad_U, method = mod_Euler_step, noise_in_u = noise_in_u, seed = seed )
  pp.subplot(2,4,2)
  pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  pp.plot( QQ.T, PP.T, 'o-')
  pp.plot( Q, P, 'ko', ms=15)
  pp.axis(axis_to_use)
  pp.xlabel( 'position (q)')
  pp.ylabel( 'momentum (p)')
  pp.title('Mod Eulers HMC, eps %0.1f'%(epsilon1))

  # eps = 0.3, L=20 leapfrog HMC
  Q,P,E,QQ,PP,EE = HMC( T, p, q, epsilon1, L, U, grad_U, method = leapfrog_HMC_step, noise_in_u = noise_in_u, seed = seed )
  
  pp.subplot(2,4,3)
  pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  pp.plot( QQ.T, PP.T, 'o-')
  pp.plot( Q, P, 'ko', ms=15)
  pp.axis(axis_to_use)
  pp.xlabel( 'position (q)')
  pp.ylabel( 'momentum (p)')
  pp.title('Leapfrog HMC, eps %0.1f'%(epsilon1))
  
  Q,P,E,QQ,PP,EE = HMC( T, p, q, epsilon1, L, U, grad_U, method = naive_SGHMC_step, noise_in_u = noise_in_u, seed = seed )
  pp.subplot(2,4,5)
  pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  pp.plot( QQ.T, PP.T, 'o-')
  pp.plot( Q, P, 'ko', ms=15)
  pp.axis(axis_to_use)
  pp.xlabel( 'position (q)')
  pp.ylabel( 'momentum (p)')
  pp.title('naive SGHMC, eps %0.1f'%(epsilon1))
    
  Q,P,E,QQ,PP,EE = HMC( T, p, q, epsilon1, L, U, grad_U, method = mod_Euler_friction_SGHMC_step, noise_in_u = noise_in_u, seed = seed )
  pp.subplot(2,4,6)
  pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  pp.plot( QQ.T, PP.T, 'o-')
  pp.plot( Q, P, 'ko', ms=15)
  pp.axis(axis_to_use)
  pp.xlabel( 'position (q)')
  pp.ylabel( 'momentum (p)')
  pp.title('mod Euler HMC w friction, eps %0.1f'%(epsilon1))
  
  Q,P,E,QQ,PP,EE = HMC( T, p, q, epsilon1, L, U, grad_U, method = leapfrog_friction_SGHMC_step, noise_in_u = noise_in_u, seed = seed )
  pp.subplot(2,4,7)
  pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  pp.plot( QQ.T, PP.T, 'o-')
  pp.plot( Q, P, 'ko', ms=15)
  pp.axis(axis_to_use)
  pp.xlabel( 'position (q)')
  pp.ylabel( 'momentum (p)')
  pp.title('leapfrog SGHMC w friction, eps %0.1f'%(epsilon1))
  
  Q,P,E,QQ,PP,EE = HMC( T, p, q, epsilon1, L, U, grad_U, method = SGNHT_step, noise_in_u = noise_in_u, seed = seed )
  pp.subplot(2,4,4)
  pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  pp.plot( QQ.T, PP.T, 'o-')
  pp.plot( Q, P, 'ko', ms=15)
  pp.axis(axis_to_use)
  pp.xlabel( 'position (q)')
  pp.ylabel( 'momentum (p)')
  pp.title('SGHNT, eps %0.1f'%(epsilon1))
  
  pp.subplot(2,4,8)
  pp.plot( EE[0], 'o-')
  pp.title('SGHNT -- etas, eps %0.1f'%(epsilon1))
  
  # Q,P,QQ,PP = HMC( T, p, q, epsilon1, L, U, grad_U, method = Hamiltonian_dynamics_with_noised_gradient, noise_in_u = noise_in_u, seed = seed )
  # pp.subplot(2,4,4)
  # pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  # pp.plot( QQ.T, PP.T, 'o-')
  # #pp.plot( Q, P, 'ko', ms=15)
  # pp.axis(axis_to_use)
  # pp.xlabel( 'position (q)')
  # pp.ylabel( 'momentum (p)')
  # pp.title('Figure 1 SGHMC, eps %0.1f'%(epsilon1))
  
  # Q,P,QQ,PP = HMC( T, p, q, epsilon1, L, U, grad_U, method = Hamiltonian_dynamics_with_noised_gradient_bad, noise_in_u = noise_in_u, seed = seed )
  #
  # pp.subplot(2,4,8)
  # pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  # pp.plot( QQ.T, PP.T, 'o-')
  # #pp.plot( Q, P, 'ko', ms=15)
  # pp.axis(axis_to_use)
  # pp.xlabel( 'position (q)')
  # pp.ylabel( 'momentum (p)')
  # pp.title('Figure 1 SGHMC, eps %0.1f'%(epsilon1))
  
 #  pp.figure(2)
 #  Q,P,QQ,PP = HMC( T, p, q, epsilon1, L, U, grad_U, method = test_HMC_step, noise_in_u = noise_in_u )
 # # pp.subplot(2,2,2)
 #  pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
 #  pp.plot( QQ.T, PP.T, 'o-')
 #  #pp.plot( Q, P, 'ko', ms=15)
 #  pp.axis([-2.2,2.2,-2.2,2.2])
 #  pp.xlabel( 'position (q)')
 #  pp.ylabel( 'momentum (p)')
 #  pp.title('test_HMC_step, stepsize %0.1f'%(epsilon1))
 #
  pp.show()
  