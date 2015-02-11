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

def Hamiltonian_dynamics_with_noised_gradient(U, grad_U, epsilon, L, current_q, p = None, noise_in_u=0):
  # %% Hamiltonian dynamics with noised gradient
  # x = xstart;
  # p = pstart;
  q = current_q
  if p is None:
    p = np.random.randn()
  nstep=L[0]
  niter=L[1]
  
  P = [p]
  Q = [q]
  
  # xs = zeros( nstep, 1 );
  # ys = zeros( nstep, 1 );
  # % do leapfrog
  # for i = 1 : nstep
  #     for j = 1: niter
  print "p half-step"
  
  #p = p - epsilon*( grad_U( q ) + noise_in_u*np.random.randn() ) / 2.0
  for i in range(nstep):
    p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )/2.0
    for j in range(niter):
      print "q full-step"
      q = q + epsilon*p
      
      
      if j+1 < L:
        print "p full-step"
        #P.append( p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )/2.0)
        p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )
    
    p = p - epsilon*( grad_U( q ) + noise_in_u*np.random.randn() ) / 2.0   
    P.append(p) 
    Q.append(q)   
      
  return q, p,np.array(Q), np.array(P)
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
  nstep=L[0]
  niter=L[1]
  
  P = [p]
  Q = [q]
  
  # xs = zeros( nstep, 1 );
  # ys = zeros( nstep, 1 );
  # % do leapfrog
  # for i = 1 : nstep
  #     for j = 1: niter
  for i in range(nstep):
    for j in range(niter):
      p = p - epsilon*( grad_U( q ) + noise_in_u*np.random.randn() ) / 2.0
      
      #print "p after full-step", p
      q = q + epsilon*p
      
      #print "q full-step",q
      p = p - epsilon*( grad_U( q ) + noise_in_u*np.random.randn() ) / 2.0
    print j
    Q.append(q)  
    P.append(p)
    
    
    
  return q, p,np.array(Q), np.array(P)
  # end
  # plot( xs, ys, 'rv', 'MarkerSize', 3 );
  # hold on;


def Euler_step( U, grad_U, epsilon, L, current_q, p = None, noise_in_u=0 ):
  q = current_q
  if p is None:
    p = np.random.randn()
  # else:
  #   p = 1
  nstep=L[0]
  niter=L[1]
  current_p = p
  
  P = [p]
  Q = [q]
  
  old_p = p
  old_q = q
  # half step for p
  #p = p - epsilon*grad_U(q)/2.0
  
  for i in range(nstep):
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
    return q, p,np.array(Q), np.array(P)
  else:
    return current_q, current_p,np.array(Q), np.array(P)

def mod_Euler_step( U, grad_U, epsilon, L, current_q, p = None, noise_in_u=0 ):
  q = current_q
  if p is None:
    p = np.random.randn()
  # else:
  #   p = 1 
    
  current_p = p
  nstep=L[0]
  niter=L[1]
  
  P = [p]
  Q = [q]
  
  # half step for p
  #p = p - epsilon*grad_U(q)/2.0
  
  for i in range(nstep):
    # full step for position
    q = q + epsilon*p
    
    p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )
    
    
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

def leapfrog_HMC_step_bad( U, grad_U, epsilon, L, current_q, p = None, noise_in_u=0 ):
  q = current_q
  if p is None:
    p = np.random.randn()
  # else:
  #   p = 1
  nstep=L[0]
  niter=L[1]
    
  current_p = p
  
  P = [p]
  Q = [q]
  
  
  
  for i in range(nstep):
    p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )/2.0
    for j in range(niter):
    
      # full step for position
      q = q + epsilon*p
      #Q.append(q)
    
      p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )/2.0
      
      # full step momentum except for last step
      #if i+1  < L:
      #  # collect p at full steps, not half steps
      #  #P.append(p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )/2.0)
    Q.append(q)
    
    P.append(p)
    
  # half step for momentum    
  #p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )/2.0
    
  
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
        
def leapfrog_HMC_step_good( U, grad_U, epsilon, L, current_q, p = None, noise_in_u=0 ):
  q = current_q
  if p is None:
    p = np.random.randn()
  # else:
  #   p = 1
  nstep=L[0]
  niter=L[1]
    
  current_p = p
  
  P = [p]
  Q = [q]
  
  for i in range(nstep):
    p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )/2.0
    for j in range(niter):
    
      # full step for position
      q = q + epsilon*p
      Q.append(q)
    
      # full step momentum except for last step
      if j+1  < L:
        # collect p at full steps, not half steps
        P.append(p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )/2.0)
        p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )
    
    # half step for momentum  
    #P.append(epsilon*( grad_U(q)+noise_in_u*np.random.randn() )/2.0)  
    p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )/2.0
    
    #Q.append(q)
  
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

def mod_Euler_friction_SGHMC_step( U, grad_U, epsilon, L, current_q, p = None, noise_in_u=0 ):
  q = current_q
  if p is None:
    p = np.random.randn()
   
  nstep=L[0]
  niter=L[1]
    
  current_p = p
  
  P = [p]
  Q = [q]
  
  for i in range(nstep):
    # full step for position
    q = q + epsilon*p
    
    p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn()+ 0.5*epsilon*noise_in_u**2 )
    
    Q.append(q)
    P.append(p)
  
  # not needed, but negate momentum to preserve reversibility
  p = -p
  
  current_U = U( current_q ); current_K = K( current_p );proposed_U = U( q ) ;proposed_K = K( p )
  
  if np.random.rand() < np.exp( current_U - proposed_U + current_K - proposed_K ):
    return q, p,np.array(Q), np.array(P)
  else:
    return current_q, current_p,np.array(Q), np.array(P)  

def leapfrog_friction_SGHMC_step( U, grad_U, epsilon, L, current_q, p = None, noise_in_u=0 ):
  
  known_friction = 0.5 * epsilon*noise_in_u**2 
  
  q = current_q
  if p is None:
    p = np.random.randn()
    
  current_p = p
  
  nstep=L[0]
  niter=L[1]
  
  P = [p]
  Q = [q]
  
  # half step for p
  for i in range(nstep):
    p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )/2.0
    for j in range(niter):
    
      # full step for position
      q = q + epsilon*p
      
    
      # full step momentum except for last step
      if j+1  < L:
        # collect p at full steps, not half steps
        #P.append(p - epsilon*( grad_U(q)+noise_in_u*np.random.randn()+ known_friction ))
        p = p - epsilon*( grad_U(q) + noise_in_u*np.random.randn() )
    
    # half step for momentum    
    p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() + known_friction )/2.0
    Q.append(q)
    P.append(p)
  
  # not needed, but negate momentum to preserve reversibility
  p = -p
  
  current_U = U( current_q ); current_K = K( current_p );proposed_U = U( q ) ;proposed_K = K( p )
  
  if np.random.rand() < np.exp( current_U - proposed_U + current_K - proposed_K ):
    return q, p,np.array(Q), np.array(P)
  else:
    return current_q, current_p,np.array(Q), np.array(P)
        
def naive_SGHMC_step( U, grad_U, epsilon, L, current_q, p = None, noise_in_u=0 ):
  q = current_q
  if p is None:
    p = np.random.randn()
  # else:
  #   p = 1
    
  current_p = p
  
  nstep=L[0]
  niter=L[1]
  
  P = [p]
  Q = [q]
  
  # half step for p
  p = p - epsilon*( grad_U(q)+noise_in_u*np.random.randn() )/2.0
  
  for i in range(nstep):
    
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
  p = -p
  
  current_U = U( current_q ); current_K = K( current_p );proposed_U = U( q ) ;proposed_K = K( p )
  
  if np.random.rand() < np.exp( current_U - proposed_U + current_K - proposed_K ):
    return q, p,np.array(Q), np.array(P)
  else:
    return current_q, current_p,np.array(Q), np.array(P)
    

    
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
  
  PP = []
  QQ = []
  for t in range(T):
    
    q,p,Qs,Ps= method( U, grad_U, epsilon, L, current_q, p=current_p, noise_in_u=noise_in_u )
    
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
  L = [300,50] # leapfrog steps
  q = 0  # initial position
  p = 1 #None #1  # set to None for actual sampling
  T = 1  # nbr samples
  seed = None #10
  noise_in_u = 0.5
  # r = 1
  # a = np.pi/2.0
  
  axis_to_use = [-8,8,-8,8]
  #axis_to_use = [-2.2,2.2,-2.2,2.2]
  # true trajectory
  t=np.linspace(0,2*np.pi,100)
  
  epsilon1 = 0.1
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
  
  Q,P,QQ,PP = HMC( T, p, q, epsilon1, L, U, grad_U, method = Euler_step, noise_in_u = noise_in_u, seed = seed )
  pp.subplot(2,4,1)
  pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  pp.plot( QQ.T, PP.T, 'o-')
  #pp.plot( Q, P, 'ko', ms=15)
  pp.axis(axis_to_use)
  pp.xlabel( 'position (q)')
  pp.ylabel( 'momentum (p)')
  pp.title('Eulers Method, eps %0.1f'%(epsilon1))
  
  Q,P,QQ,PP = HMC( T, p, q, epsilon1, L, U, grad_U, method = mod_Euler_step, noise_in_u = noise_in_u, seed = seed )
  pp.subplot(2,4,2)
  pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  pp.plot( QQ.T, PP.T, 'o-')
  #pp.plot( Q, P, 'ko', ms=15)
  pp.axis(axis_to_use)
  pp.xlabel( 'position (q)')
  pp.ylabel( 'momentum (p)')
  pp.title('Mod Eulers HMC, eps %0.1f'%(epsilon1))

  # eps = 0.3, L=20 leapfrog HMC
  Q,P,QQ,PP = HMC( T, p, q, epsilon1, L, U, grad_U, method = leapfrog_HMC_step_good, noise_in_u = noise_in_u, seed = seed )
  
  pp.subplot(2,4,3)
  pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  pp.plot( QQ.T, PP.T, 'o-')
  #pp.plot( Q, P, 'ko', ms=15)
  pp.axis(axis_to_use)
  pp.xlabel( 'position (q)')
  pp.ylabel( 'momentum (p)')
  pp.title('Leapfrog HMC -- bad, eps %0.1f'%(epsilon1))
  
  Q,P,QQ,PP = HMC( T, p, q, epsilon1, L, U, grad_U, method = naive_SGHMC_step, noise_in_u = noise_in_u, seed = seed )
  pp.subplot(2,4,5)
  pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  pp.plot( QQ.T, PP.T, 'o-')
  #pp.plot( Q, P, 'ko', ms=15)
  pp.axis(axis_to_use)
  pp.xlabel( 'position (q)')
  pp.ylabel( 'momentum (p)')
  pp.title('naive SGHMC, eps %0.1f'%(epsilon1))
    
  Q,P,QQ,PP = HMC( T, p, q, epsilon1, L, U, grad_U, method = mod_Euler_friction_SGHMC_step, noise_in_u = noise_in_u, seed = seed )
  pp.subplot(2,4,6)
  pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  pp.plot( QQ.T, PP.T, 'o-')
  #pp.plot( Q, P, 'ko', ms=15)
  pp.axis(axis_to_use)
  pp.xlabel( 'position (q)')
  pp.ylabel( 'momentum (p)')
  pp.title('mod Euler HMC w friction, eps %0.1f'%(epsilon1))
  
  Q,P,QQ,PP = HMC( T, p, q, epsilon1, L, U, grad_U, method = leapfrog_friction_SGHMC_step, noise_in_u = noise_in_u, seed = seed )
  pp.subplot(2,4,7)
  pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  pp.plot( QQ.T, PP.T, 'o')
  #pp.plot( Q, P, 'ko', ms=15)
  pp.axis(axis_to_use)
  pp.xlabel( 'position (q)')
  pp.ylabel( 'momentum (p)')
  pp.title('leapfrog SGHMC w friction, eps %0.1f'%(epsilon1))
  
  Q,P,QQ,PP = HMC( T, p, q, epsilon1, L, U, grad_U, method = Hamiltonian_dynamics_with_noised_gradient, noise_in_u = noise_in_u, seed = seed )
  pp.subplot(2,4,4)
  pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  pp.plot( QQ.T, PP.T, 'o')
  #pp.plot( Q, P, 'ko', ms=15)
  pp.axis(axis_to_use)
  pp.xlabel( 'position (q)')
  pp.ylabel( 'momentum (p)')
  pp.title('Figure 1 SGHMC, eps %0.1f'%(epsilon1))
  
  Q,P,QQ,PP = HMC( T, p, q, epsilon1, L, U, grad_U, method = Hamiltonian_dynamics_with_noised_gradient_bad, noise_in_u = noise_in_u, seed = seed )
  
  pp.subplot(2,4,8)
  pp.plot( pp.sin(t),pp.cos(t), 'k-',lw=8,alpha=0.25)
  pp.plot( QQ.T, PP.T, 'o')
  #pp.plot( Q, P, 'ko', ms=15)
  pp.axis(axis_to_use)
  pp.xlabel( 'position (q)')
  pp.ylabel( 'momentum (p)')
  pp.title('Figure 1 SGHMC, eps %0.1f'%(epsilon1))
  
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
  