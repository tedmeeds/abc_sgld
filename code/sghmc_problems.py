import numpy as np
import pylab as pp
from scipy import stats
import pdb

def logsumexp(x,dim=0):
    """Compute log(sum(exp(x))) in numerically stable way."""
    #xmax = x.max()
    #return xmax + log(exp(x-xmax).sum())
    if dim==0:
        xmax = x.max(0)
        return xmax + np.log(np.exp(x-xmax).sum(0))
    elif dim==1:
        xmax = x.max(1)
        return xmax + np.log(np.exp(x-xmax[:,np.newaxis]).sum(1))
    else: 
        raise 'dim ' + str(dim) + 'not supported'
        
def generate_sgld_fig1_problem( theta1 = 0, theta2 = 1, var1 = 10, var2 = 1, varx = 2, N = 100, batchsize = 1, seed = 1 ):
  
  std1 = np.sqrt(var1)
  std2 = np.sqrt(var2)
  stdx = np.sqrt(varx)
  
  # 0/1 assignments to first or second gaussian
  Z = (np.random.randn( N ) > 0).astype(int)
  
  # use Z as selectors
  X = Z*( theta1 + stdx*np.random.randn(N) ) + (1-Z)*(theta1+theta2 + stdx*np.random.randn(N))
  
  #prior = stats.mvn( np.zeros(2), )
  def U(q):
    # initialize model at q's locations
    p1 = stats.norm( q[0], stdx )
    p2 = stats.norm( q[0]+ q[1] , stdx )
    
    
    
    # full dataset:
    if True:
      L  = np.zeros( (N,2) )
      L[:,0] = np.log(0.5) + p1.logpdf( X )
      L[:,1] = np.log(0.5) + p2.logpdf( X )
    
      prior1 = stats.norm( 0, std1 )
      prior2 = stats.norm( 0, std2 )
      return -(np.sum( logsumexp( L, 1 ) )  + prior1.logpdf( q[0] ) + prior2.logpdf( q[1] ))
    else:
      I = np.random.permutation(N)[:batchsize]
      data = X[I]
      scale = float(N) / float(batchsize)
      L  = np.zeros( (batchsize,2) )
      L[:,0] = np.log(0.5) + p1.logpdf( data )
      L[:,1] = np.log(0.5) + p2.logpdf( data )
    
      prior1 = stats.norm( 0, std1 )
      prior2 = stats.norm( 0, std2 )
      
      #pdb.set_trace()
      return -scale*(np.sum( logsumexp( L, 1 ) ))  + prior1.logpdf( q[0] ) + prior2.logpdf( q[1] )
    
    
  def K(p):
    return 0.5*np.dot(p.T,p)
    
  def H(q,p):
    return U(q) + K(p)
    
  def dU(q):
    I = np.random.permutation(N)[:batchsize]
    data = X[I]
    scale = float(N) / float(batchsize)
    

    p1 = stats.norm( q[0], stdx ).pdf(data)
    p2 = stats.norm( q[0]+ q[1] , stdx ).pdf(data)
    prior1 = stats.norm( 0, std1 )
    prior2 = stats.norm( 0, std2 )
    
    prob_x = 0.5*p1 + 0.5*p2
    g1 = scale * np.sum( 0.5*( p1*(data-q[0]) + p2*(data-q[0]-q[1]) ) /(varx*prob_x) ) + prior1.pdf(q[0])*(-q[0]/var1)
    g2 = scale * np.sum( 0.5*( p2*(data-q[0]-q[1]) ) /(varx*prob_x) ) + prior2.pdf(q[1])*(-q[1]/var2)
    
    return  -np.array([g1,g2])
    
  def dK(p):
    return p
    
  def true_posterior(theta):
    return np.exp( -U(theta) )/5.365160237834569
    
  def theta_range():
    return np.linspace(-2,2,Nx)
    
  z = {}
  z["U"] = U
  z["K"] = K
  z["H"] = H
  z["dU"] = dU
  z["dK"] = dK
  z["true_posterior"] = true_posterior
  z["theta_range"] = theta_range
  return z
  
# used by R. Neal
def generate_1d_problem( noise=0, Nx = 100 ):
  def U(q):
    return 0.5*q**2
    
  def K(p):
    return 0.5*p**2
    
  def H(q,p):
    return U(q) + K(p)
    
  def dU(q):
    return q + noise*np.random.randn()
    
  def dK(p):
    return p
    
  def true_posterior(theta):
    return pow( 2*np.pi, -0.5)*np.exp( -U(theta) )
    
  def theta_range():
    return np.linspace(-3,3,Nx)
    
  z = {}
  z["U"] = U
  z["K"] = K
  z["H"] = H
  z["dU"] = dU
  z["dK"] = dK
  z["true_posterior"] = true_posterior
  z["theta_range"] = theta_range
  return z
  

def generate_sghmc_fig1_problem( noise = 0, Nx = 100 ):
  def U(q):
    return -2*q**2 + q**4
    
  def K(p):
    return 0.5*p**2
    
  def H(q,p):
    return U(q) + K(p)
    
  def dU(q):
    return -4*q + 4*q**3 + noise*np.random.randn()
    
  def dK(p):
    return p
    
  def true_posterior(theta):
    return np.exp( -U(theta) )/5.365160237834569
    
  def theta_range():
    return np.linspace(-2,2,Nx)
    
  z = {}
  z["U"] = U
  z["K"] = K
  z["H"] = H
  z["dU"] = dU
  z["dK"] = dK
  z["true_posterior"] = true_posterior
  z["theta_range"] = theta_range
  return z