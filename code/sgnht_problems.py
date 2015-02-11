import numpy as np
import pylab as pp
from scipy import stats
import pdb

class generate_1d_gaussian_var_unknown( object ):
  def __init__( self, N = 100, Ntilde = 10, mu_star = 0.0, gamma_star = 1.0,  Nx = 100  ):
  
    self.mu0 = 0
    self.N = N
    self.Ntilde = Ntilde
    self.mu_star = mu_star
    self.gamma_star = gamma_star
    self.varx = 1.0 / gamma_star
    self.stdx = np.sqrt(self.varx)
    self.X = self.mu_star + self.stdx*np.random.randn(self.N)
    self.muX = self.X.mean()
  
    self.gamma_a = 1.0
    self.gamma_b = 1.0
  
    self.gamma_prior = stats.gamma( self.gamma_a, 0.0, 1.0/self.gamma_b )
    self.mu_prior    = stats.norm( self.mu0, self.stdx )
  
    self.SS     = np.sum( (self.X - self.muX)**2 )
    self.muN    = self.muX*float(self.N)/float(self.N+1.0)
    self.kappaN = float(self.N+1.0)
    
    self.aN = self.gamma_a + float(self.N)/2.0
    self.bN = self.gamma_b + 0.5*self.SS + (float(self.N)/float(2*(1+self.N)))*( self.muX - self.mu0 )**2
    
    self.post_var = 1.0/self.N
    self.post_std = np.sqrt(self.post_var)
  
  
    self.SS0 = 2*self.gamma_b
    self.nu0 = 2*self.gamma_a
    self.nuN = self.nu0 + self.N
    
    self.mu_post    = stats.nct( self.nuN, self.muN, np.sqrt(((1.0/(1.0+self.N))*(1.0/self.nuN)*(self.SS0 + self.SS + self.kappaN*( self.muX - self.mu0 )**2)))  )
    
    self.gamma_post = stats.gamma( self.aN, 0.0, 1.0/self.bN )
    
    self.gamma_post_samples = self.gamma_post.rvs(100000)
    self.mu_post_samples    = self.muN + np.sqrt(1.0 / (self.kappaN*self.gamma_post_samples) )*np.random.randn(100000)
    self.Nx = Nx
    
    self.bins_mu = np.linspace( -0.6,1.0, Nx )
    self.bins_gamma = np.linspace( 0,2,Nx)
  #mu_post    = stats.norm( muN, 1.0/(kappaN*) )
  
  #true_post = stats.norm( post_mu, post_std )
    
  def U(self,q):
    try:
      varx = 1.0/q[1]
      mux  = q[0]
      return -stats.norm( mux, np.sqrt(varx) ).logpdf( X ).sum()
    except:
      
      u=[-stats.norm( qi[0], np.sqrt(1.0/qi[1]) ).logpdf( X ).sum() for qi in q]
      return np.array(u)
    
  def K(self,p):
    return 0.5*p*p
    
  def H(self,q,p):
    return U(q) + K(p)
    
  def dU(self,q, ids = None):
    mu    = q[0]
    gamma = q[1]
    
    if ids is None:
      ids = np.random.permutation(self.N)[:self.Ntilde]
      
    g=np.zeros(2)
    g[0] = (self.N+1)*mu*gamma - gamma*float(self.N)*self.X[ids].sum()/self.Ntilde
    g[1] = -(self.N+1)/(2*gamma) + 1 + 0.5*mu**2 + 0.5*float(self.N)*np.sum( (self.X[ids]-mu)**2 )/self.Ntilde
    
    # g[0] = (self.N+1)*mu*gamma - gamma*self.X.sum()
    # g[1] = -(self.N+1)/(2*gamma) + 1 + 0.5*mu**2 + 0.5*np.sum( (self.X-mu)**2 )
    
    return g
    
  def dK(self,p):
    return p
    
  def true_posterior(self,theta):
    return true_post.pdf(theta)
    
  def posterior(self):
    return true_post
    
  def posterior(self,theta):
    return true_post
    
  def theta_range(self):
    return np.linspace(-1.0,1.0,self.Nx)
  
  def bin_density( self, bins ):
    p = np.zeros(len(bins)-1)
    
    i = 0
    for l,r in zip( bins[:-1], bins[1:] ):
      p[i] = true_post.cdf(r)-true_post.cdf(l)
      i+=1
    return p
 
  def sample_density( self,samples, bins ):
    cnts, bins, patches = pp.hist( samples, bins )
    probs = cnts / float(len(samples))
    return probs
    
  def rmse( self, samples, bins, true_density ):
    probs = sample_density( samples, bins )
    
    error = np.sqrt( ( ( probs-true_density )**2 ).mean() )
    
    return error
    
  def autocorr( self, samples ):
    return


def generate_1d_gaussian_mu_unknown( N = 100, Ntilde = 10, mu_star = 0.0, varx = 1.0, Nx = 100  ):
  
  X = mu_star + np.sqrt(varx)*np.random.randn(N)
  post_mu  = X.mean()
  post_var = 1.0/N
  post_std = np.sqrt(post_var)
  
  true_post = stats.norm( post_mu, post_std )
    
  def U(q):
    try:
      return -stats.norm( q, np.sqrt(varx) ).logpdf( X ).sum()
    except:
      
      u=[-stats.norm( qi, np.sqrt(varx) ).logpdf( X ).sum() for qi in q]
      return np.array(u)
    
  def K(p):
    return 0.5*p*p
    
  def H(q,p):
    return U(q) + K(p)
    
  def dU(q, ids = None):
    if ids is None:
      ids = np.random.permutation(N)[:Ntilde]
      
    return N*q - float(N)*X[ids].mean()
    
  def dK(p):
    return p
    
  def true_posterior(theta):
    return true_post.pdf(theta)
    
  def posterior():
    return true_post
    
  def posterior(theta):
    return true_post
    
  def theta_range():
    return np.linspace(-1.0,1.0,Nx)
  
  def bin_density( bins ):
    p = np.zeros(len(bins)-1)
    
    i = 0
    for l,r in zip( bins[:-1], bins[1:] ):
      p[i] = true_post.cdf(r)-true_post.cdf(l)
      i+=1
    return p
 
  def sample_density( samples, bins ):
    cnts, bins, patches = pp.hist( samples, bins )
    probs = cnts / float(len(samples))
    return probs
    
  def rmse( samples, bins, true_density ):
    probs = sample_density( samples, bins )
    
    error = np.sqrt( ( ( probs-true_density )**2 ).mean() )
    
    return error
    
  def autocorr( samples ):
    return
      
  z = {}
  z["U"] = U
  z["K"] = K
  z["H"] = H
  z["dU"] = dU
  z["dK"] = dK
  z["true_posterior"] = true_posterior
  z["theta_range"] = np.linspace(-1.0,1.0,Nx)
  z["bins"]        = z["theta_range"]
  z["posterior"] = true_post
  z["sample_density"] = sample_density
  z["bin_density"] = bin_density( z["bins"])
  z["rmse"] = rmse
  z["autocorr"] = autocorr
  return z
  
def generate_double_well( h, B, Nx = 200  ):
  
  def true_grad_U(q):
    # note SGNHT says: grad_U(q)*h = grad_U(q)*h + N(0,2Bh)
    return (4*q**3 + 3*q**2 - 26*q - 1)/14.0
    
  def U(q):
    return (q+4)*(q+1)*(q-1)*(q-3)/14.0 + 0.5
    
  def K(p):
    return 0.5*p*p
    
  def H(q,p):
    return U(q) + K(p)
    
  def dU(q):
    return true_grad_U(q) + np.sqrt(2*B*h)*np.random.randn()/h
    
  def dK(p):
    return p
    
  def true_posterior(theta):
    return np.exp( -U(theta) )/5.365160237834569
    
  def theta_range():
    return np.linspace(-6,6,Nx)
    
  z = {}
  z["U"] = U
  z["K"] = K
  z["H"] = H
  z["dU"] = dU
  z["dK"] = dK
  z["true_posterior"] = true_posterior
  z["theta_range"] = theta_range
  return z