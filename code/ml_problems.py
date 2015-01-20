import numpy as np
import pylab as pp
import scipy as sp
from numpy import newaxis

def logsumexp(x,dim=0):
  """Compute log(sum(exp(x))) in numerically stable way."""
  #xmax = x.max()
  #return xmax + log(exp(x-xmax).sum())
  if dim==0:
    xmax = x.max(0)
    return xmax + np.log(np.exp(x-xmax).sum(0))
  elif dim==1:
    xmax = x.max(1)
    return xmax + np.log(np.exp(x-xmax[:,newaxis]).sum(1))
  else: 
    raise 'dim ' + str(dim) + 'not supported'
        
def onehot( t, K ):
  N=len(t)
  T = np.zeros( (N,K))
  
  for trow,tk in zip( T, t ):
    trow[tk] = 1.0
  return T
  
def softmax( A, return_log=False ):
  # input is the N*K activations
  
  log_probabilities = A - logsumexp(A,1).reshape( (len(A),1) )
  
  probabilities = np.exp( log_probabilities )
  
  if return_log:
    return probabilities, log_probabilities
  else:
    return probabilities
  
def classification_error( t, y ):
  N = len(t)
  I = pp.find( t != y )
  return float(len(I))/N

class MulticlassLogisticRegression(object):
  def __init__( self, T_mat, X_mat, T_test, X_test ):
    self.X = X_mat
    self.T = T_mat
    self.Xtest = X_test
    self.Ttest = T_test
    
    self.t_train = np.argmax(self.T,1)
    self.t_test  = np.argmax(self.Ttest,1)
    
    # assume N*K for T_mat
    self.N,self.K = T_mat.shape
    self.Ntest,self.K = self.Ttest.shape
    
    # assume X_mat N*D
    N,self.D = X_mat.shape
  
    assert N == self.N, "should be same shape"
    
  def train_cost( self, w, ids, l1=0,l2=0 ):
    W = w.reshape( (self.D,self.K))
    
    utility = self.loglikelihood( W, ids = ids )
    
    utility -= l1*np.abs(w).sum()
    utility -= l2*pow(w,2).sum()
    
    return utility 
    
  def class_error( self, w, X, t ):
    W = w.reshape( (self.D,self.K))
    Y, logY = softmax( np.dot( X, W ), return_log = True ) 
    y = np.argmax(Y,1)
    cl_error  = classification_error( t, y )
    return cl_error
    
  def train_error( self, w ):
    return self.class_error( w, self.X, self.t_train )
  
  def test_error( self, w ):
    return self.class_error( w, self.Xtest, self.t_test )
      
  def loglikelihood( self, W, X = None, ids = None, return_Y = False ):
    #if X is None:
    #  X = self.X
      
    if ids is None:
      X = self.X
      T = self.T
      N = self.N
    else:
      X = self.X[ids,:]
      T = self.T[ids,:]
      N = len(ids)  
      
    Y, logY = softmax( np.dot( X, W ), return_log = True )
    
    log_like_per_n = np.sum( T * logY, 1 )
    
    loglike_data = np.sum( log_like_per_n )/N
    
    loglike_data *= self.N
    
    if return_Y:
      return loglike_data, Y
    else:
      return loglike_data
    
  def gradient( self, W, Y = None ):
    new_ids = np.random.permutation( self.N )[:self.batchsize]
    #ids = self.ids.copy()
    #self.ids[:self.batchsize/4] = new_ids[:self.batchsize/4]
    self.ids = new_ids
    ids = self.ids
    #if Y is None:
    Y = softmax( np.dot( self.X[ids,:], W ) )
    
    
    DIF = self.T[ids,:]-Y
    
    G = np.zeros( (self.D,self.K) )
    
    for k in range(self.K):
      g_k = np.dot( DIF[:,k].T, self.X[ids,:] ).T
      G[:,k] = g_k
    return self.batchsize*G/self.N