import numpy as np
import pylab as pp
import scipy as sp
from numpy import newaxis
from helpers import *

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

    return -utility

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

  def gradient( self, w, ids = None ):
    W = w.reshape( (self.D,self.K))

    if ids is None:
      new_ids = np.random.permutation( self.N )[:self.batchsize]
    else:
      new_ids = ids

    self.ids = new_ids
    ids = self.ids
    n = len(ids)
    Y = softmax( np.dot( self.X[ids,:], W ) )


    DIF = self.T[ids,:]-Y

    G = np.zeros( (self.D,self.K) )

    for k in range(self.K):
      g_k = np.dot( DIF[:,k].T, self.X[ids,:] ).T
      G[:,k] = g_k
    g = G.reshape( (self.D*self.K,))
    return self.N*g/n
