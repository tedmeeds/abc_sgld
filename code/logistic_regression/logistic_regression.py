import numpy as np
import pylab as pp
import scipy as sp
import pdb
import gzip, cPickle
import time
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

def load_mnist(digits=None):
  f = gzip.open('../../data/mnist.pkl.gz', 'rb')
  data = cPickle.load(f)

  if digits is not None:
    Xtrain = []
    Xvalid = np.array( ([]))
    Xtest  = np.array( ([]))
    k = 0
    for digit in digits:
      train_ids = pp.find( data[0][1] == digit )
      valid_ids = pp.find( data[1][1] == digit )
      test_ids  = pp.find( data[2][1] == digit )

      if digit == digits[0]:
        Xtrain = data[0][0][train_ids,:]
        Xvalid = data[1][0][valid_ids,:]
        Xtest  = data[2][0][test_ids,:]
        Ttrain = k*np.ones(len(train_ids))
        Tvalid = k*np.ones(len(valid_ids))
        Ttest = k*np.ones(len(test_ids))
      else:
        Xtrain = np.vstack( ( Xtrain, data[0][0][train_ids,:] ))
        Xvalid = np.vstack( ( Xvalid, data[1][0][valid_ids,:] ))
        Xtest = np.vstack( ( Xtest, data[2][0][test_ids,:] ))

        Ttrain = np.hstack( ( Ttrain, k*np.ones(len(train_ids)) ))
        Tvalid = np.hstack( ( Tvalid, k*np.ones(len(valid_ids)) ))
        Ttest = np.hstack( ( Ttest, k*np.ones(len(test_ids)) ))
      k += 1
    return (Xtrain,Ttrain),(Xvalid,Tvalid),(Xtest,Ttest)
  return data

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
  def __init__( self, T_mat, X_mat, T_test, X_test, std ):
    self.sp_eps = 0.01

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
    self.W = np.random.normal(0, std, (self.D, self.K))

    self.batchsize = 100
    assert N == self.N, "should be same shape"

  def loglikelihood( self, W, X = None, ids = None, return_Y = False ):
    if X is None:
      X = self.X

    if ids is None:
      Y, logY = softmax( np.dot( self.X, W ), return_log = True )

      log_like_per_n = np.sum( self.T * logY, 1 )

      loglike_data = np.sum( log_like_per_n )/self.N

      if return_Y:
        return loglike_data, Y
      else:
        return loglike_data
    else:
      Y, logY = softmax( np.dot( self.X[ids,:], W ), return_log = True )

      log_like_per_n = np.sum( self.T[ids,:] * logY, 1 )

      loglike_data = np.sum( log_like_per_n )/len(ids)

      if return_Y:
        return loglike_data, Y
      else:
        return loglike_data

  def loglikelihood_ema_tt( self, W, X = None, ids = None, return_Y = False ):
    if X is None:
      X = self.X

    if ids is None:
      Y, logY = softmax( np.dot( self.X, W ), return_log = True )

      log_like_per_n = np.sum( self.T * logY, 1 )

      loglike_data = np.sum( log_like_per_n )/self.N

      #pdb.set_trace()
      if return_Y:
        return loglike_data, Y
      else:
        return loglike_data
    else:
      Y, logY = softmax( np.dot( self.X[ids,:], W ), return_log = True )

      log_like_per_n = np.sum( self.T[ids,:] * logY, 1 )

      loglike_data = np.sum( log_like_per_n )/len(ids)
      #pdb.set_trace()
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

  def gradient_ema_targets( self, W):
    new_ids = np.random.permutation( self.N )[:self.batchsize]
    #ids = self.ids.copy()
    #self.ids[:self.batchsize/4] = new_ids[:self.batchsize/4]
    self.ids = new_ids
    ids = self.ids

    Y = softmax( np.dot( self.X[ids,:], W ) )

    DT = np.sum( self.T[ids,:]*Y, 1 )

    #DIF = self.TT[ids,:]-Y

    G = np.zeros( (self.D,self.K) ) #- np.dot(self.X[ids,:].T, DT )

    for k in range(self.K):
      g_k = np.dot( self.T[ids,k], self.X[ids,:] ) - np.dot(self.X[ids,:].T, DT )
      #np.dot( DIF[:,k].T, self.X[ids,:] ).T
      G[:,k] += g_k
    return self.batchsize*G/self.N

  def sp_gradient( self, W, Y = None ):



    G = np.zeros( (self.D,self.K))
    nrepeats = 10
    new_ids = np.random.permutation( self.N )[:self.batchsize]
    for i in range(nrepeats):

      #ids = self.ids.copy()
      #self.ids[:self.batchsize/4] = new_ids[:self.batchsize/4]
      ids=new_ids

      mask = 2*np.random.binomial(1,0.5,W.shape)-1

      W1   = W + self.sp_eps*mask
      W2   = W - self.sp_eps*mask

      #Y = softmax( np.dot( self.X, W ) )

      LL_plus = self.loglikelihood_ema_tt( W1, ids = ids )
      LL_minus = self.loglikelihood_ema_tt( W2, ids = ids )
      #DIF = self.T-Y

      G += (LL_plus-LL_minus)/(2*self.sp_eps*mask)

    return G/nrepeats

  def gradient_ascent( self, lr, decay, mom, init_w_std, max_steps = 10 ):
    self.ids = np.random.permutation( self.N )[:self.batchsize]

    self.W = init_w_std * np.random.randn( self.D, self.K)
    LL, Y = self.loglikelihood( self.W, return_Y = True )

    Ytest, logYtest = softmax( np.dot( self.Xtest, self.W ), return_log = True )
    y = np.argmax(Y,1)
    y_test = np.argmax(Ytest,1)
    train_error = classification_error( self.t_train, y )
    test_error = classification_error( self.t_test, y_test )

    last_time = time.time()

    tt_mom = 0.5
    self.TT = self.T.copy()
    #mom=0.9
    count = 0
    last_train = train_error
    GM = np.zeros( self.W.shape )
    print "INIT LL %0.4f train_error %0.3f  test_error %0.3f"%(LL, train_error, test_error)
    for t in xrange( max_steps ):
      G = self.sp_gradient( self.W )
      #Y = softmax( np.dot( self.X[self.ids,:], self.W ) )
      #G = self.gradient_ema_targets( self.W )
      # G = self.gradient( self.W )
      #self.W = self.W + lr*self.gradient( self.W, Y )

      #self.TT[self.ids,:] = tt_mom*self.T[self.ids,:] + (1-tt_mom)*Y

      GM = (1-mom)*GM + mom*G
      self.W = self.W + lr*GM

      lr *= decay #pow(lr,decay)
      cur_time = time.time()

      if cur_time - last_time > 15:
        Ytest, logYtest = softmax( np.dot( self.Xtest, self.W ), return_log = True )
        LL, Y = self.loglikelihood( self.W, return_Y = True )
        y = np.argmax(Y,1)
        y_test = np.argmax(Ytest,1)
        train_error = classification_error( self.t_train, y )
        test_error = classification_error( self.t_test, y_test )


        if train_error > last_train:
          count += 1
        else:
          count = 0
        if count > 1:
          lr *=0.75
          count = 0
        last_time = time.time()

        print "%4d LL %0.4f grad %g\t lr %g train_error %0.5f %%  test_error %0.5f %%"%(t+1,LL,G[0][0], lr, train_error*100, test_error*100)
        last_train = train_error
      #self.sp_eps *= 0.99

if __name__ == "__main__":
  #assert False, "add binary label per class, compare log-likelihoods, classification error"
  np.random.seed(1)
  K = 10
  lr = 0.1
  decay = 0.99999
  init_w_std = 0.001
  mom = 0.9
  max_steps = 300000
  digits = [0,1]
  ndigits = len(digits)
  K = ndigits
  mnist = load_mnist(digits)
  (X_train,t_train),(X_valid,t_valid),(X_test,t_test) = mnist
  T_train = onehot( t_train, ndigits )
  T_valid = onehot( t_valid, ndigits )
  T_test  = onehot( t_test, ndigits )
  m = X_train.mean(0)
  s = X_train.std(0)
  ok = pp.find(s>0)
  X_train -= m
  X_valid -= m
  X_test  -= m
  X_train[:,ok] /= s[ok]
  X_valid[:,ok] /= s[ok]
  X_test[:,ok] /= s[ok]

  # print "constructing LR model..."
  # #LR_model = MulticlassLogisticRegression( T_valid, X_valid, T_test, X_test )
  LR_model = MulticlassLogisticRegression( T_train, X_train, T_test, X_test )
  # LR_model2 = MulticlassLogisticRegression( T_train, X_train, T_test, X_test )
  # LR_model3 = MulticlassLogisticRegression( T_train, X_train, T_test, X_test )
  # LR_model4 = MulticlassLogisticRegression( T_train, X_train, T_test, X_test )
  #
  # models = [LR_model, LR_model2, LR_model3, LR_model4]
  #ALR = AveragedLogisticRegression(models)
  # print "gradient ascent..."
  # ALR.gradient_ascent(lr, decay, mom, init_w_std, max_steps)

  LR_model.gradient_ascent(lr, decay, mom, init_w_std, max_steps)
