import gzip, cPickle
import numpy as np
import pylab as pp

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

def onehot( t, K ):
  N=len(t)
  T = np.zeros( (N,K))

  for trow,tk in zip( T, t ):
    trow[tk] = 1.0
  return T


def load_mnist():
  file = gzip.open('./data/mnist.pkl.gz', 'rb')
  (X_train, t_train), (X_valid, t_valid), (X_test, t_test) = cPickle.load(file)
  num_classes = 10

  T_train = onehot(t_train, num_classes)
  T_valid = onehot(t_valid, num_classes)
  T_test  = onehot(t_test, num_classes)

  mean = X_train.mean(0)
  X_train -= mean
  X_valid -= mean
  X_test  -= mean

  return (X_train, T_train), (X_valid, T_valid), (X_test, T_test)

