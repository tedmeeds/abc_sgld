import numpy as np
import pylab as pp
import scipy as sp
import gzip, cPickle
import time
from numpy import newaxis

from ml_problems import *
from sa_algorithms import *

def onehot( t, K ):
  N=len(t)
  T = np.zeros( (N,K))
  
  for trow,tk in zip( T, t ):
    trow[tk] = 1.0
  return T
def load_mnist():
  f = gzip.open('../../data/mnist.pkl.gz', 'rb')
  data = cPickle.load(f)
  return data
  
if __name__ == "__main__":
  #assert False, "add binary label per class, compare log-likelihoods, classification error"
  np.random.seed(1)
  K = 10
  lr = 0.002
  decay = 0.99999
  init_w_std = 0.000001
  mom = 0.0
  max_steps = 300000
  print "loading mnist..."
  (X_train,t_train),(X_valid,t_valid),(X_test,t_test) = load_mnist()
  print "one-hotting labels.."
  T_train = onehot( t_train, K )
  T_valid = onehot( t_valid, K )
  T_test  = onehot( t_test, K )
  
  m = X_train.mean(0)
  X_train -= m
  X_valid -= m
  X_test  -= m
  
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
  
  #LR_model.gradient_ascent(lr, decay, mom, init_w_std, max_steps)
  D = X_train.shape[1]

  C = np.cov( X_train.T )
  f=0.1
  #cc = f*np.sqrt(1.0/(np.diag( C )+f*f)) 
  cc = f*np.sqrt((np.diag( C )+f)/f)
  ccc= np.hstack( (cc,cc,cc,cc,cc,cc,cc,cc,cc,cc))
  #ccc=0.01
  max_iters = 5000
  q         = 1
  c         = 1
  N         = len(T_train)
  batchsize = 10
  alpha     = 0.1
  gamma     = 0.9
  
  #cs = [0.5,0.1,0.2,0.3]
  cs = [ccc]
  gammas = [0.9999]
  moms = [0.5]
  batchsizes = [50]
  qs = [20]
  result = []
  batchreplaces = [1]
  for c in cs:
    alpha = 3*0.1/(4*N)
    for gamma in gammas:
      for mom in moms:
        for batchsize in batchsizes:
          for q in qs:
            for batchreplace in batchreplaces:
              np.random.seed(1)
              w = 0.0001*np.random.randn( D*K )
              spall_params = {"ml_problem":LR_model, "max_iters":max_iters, "q":q,"c":c,"N":N, "batchsize":batchsize, "alpha":alpha, "gamma":gamma,"mom":mom,"batchreplace":batchreplace, "l1":0.0,"l2":0.0,"diag_add":0.01}
              
              wout, errors = spall( w, spall_params )
              #wout, errors = spall_with_hessian( w, spall_params )
      
              result.append({"c":c,"alpha":alpha,"gamma":gamma, "mom":mom, "batchsize":batchsize,"q":q,"errors":errors,"w":wout})
  
  print "-------------------------------------------"
  vals = []
  i=0
  for r in result:
    train_error = r["errors"][-1][0]  # train error 
    vals.append(train_error)
  vals = np.array(vals)
  iorder = np.argsort(vals)
  
  for idx in iorder:
    print "error = %0.4f   c= %0.4f   a = %0.4f   mom = %0.4f  g = %0.4f b = %d  q = %d"%( vals[idx], result[idx]["c"], result[idx]["alpha"],result[idx]["mom"], result[idx]["gamma"], result[idx]["batchsize"], result[idx]["q"]) 
  
  # experiments:
  #  1) full gradient descent with gradients (MAP)
  #  2) SGD gradient descent with varying batchsizes
  #  3) SAG descent with batchsize
  #  4) adaboost full SGD, other boost SGD
  #  5) FDSA (finite difference)
  #  6) SPSA varying q, c, alpha, batchsize
  #  7) SPSA adaboost, otherboost
  #  8) SGLD gradients with SGD/batchsize
  #  9) laplace approximation
  # 10) SGLD with SPSA