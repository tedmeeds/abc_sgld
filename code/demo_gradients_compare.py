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
  D = X_train.shape[1]
  N         = len(T_train)
  
  m = X_train.mean(0)
  X_train -= m
  X_valid -= m
  X_test  -= m
  
  mom = 0.5
  max_iters = 5000
  batchsize = 100
  alpha     = 0.0001
  gamma     = 0.999
  
  q = 100
  c = 0.01
  
  # print "constructing LR model..."
  # #LR_model = MulticlassLogisticRegression( T_valid, X_valid, T_test, X_test )
  LR_model = MulticlassLogisticRegression( T_train, X_train, T_test, X_test )
  
  bcompare = True
  np.random.seed(1)
  w = 0.0001*np.random.randn( D*K )
  sg_params = { "ml_problem":LR_model, "max_iters":max_iters, "N":N, "batchsize":batchsize, "alpha":alpha, "gamma":gamma,"mom":mom, "l1":100.0,"l2":0.0,"q":q,"c":c,"compare":bcompare}
  wout, errors = sgd( w, sg_params )
    
  #LR_model.gradient_ascent(lr, decay, mom, init_w_std, max_steps)
  

  # max_iters = 500000
  # q         = 1
  # c         = 0.2
  # N         = len(T_train)
  # batchsize = 10
  # alpha     = 0.1
  # gamma     = 0.9
  #
  # #cs = [0.05,0.1,0.2,0.3]
  # cs = [0.01]
  # gammas = [0.99999]
  # moms = [0.5]
  # batchsizes = [20]
  # qs = [10]
  # result = []
  # batchreplaces = [1]
  # for c in cs:
  #   alpha = 3*c/(4*N)
  #   for gamma in gammas:
  #     for mom in moms:
  #       for batchsize in batchsizes:
  #         for q in qs:
  #           for batchreplace in batchreplaces:
  #             np.random.seed(1)
  #             w = 0.0001*np.random.randn( D*K )
  #             spall_params = {"ml_problem":LR_model, "max_iters":max_iters, "q":q,"c":c,"N":N, "batchsize":batchsize, "alpha":alpha, "gamma":gamma,"mom":mom,"batchreplace":batchreplace, "l1":100.0,"l2":0.0}
  #             wout, errors = spall( w, spall_params )
  #
  #             result.append({"c":c,"alpha":alpha,"gamma":gamma, "mom":mom, "batchsize":batchsize,"q":q,"errors":errors,"w":wout})
  #
  # print "-------------------------------------------"
  # vals = []
  # i=0
  # for r in result:
  #   train_error = r["errors"][-1][0]  # train error
  #   vals.append(train_error)
  # vals = np.array(vals)
  # iorder = np.argsort(vals)
  #
  # for idx in iorder:
  #   print "error = %0.4f   c= %0.4f   a = %0.4f   mom = %0.4f  g = %0.4f b = %d  q = %d"%( vals[idx], result[idx]["c"], result[idx]["alpha"],result[idx]["mom"], result[idx]["gamma"], result[idx]["batchsize"], result[idx]["q"])
  
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