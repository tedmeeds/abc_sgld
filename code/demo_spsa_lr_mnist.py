import numpy as np
import pylab as pp
import scipy as sp
import itertools
import time
from helpers import *
from numpy import newaxis

from ml_problems import *
from sa_algorithms import *

if __name__ == "__main__":
  #assert False, "add binary label per class, compare log-likelihoods, classification error"
  np.random.seed(1)
  # lr = 0.002
  # decay = 0.99999
  # init_w_std = 0.000001
  # max_steps = 300000
  print "loading mnist..."
  (X_train, T_train), (X_valid, T_valid), (X_test, T_test) = load_mnist()
  num_training, dimension = X_train.shape
  num_classes = T_train.shape[1]

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

  covariance = np.cov( X_train.T )
  f=0.1
  #cc = f*np.sqrt(1.0/(np.diag( covariance )+f*f))
  cc = f*np.sqrt((np.diag( covariance )+f)/f)
  ccc= np.hstack(tuple([cc]*num_classes))
  #ccc=0.01
  max_iters = 5000
  #cs = [0.5,0.1,0.2,0.3]
  cs = [ccc]
  gammas = [0.9999]
  moms = [0.5]
  batchsizes = [50]
  qs = [20]
  batchreplaces = [1]
  alpha = 3*0.1/(4*num_training)

  results = []
  for c, gamma, mom, batchsize, q, batchreplace in itertools.product(cs, gammas, moms, batchsizes, qs, batchreplaces):
    weights = 0.0001*np.random.randn(dimension*num_classes)
    spall_params = {
      "ml_problem":LR_model,
      "max_iters":max_iters,
      "q":q,
      "c":c,
      "N":num_training,
      "batchsize":batchsize,
      "alpha":alpha,
      "gamma":gamma,
      "mom":mom,
      "batchreplace":batchreplace,
      "l1":0.0,
      "l2":0.0,
      "diag_add":0.01
    }

    wout, errors = spall(weights, spall_params)
    #wout, errors = spall_with_hessian( w, spall_params )

    results.append({
      "c":c,
      "alpha":alpha,
      "gamma":gamma,
      "mom":mom,
      "batchsize":batchsize,
      "q":q,
      "errors":errors,
      "w":wout
    })

  print "-"*50

  values = np.array([r["errors"][-1][0] for r in results])
  iorder = np.argsort(values)

  for idx in iorder:
    print "error: %0.4f" % values[idx]
    print "parameters: " + str(results[idx])
    #print "error = %0.4f   c= %0.4f   a = %0.4f   mom = %0.4f  g = %0.4f b = %d  q = %d"%( values[idx], results[idx]["c"], results[idx]["alpha"],results[idx]["mom"], results[idx]["gamma"], results[idx]["batchsize"], results[idx]["q"])

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
