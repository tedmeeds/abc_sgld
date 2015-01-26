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
  print "loading mnist..."
  (X_train, T_train), (X_valid, T_valid), (X_test, T_test) = load_mnist()
  num_training, dimension = X_train.shape
  num_classes = T_train.shape[1]

  LR_model = MulticlassLogisticRegression( T_train, X_train, T_test, X_test, l1,  l2 )
  covariance = np.cov( X_train.T )
  f=0.1
  #cc = f*np.sqrt(1.0/(np.diag( covariance )+f*f))
  cc = f*np.sqrt((np.diag( covariance )+f)/f)
  ccc= np.hstack(tuple([cc]*num_classes))
  #ccc=0.01

  max_iters = 10000
  q         = 1
  c         = 1
  N         = len(T_train)
  batchsize = 10
  alpha     = 0.1
  gamma     = 0.9
  mom_beta1 = 0.9 # on gradient
  mom_beta2 = 0.9 # on gradient_squared
  
  #cs = [0.5,0.1,0.2,0.3]
  cs = [ccc]
  gammas = [0.9999]
  moms = [0.0]
  batchsizes = [100]
  qs = [10]
  result = []
  batchreplaces = [1]
  for c in cs:
    # for "grad"
    #alpha = 0.01/N
    
    # for others
    alpha = 0.01 #e-1 #*3*c/(4)
    
    for gamma in gammas:
      for mom in moms:
        for batchsize in batchsizes:
          for q in qs:
            for batchreplace in batchreplaces:
              np.random.seed(1)
              w = 0.000001*np.random.randn( D*K )
              spall_params = {  "ml_problem":LR_model, 
                                "max_iters":max_iters, 
                                "q":q,
                                "c":c,
                                "N":N, 
                                "batchsize":batchsize, "q":q,
                                "c":c,
                                "alpha":alpha, 
                                "gamma_alpha":0.9999,
                                "gamma_c":0.9999,
                                "gamma_eps":0.9999,
                                "mom_beta1":mom_beta1,
                                "mom_beta2":mom_beta2,
                                "update_method":"adam",
                                "batchreplace":batchreplace, 
                                "diag_add":0.01}
              
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
