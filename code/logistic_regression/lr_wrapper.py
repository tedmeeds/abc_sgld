import numpy as np
from logistic_regression import *

class LogisticRegressionProblem(object):
    def __init__(self, lr):
        self.lr = lr
        self.N = len(lr.X)
        self.prior_penalty = 0

    def loglike_x(self, X, omega):
        return self.lr.loglikelihood(self.lr.W, ids=omega)

    def simulate(self, theta, seed = None, S = 1):
        return softmax( np.dot( self.lr.X[seed, :], theta.reshape(28*28, 10) ), return_log = False )

    def true_gradient(self, theta, omega):
        Y = softmax( np.dot( self.lr.X[omega,:], theta.reshape(28*28, 10) ) )
        DIF = self.lr.T[omega,:]-Y
        G = np.zeros( (28*28, 10) )
        for k in range(10):
          g_k = np.dot( DIF[:,k].T, self.lr.X[omega,:] ).T
          G[:,k] = g_k
          # G[:,k] = np.log(g_k+1e10)
        return (len(omega)*G/self.N).flatten()

    def two_sided_keps_gradient(self, theta, d_theta, omega, S, params):
        # return -self.true_gradient(theta, omega)
        R = params['2side_keps']['R']
        percent_change = params['2side_keps']["percent_to_change"]
        gradient = 0.0*theta
        # change_mask = np.zeros(theta.shape)
        # perm = np.random.permutation( len(theta) )[:int(percent_change*len(theta))]
        # change_mask[perm] = 1
        for r in range(R):
            delta = 2*np.random.binomial(1, 0.5, theta.shape)-1
            self.lr.W = (theta + d_theta*delta).reshape(28*28, 10)
            f_plus  = self.loglike_x( None, omega )
            self.lr.W = (theta - d_theta*delta).reshape(28*28, 10)
            f_minus = self.loglike_x( None, omega )
            # gradient += (f_plus-f_minus) * delta * change_mask
            gradient += (f_plus-f_minus) / delta
        gradient /= 2*d_theta*R
        gradient_prior = -self.prior_penalty*np.sign(theta)
        gradient += gradient_prior
        return -gradient

