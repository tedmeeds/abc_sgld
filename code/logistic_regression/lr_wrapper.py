import numpy as np
from logistic_regression import *

class LogisticRegressionProblem(object):
    def __init__(self, lr, K):
        self.lr = lr
        self.N = len(lr.X)
        self.K = K

    def loglike_x(self, X, omega):
        return self.lr.loglikelihood(self.lr.W, ids=omega)

    def simulate(self, theta, seed = None, S = 1):
        self.lr.W = theta.reshape(28*28, self.K)
        return softmax( np.dot( self.lr.X[seed, :], theta.reshape(28*28, self.K) ), return_log = False )

    def true_gradient(self, theta, omega):
        gamma = params['prior']['gamma']
        self.lr.W = theta.reshape(28*28, self.K)
        Y = softmax( np.dot( self.lr.X[omega,:], theta.reshape(28*28, self.K) ) )
        DIF = self.lr.T[omega,:]-Y
        G = np.zeros( (28*28, self.K) )
        for k in range(self.K):
          g_k = np.dot( DIF[:,k].T, self.lr.X[omega,:] ).T
          G[:,k] = g_k
        G = (len(omega)*G/self.N).flatten()
        G -= gamma*theta
        return G

    def two_sided_keps_gradient(self, theta, d_theta, omega, S, params):
        R = params['2side_keps']['R']
        percent_change = params['2side_keps']["percent_to_change"]
        prior_penalty = params['2side_keps']['prior_penalty']
        gamma = params['prior']['gamma']
        gradient = 0.0*theta
        for r in range(R):
            delta = 2*np.random.binomial(1, 0.5, theta.shape)-1
            self.lr.W = (theta + d_theta*delta).reshape(28*28, self.K)
            f_plus  = self.loglike_x( None, omega )
            self.lr.W = (theta - d_theta*delta).reshape(28*28, self.K)
            f_minus = self.loglike_x( None, omega )
            gradient += (f_plus-f_minus) / delta
        gradient /= 2*d_theta*R
        gradient -= gamma*theta
        return -gradient

    def propose(self, theta, params):
        q = params['propose']['q']
        return np.random.multivariate_normal(theta, q**2*np.eye(len(theta)))

    def loglike_prior(self, theta, params):
        gamma = params['prior']['gamma']
        return -0.5*gamma*theta**2

