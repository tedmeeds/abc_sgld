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
        self.lr.W = theta.reshape(28*28, self.K)
        Y = softmax( np.dot( self.lr.X[omega,:], theta.reshape(28*28, self.K) ) )
        DIF = self.lr.T[omega,:]-Y
        G = np.zeros( (28*28, self.K) )
        for k in range(self.K):
          g_k = np.dot( DIF[:,k].T, self.lr.X[omega,:] ).T
          G[:,k] = g_k
          # G[:,k] = np.log(g_k+1eself.K)
        return (len(omega)*G/self.N).flatten()

    def two_sided_keps_gradient(self, theta, d_theta, omega, S, params):
        # return -self.true_gradient(theta, omega)
        R = params['2side_keps']['R']
        percent_change = params['2side_keps']["percent_to_change"]
        prior_penalty = params['2side_keps']['prior_penalty']
        gamma = params['prior']['gamma']
        gradient = 0.0*theta
        # change_mask = np.zeros(theta.shape)
        # perm = np.random.permutation( len(theta) )[:int(percent_change*len(theta))]
        # change_mask[perm] = 1
        for r in range(R):
            delta = 2*np.random.binomial(1, 0.5, theta.shape)-1
            self.lr.W = (theta + d_theta*delta).reshape(28*28, self.K)
            f_plus  = self.loglike_x( None, omega )
            self.lr.W = (theta - d_theta*delta).reshape(28*28, self.K)
            f_minus = self.loglike_x( None, omega )
            # gradient += (f_plus-f_minus) * delta * change_mask
            gradient += (f_plus-f_minus) / delta
        gradient /= 2*d_theta*R
        gradient = -gamma*gradient/2.0
        # gradient_prior = -prior_penalty*np.sign(theta) # TODO: Replace this?
        # gradient += gradient_prior
        return -gradient

    def propose(self, theta, params):
        q = params['propose']['q']
        return np.random.multivariate_normal(theta, q**2*np.eye(len(theta)))

    def loglike_prior(self, theta, params):
        gamma = params['prior']['gamma']
        return -gamma*theta/2.0

