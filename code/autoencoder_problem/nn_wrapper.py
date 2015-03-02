import numpy as np
from nn import one_hot
# Theta=weights
class NeuralNetworkProblem(object):
    def __init__(self, nn, data):
        self.nn = nn
        (self.X_train, self.T_train), (self.X_valid, self.T_valid), (self.X_test, self.T_test) = data
        self.N = len(self.X_train)
        self.training_set, self.validation_set, self.test_set = data
        self.prior_penalty = 0.001

    # Skip
    def prior_rand(self):
        pass

    # Skip
    def propose(self, theta):
        pass

    def loglike_x(self, X, omega, simulated=True):
        Y = self.X_train[omega,:].T
        # Y = []
        # for i in omega:
        #     Y.append(self.training_set[i][1])
        # # Y = self.training_set[omega, 1]
        # assert len(X) == len(Y)
        # likelihood = 0.0
        # for i in range(len(X)):
        #     x = X[i]
        #     y = Y[i]
        #     if simulated:
        #         x[x < 0.5] = 0
        #         x[x >= 0.5] = 1
        #         y_copy = y.copy()
        #         y_copy[y_copy <= 0.5] = 0
        #         y_copy[y_copy > 0.5] = 1
        #         # (??) Take log?
        #         likelihood += np.sum(x == y_copy) / float(len(y_copy))
        #     else:
        #         # We are given a sample so we don't need to do feedforward
        #         likelihood += np.nan_to_num(np.sum(-y*np.log(x+10**-10)-(1-y)*np.log(1-x+10**-10)))

        n = len(omega)
        likelihood_batch = Y*np.log( X + 1e-10 ) + (1-Y)*np.log(1 - X + 1e-10)
        likelihood = self.N*likelihood_batch.sum()/n
        return likelihood

    # Log prior for some theta
    def loglike_prior(self, theta):
        pass

    # Log posterior for some theta
    def loglike_posterior(self, theta):
        pass

    # Skip
    def loglike_proposal_theta(self, to_theta, from_theta):
        pass

    # def simulate(self, theta, seed = None, S = 1):
    #     samples = []
    #     for batch_id in seed:
    #         x = self.training_set[batch_id][0]
    #         output = self.forwardpass(theta, x)
    #         u = np.random.uniform(0, 1, output.shape)
    #         larger = u > output
    #         smaller = u < output
    #         output[larger] = 1
    #         output[smaller] = 0
    #         samples.append(output)
    #     return np.array(samples)

    def simulate(self, theta, seed = None, S = 1):
        self.set_weights( theta )
        samples = self.forwardpass( self.X_train[seed,:])
        # samples = []
        # for batch_id in seed:
        #     x = self.training_set[batch_id][0]
        #     output = self.forwardpass(theta, x)
        #     u = np.random.uniform(0, 1, output.shape)
        #     larger = u > output
        #     smaller = u < output
        #     output[larger] = 1
        #     output[smaller] = 0
        #     samples.append(output)
        return np.array(samples)

    # Use backprop, probably should use some regularization or learning rate
    def true_gradient(self, theta, omega):
        self.set_weights(theta)
        grad_b = 0.0*self.nn.biases
        grad_w = 0.0*self.nn.weights
        for i in omega:
            x = self.X_train[i][:,np.newaxis]
            y = self.T_train[i]
            self.nn.feedforward(x)
            gb, gw = self.nn.backpropagation(x, y)
            grad_b += gb
            grad_w += gw
        grad_b /= len(omega)
        grad_w /= len(omega)

        return self.flatten(grad_b, grad_w)

    # Does NN have this?
    def true_abc_gradient(self, theta, d_theta, S, params):
        pass

    # This is a helper function for exp_wrapper
    def simulate_for_gradient(self, theta, d_theta, omega, S, params):
        pass

    # (??) What do we do with S?
    # def two_sided_keps_gradient(self, theta, d_theta, omega, S, params):
    #     return self.true_gradient(theta, omega)
    #     R = params['2side_keps']['R']
    #     gradient = 0.0*theta
    #     for r in range(R):
    #         delta = [0]*len(theta)
    #         for i in range(len(theta)):
    #             delta[i] = 2*np.random.binomial(1, 0.5, theta[i].shape)-1
    #         delta = np.array(delta)
    #         x_plus = np.array([self.forwardpass(theta + delta, self.training_set[i][0]) for i in omega])
    #         x_minus = np.array([self.forwardpass(theta - delta, self.training_set[i][0]) for i in omega])
    #         # Should take the mean of all x_plus and x_minus
    #         # How to multiply eps
    #         grad = (self.loglike_x(x_plus, omega, False) - self.loglike_x(x_minus, omega, False)) / delta
    #         # print grad
    #         gradient += grad
    #     gradient /= 2*d_theta*R
    #     gradient_prior = np.sign(theta) # (??) sign of the weights?
    #     gradient += gradient_prior
    #     return -gradient
    def two_sided_keps_gradient(self, theta, d_theta, omega, S, params):
        # return -self.true_gradient(theta, omega)
        R = params['2side_keps']['R']
        percent_change = params['2side_keps']["percent_to_change"]
        gradient = 0.0*theta
        change_mask = np.zeros(theta.shape)
        perm = np.random.permutation( len(theta) )[:int(percent_change*len(theta))]
        change_mask[perm] = 1
        for r in range(R):
            delta = 2*np.random.binomial(1, 0.5, theta.shape)-1
            self.set_weights( theta + d_theta*delta * change_mask )
            x_plus = self.forwardpass( self.X_train[omega,:])
            self.set_weights( theta - d_theta*delta * change_mask )
            x_minus = self.forwardpass( self.X_train[omega,:])
            f_plus  = self.loglike_x( x_plus, omega )
            f_minus = self.loglike_x( x_minus, omega )
            gradient += (f_plus-f_minus) * delta * change_mask
        gradient /= 2*d_theta*R
        gradient_prior = -self.prior_penalty*np.sign(theta)
        gradient += gradient_prior
        return -gradient

    # For now skip this I don't have a syntetic likelihood
    def two_sided_sl_gradient(self, theta, d_theta, omega, S, params):
        pass

    # Get true posterior (can we do this for NN?)
    # working_code doens't call this
    def posterior(self, thetas):
        pass

    # Helper functions
    def forwardpass(self, x):
        return self.nn.feedforward(x.T)

    def set_weights( self, theta ):
        biases, weights = self.unpack_theta(theta)
        self.nn.biases = biases
        self.nn.weights = weights

    # theta is flattened, and we reconstruct it so the NN can use it
    def unpack_theta(self, theta):
        biases = []
        weights = []
        prev_size = 0
        for i in range(1, len(self.nn.config)):
            size = self.nn.config[i]
            x = theta[prev_size:prev_size+size]
            z = x.reshape(size, 1)
            biases.append(z)
            prev_size = size
        for i in range(len(self.nn.config)-1):
            l1 = self.nn.config[i]
            l2 = self.nn.config[i+1]
            size = l1*l2
            x=theta[prev_size:prev_size+size]
            z = x.reshape(l2, l1)
            weights.append(z)
            prev_size = size
        return np.array(biases), np.array(weights)

    def flatten(self, b, w):
        theta = np.concatenate([b[i].flatten() for i in range(len(b))])
        theta = np.append(theta, np.concatenate([w[i].flatten() for i in range(len(w))]))
        return theta

