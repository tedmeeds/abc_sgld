import numpy as np
import pdb
# Theta=weights
class NeuralNetworkProblem(object):
    def __init__(self, nn, data):
        self.nn = nn
        (self.X_train, self.T_train), (self.X_valid, self.T_valid), (self.X_test, self.T_test) = data
        self.N = len(self.X_train)
        #self.training_set, self.validation_set, self.test_set = data
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

    # Use backprop
    def true_gradient(self, theta, omega):
        self.nn.weights = theta
        grad_b = 0*self.nn.biases
        grad_w = 0*theta
        for i in omega:
            x, y = self.training_set[i]
            gb, gw = self.nn.backpropagation(x, y)
            grad_b += gb
            grad_w += gw
        # Return mean
        return grad_w / len(omega)

    # Does NN have this?
    def true_abc_gradient(self, theta, d_theta, S, params):
        pass

    # This is a helper function for exp_wrapper
    def simulate_for_gradient(self, theta, d_theta, omega, S, params):
        pass

    # (??) What do we do with S?
    def two_sided_keps_gradient(self, theta, d_theta, omega, S, params):
        #return -self.true_gradient(theta, omega)
        R = params['2side_keps']['R']
        gradient = 0.0*theta
        
        percent_change = params["percent_to_change"]

        change_mask = [0]*len(theta)
        for i in range(len(theta)):
            
            #delta[i] = 2*np.random.binomial(1, 0.5, theta[i].shape)-1
            change_mask[i] = np.zeros(theta[i].shape)
            sz = theta[i].shape[0]*theta[i].shape[1]
            ids_to_change = np.random.permutation( sz )[:int(percent_change*sz)]
            changes = np.zeros(sz)
            changes[ids_to_change]=1
            change_mask[i] = changes.reshape(theta[i].shape)
        change_mask = np.array(change_mask)    
        for r in range(R):
            delta   = [0]*len(theta)
            #change_mask = [0]*len(theta)
            for i in range(len(theta)):
                
                delta[i] = 2*np.random.binomial(1, 0.5, theta[i].shape)-1
                #change_mask[i] = delta[i]*0
                #sz = theta[i].shape[0]*theta[i].shape[1]
                #ids_to_change = np.random.permutation( sz )[:int(percent_change*sz)]
                #changes = np.zeros(sz)
                #changes[ids_to_change]=1
                #change_mask[i] = changes.reshape(theta[i].shape)
                #pdb.set_trace()
            delta = np.array(delta)
            
            
            self.set_weights( theta + d_theta*delta*change_mask )
            x_plus = self.forwardpass( self.X_train[omega,:]) 
            #pdb.set_trace()
            self.set_weights( theta - d_theta*delta*change_mask )
            x_minus = self.forwardpass( self.X_train[omega,:]) 
            #x_plus = np.array([self.forwardpass(theta + delta, self.training_set[i][0]) for i in omega])
            #x_minus = np.array([self.forwardpass(theta - delta, self.training_set[i][0]) for i in omega])
            # Should take the mean of all x_plus and x_minus
            # How to multiply eps
            f_plus  = self.loglike_x( x_plus, omega )
            f_minus = self.loglike_x( x_minus, omega )
            #grad = (self.loglike_x(x_plus, omega, False) - self.loglike_x(x_minus, omega, False)) / delta
            # print grad
            gradient += (f_plus-f_minus) * change_mask*delta
        gradient /= 2*d_theta*R
        #gradient_prior = -self.prior_penalty*np.array([np.sign(theta[0]), np.sign(theta[1])]) # (??) sign of the weights?
        gradient_prior = -self.prior_penalty*np.array([theta[0], theta[1]]) # (??) sign of the weights?
        gradient += gradient_prior
        return -gradient


    # For now skip this I don't have a syntetic likelihood
    def two_sided_sl_gradient(self, theta, d_theta, omega, S, params):
        pass

    # Get true posterior (can we do this for NN?)
    # working_code doens't call this
    def posterior(self, thetas):
        pass

    def set_weights( self, theta ):
      biases, weights = self.unpack_theta(theta)
      self.nn.weights = weights
      self.nn.biases = biases
      
    # Helper functions
    def forwardpass(self, x):
      return self.nn.feedforward(x.T)

    # TODO: Given theta return bias/weights
    def unpack_theta(self, theta):
        return self.nn.biases, theta
