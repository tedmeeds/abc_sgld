import numpy as np
from sgld import SGLDOptimizer


class SGLDSpallOptimizer(SGLDOptimizer):
    def __init__(self, problem, num_repeats, epsilon):
        super(SGLDSpallOptimizer, self).__init__(problem)
        self.likelihood_gradient = self.likelihood_spall_gradient
        self.num_repeats = num_repeats
        self.epsilon = epsilon

    def likelihood_spall_gradient(self, batch_indices, params):
        gradient = np.zeros(self.num_params)
        for i in range(self.num_repeats):
            mask = 2*np.random.binomial(1, 0.5, self.num_params)-1
            params_negative = params - self.epsilon*mask
            params_positive = params + self.epsilon*mask
            likelihood_negative = self.problem.likelihood(batch_indices,
                                                          params_negative)
            likelihood_positive = self.problem.likelihood(batch_indices,
                                                          params_positive)
            gradient += (likelihood_positive-likelihood_negative) / \
                        (2*self.epsilon*mask)
        return gradient / self.num_repeats

    def likelihood_numerical_gradient(self, batch_indices, params):
        scale = 0.01
        gradient = np.zeros(self.num_params)
        for i in range(self.num_params):
            new_params = np.copy(params)
            new_params[i] -= scale
            likelihood_negative = self.problem.likelihood(batch_indices,
                                                          new_params)
            new_params[i] += 2*scale
            likelihood_positive = self.problem.likelihood(batch_indices,
                                                          new_params)
            gradient[i] = (likelihood_positive-likelihood_negative) / (2*scale)
        return np.array(gradient)
