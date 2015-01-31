import numpy as np


class SGLDOptimizer(object):
    class StepSizeGenerator(object):
        def __init__(self, a, b, gamma):
            self.a = a
            self.b = b
            self.gamma = gamma

        def get_epsilon(self, iteration):
            return self.a * (self.b+iteration)**-self.gamma

    def __init__(self, problem):
        self.problem = problem
        self.data_size = problem.data_size
        self.num_params = problem.num_params
        self.prior_gradient = problem.prior_gradient
        self.likelihood_gradient = problem.likelihood_gradient

    def optimize(self, num_epochs, batch_size, step_size_gen):
        num_iterations = np.ceil(self.data_size/float(batch_size))*num_epochs
        validation_scores = np.zeros(num_iterations)
        epsilons = np.zeros(num_iterations)
        variances = np.zeros((num_iterations, self.num_params))
        params_draws = np.zeros((num_iterations, self.num_params))
        params = self.problem.initial_params
        iteration = 0

        for epoch in range(num_epochs):
            indices = np.random.permutation(self.data_size)
            if num_epochs < 10 or (epoch+1) % (num_epochs/10) == 0:
                print "Epoch %d" % (epoch+1)
            for index in range(0, self.data_size, batch_size):
                batch_indices = indices[index:index+batch_size]
                scale = float(self.data_size) / len(batch_indices)

                prior_grad = self.prior_gradient(params)
                likelihood_grad = self.likelihood_gradient(batch_indices,
                                                           params)

                epsilon = step_size_gen.get_epsilon(iteration)
                eta = np.random.normal(0, np.sqrt(epsilon), self.num_params)

                params += epsilon * 0.5 * (prior_grad + scale*likelihood_grad) \
                    + eta

                if "validation_score" in dir(self.problem):
                    validation_scores[iteration] = self.problem.validation_score(params)
                # Record info to plot later
                epsilons[iteration] = epsilon
                params_draws[iteration] = np.copy(params)
                if batch_size == 1:
                    variances[iteration] = np.var(params_draws[max(0, iteration-10):iteration], axis=0)
                else:
                    variances[iteration] = self.calculate_variance(batch_indices,
                                                                   params)

                iteration += 1

        return params_draws, variances, epsilons, validation_scores

    # Implementation of equation 9
    def calculate_variance(self, batch_indices, params):
        batch_size = len(batch_indices)
        s = np.zeros((batch_size, self.num_params))
        prior_grad = self.prior_gradient(params)
        for i in range(batch_size):
            likelihood_grad = self.likelihood_gradient([batch_indices[i]],
                                                       params)
            s[i] = likelihood_grad + 1.0/self.data_size * prior_grad
        s_mean = np.mean(s, axis=0)
        scale = (self.data_size / float(batch_size))**2
        cov = scale * np.sum([np.outer(s[i] - s_mean, s[i] - s_mean)
                              for i in range(batch_size)], axis=0)
        return cov.diagonal()
