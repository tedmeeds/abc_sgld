import numpy as np
from helpers import sigmoid


class ToyProblem(object):
    def __init__(self, data_size):
        self.data_size = data_size
        self.data_sigma = np.sqrt(2.0)
        self.theta_mean = np.array([0.0, 0.0])
        self.theta_sigma = np.sqrt([10.0, 1.0])
        self.data = self.generate_data(np.array([0.0, 1.0]))
        self.num_params = 2
        self.initial_params = np.zeros(2)

    def generate_data(self, theta):
        data = []
        for i in range(self.data_size):
            if np.random.random() < 0.5:
                point = np.random.normal(theta[0], self.data_sigma)
            else:
                point = np.random.normal(np.sum(theta), self.data_sigma)
            data.append(point)
        return np.array(data)

    def prior(self, theta):
        return np.log(np.prod(1.0 / (self.theta_sigma * np.sqrt(2*np.pi)) *
                              np.exp(-0.5 * ((theta-self.theta_mean) /
                                             self.theta_sigma)**2)))

    def likelihood(self, data_indices, theta):
        data = self.data[data_indices]
        return self.data_size * \
            np.log(0.5*(2*np.pi*self.data_sigma**2)**-0.5) + \
            np.sum(np.log(np.exp(-0.5 * ((data-theta[0])/self.data_sigma)**2) +
                          np.exp(-0.5 * ((data-np.sum(theta)) /
                                 self.data_sigma)**2)))

    def prior_gradient(self, theta):
        return (-(theta-self.theta_mean) / self.theta_sigma**2)

    def likelihood_gradient(self, data_indices, theta):
        data = self.data[data_indices]
        diff = [data-theta[0], data-theta[0]-theta[1]]
        a = np.exp(-0.5 * (diff[0] / self.data_sigma)**2)
        b = np.exp(-0.5 * (diff[1] / self.data_sigma)**2)
        gradient_theta1 = np.sum(
            ((diff[0] * a + diff[1] * b) / self.data_sigma**2) / (a+b)
        )
        gradient_theta2 = np.sum((diff[1] / self.data_sigma**2 * b) / (a+b))
        return np.array([gradient_theta1, gradient_theta2])


class A9AProblem(object):
    def __init__(self):
        self.scale = 1
        self.location = 0
        self.num_params = 123 + 1  # 14 features + 1 for bias
        self.initial_params = np.ones(self.num_params)
        self.initial_params[0] = 1.0
        self.load_data()

    def load_data(self):
        total_size = 32561  # Precalculated
        self.data_size = 26049  # Precalculated, rest is validation
        self.valid_size = total_size - self.data_size
        labels = np.zeros(total_size)
        data = np.zeros((total_size, self.num_params))
        with open('../data/a9a.txt', 'r') as dataset:
            for i, row in enumerate(dataset):
                row_split = row.split()
                labels[i] = int(row_split[0])
                features_present = [int(f.split(':')[0]) for f in row_split[1:]]
                features = np.zeros(self.num_params)
                features[[0]+features_present] = 1.0
                data[i] = features
        self.data = data[:self.data_size]
        self.data_labels = labels[:self.data_size]
        self.valid_data = data[self.data_size:]
        self.valid_labels = labels[self.data_size:]

    def prior(self, beta):
        return np.log(1.0/(2*self.scale)) - (np.abs(beta-self.location) /
                                             self.scale)

    def likelihood(self, data_indices, beta):
        return np.sum([sigmoid(self.data_labels[i]*beta.dot(self.data[i]))
                       for i in data_indices])

    def prior_gradient(self, beta):
        return -np.sign(beta)

    def likelihood_gradient(self, data_indices, beta):
        return np.sum([sigmoid(-self.data_labels[i] * beta.dot(self.data[i]))
                       * self.data_labels[i] * self.data[i]
                       for i in data_indices],
                      axis=0)

    def predict(self, index, beta):
        prob = sigmoid(beta.dot(self.valid_data[index]))
        return 1 if prob >= 0.5 else -1

    def validation_score(self, beta):
        num_correct = 0.0
        for i in range(self.valid_size):
            num_correct += self.predict(i, beta) == self.valid_labels[i]
        return num_correct/self.valid_size
