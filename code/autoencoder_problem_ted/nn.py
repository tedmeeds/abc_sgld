import numpy as np
import json
import gzip
import cPickle
from time import time
import matplotlib.pyplot as plt
import pdb

def load_mnist():
    f = gzip.open('../../data/mnist.pkl.gz', 'rb')
    data = cPickle.load(f)
    f.close()
    return data

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_grad(x):
    ex = np.exp(-x)
    return ex / (1 + ex)**2

def one_hot(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_sets_mnist():
  
    (x_train, t_train), (x_valid, t_valid), (x_test, t_test) = load_mnist()
    # Transform the data into something easier to work with (imo)
    training_inputs = [np.reshape(x, (784, 1)) for x in x_train]
    training_results = [one_hot(y) for y in t_train]
    training_set = zip(training_inputs, training_inputs)

    validation_inputs = [np.reshape(x, (784, 1)) for x in x_valid]
    valiation_results = [one_hot(y) for y in t_valid]
    validation_set = zip(validation_inputs, validation_inputs)

    test_inputs = [np.reshape(x, (784, 1)) for x in x_test]
    test_set = zip(test_inputs, test_inputs)

    return training_set, validation_set, test_set
    # v--- really slow
    # return np.array(training_set), np.array(validation_set), np.array(test_set)

def plot_digits(data, numcols, shape=(28,28)):
    numdigits = data.shape[0]
    numrows = int(numdigits/numcols)
    for i in range(numdigits):
        plt.subplot(numrows, numcols, i)
        plt.axis('off')
        plt.imshow(data[i].reshape(shape), interpolation='nearest', cmap='Greys')
    plt.show()

class NeuralNetwork(object):
    def __init__(self, config):
        # config is a list with the number of neurons per layer, so len(config) equals the number of layers.
        self.config = config
        self.num_layers = len(config)
        # Initialize the biases
        self.biases = np.array([0.001*np.random.randn(layer_size, 1) for layer_size in self.config[1:]])
        # Initialize the weights
        self.weights = []
        for i in range(self.num_layers-1):
            layer_size = self.config[i]
            next_layer_size = self.config[i+1]
            weights = 0.001*np.random.randn(next_layer_size, layer_size) / np.sqrt(layer_size)
            self.weights.append(weights)
        self.weights = np.array(self.weights)

    def feedforward(self, x):
        self.pre_activations = []
        self.activations = [x]
        activation = x
        for i in range(self.num_layers-1):
            pre_activation = np.dot(self.weights[i], activation) + self.biases[i]
            self.pre_activations.append(pre_activation)
            activation = sigmoid(pre_activation)
            self.activations.append(activation)
        return activation

    def train(self, training_data, epochs, mini_batch_size, learning_rate, regularization_rate, validation_set):
        start_time = time()
        num_validation_samples = len(validation_set)
        num_training_samples = len(training_data)
        for epoch in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, num_training_samples, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate, regularization_rate, num_training_samples)
            print "Epoch %s done." % epoch
            reconstruction_error, num_correct = self.test(validation_set)
            print "Reconstruction error: {}, average percentage of pixels correct: {}%".format(reconstruction_error, num_correct*100)
            self.save('{}-epoch{}.json'.format(start_time, epoch))

    def update_mini_batch(self, mini_batch, learning_rate, regularization_rate, num_training_samples):
        grad_b = 0*self.biases
        grad_w = 0*self.weights
        for x, y in mini_batch:
            self.feedforward(x) # Forward pass
            delta_grad_b, delta_grad_w = self.backpropagation(x, y) # Backwards pass
            # Update the gradients
            grad_b += delta_grad_b
            grad_w += delta_grad_w

        self.biases -= learning_rate/len(mini_batch)*grad_b # update biases
        self.weights *= (1 - learning_rate*(regularization_rate/num_training_samples)) # weight decay
        self.weights -= (learning_rate/len(mini_batch))*grad_w # update weights


    def backpropagation(self, x, y):
        grad_b = 0*self.biases
        grad_w = 0*self.weights
        delta = (self.activations[-1] - y) * sigmoid_grad(self.pre_activations[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta, self.activations[-2].T)
        for layer in range(2, self.num_layers):
            pre_activation = self.pre_activations[-layer]
            delta = np.dot(self.weights[-layer+1].T, delta) * sigmoid_grad(pre_activation)
            grad_b[-layer] = delta
            grad_w[-layer] = np.dot(delta, self.activations[-layer-1].T)
        return (grad_b, grad_w)

    def test(self, x,y):
        errors = []
        correct = []
        #for x, y in data:
        reconstruction = self.feedforward(x)
        error = np.nan_to_num(np.sum(y*np.log(reconstruction+10**-10) + (1-y)*np.log(1-reconstruction+10**-10)))
        error = error/(x.shape[0]*x.shape[1])
        errors.append(error)
        reconstruction[reconstruction <= 0.5] = 0
        reconstruction[reconstruction > 0.5] = 1
        y_copy = y.copy()
        y_copy[y_copy <= 0.5] = 0
        y_copy[y_copy > 0.5] = 1
        
        abs_correct = np.sum(reconstruction == y_copy) / float(x.shape[0]*x.shape[1])
        correct.append(abs_correct)
        return np.mean(errors), np.mean(correct)

    def test_old(self, data):
        return sum(np.argmax(self.feedforward(x)) ==  y for (x, y) in data)

    def save(self, filename):
        data = {
            "config": self.config,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases]
        }
        file = open(filename, "w")
        json.dump(data, file)
        file.close()

    def load(self, filename):
        file = open(filename, "r")
        data = json.load(file)
        file.close()
        self.config = data["config"]
        self.weights = np.array([np.array(w) for w in data["weights"]])
        self.biases = np.array([np.array(b) for b in data["biases"]])

if __name__ == '__main__':
    training_set, validation_set, test_set = load_sets_mnist()
    nn = NeuralNetwork([28*28, 100, 28*28])
    nn.load('latest_epoch.json')
    # plot_digits(np.expand_dims(nn.weights[1][:,0], axis=0), 1)
    nn.train(training_set, 100, 16, 0.1, 1, validation_set)

