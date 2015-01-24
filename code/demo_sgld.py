import numpy as np
import matplotlib.pyplot as plt


def generate_data(theta, data_size, data_sigma):
    return 0.5 * (np.random.normal(theta[0], data_sigma**2, data_size) +
                  np.random.normal(np.sum(theta), data_sigma**2, data_size))


def compute_prior_gradient(theta, theta_mean, theta_sigma):
    return (-(theta-theta_mean) / theta_sigma**2)


def compute_likelihood_gradient(data, data_sigma, theta):
    diff = [data-theta[0], data-theta[0]-theta[1]]
    a = np.exp(-0.5 * (diff[0] / data_sigma)**2)
    b = np.exp(-0.5 * (diff[1] / data_sigma)**2)
    gradient_theta1 = np.sum(
        ((diff[0] * a + diff[1] * b) / data_sigma**2) / (a+b)
    )
    gradient_theta2 = np.sum((diff[1] / data_sigma**2 * b) / (a+b))
    return np.array([gradient_theta1, gradient_theta2])


# Implementation of equation 9
def calculate_variance(minibatch, data_sigma, data_size, theta):
    minibatch_size = len(minibatch)
    s = [0]*minibatch_size
    s_mean = np.copy(theta)*0.0
    for i in range(minibatch_size):
        x = minibatch[i]
        prior_gradient = compute_prior_gradient(theta, theta_mean, theta_sigma)
        likelihood_gradient = compute_likelihood_gradient(x, data_sigma, theta)
        s[i] = likelihood_gradient + 1.0/data_size * prior_gradient
        s_mean += s[i]
    s_mean /= minibatch_size
    scale = data_size / float(minibatch_size)**2
    return scale * np.sum([np.outer(s[i] - s_mean, s[i] - s_mean)
                           for i in range(minibatch_size)])

if __name__ == "__main__":
    # Parameters
    theta_true = np.array([0.0, 1.0])
    theta_mean = np.array([0.0, 0.0])
    theta_sigma = np.sqrt([10.0, 1.0])
    data_sigma = np.sqrt(2.0)
    data_size = 100
    num_epochs = 1000
    minibatch_size = 1
    a = 0.2
    b = 230
    gamma = 0.55
    iteration = 1
    theta_draws = []
    theta = np.array([0.0, 0.0])
    # if data_size%minibatch_size%!=0 you should put this in the loop
    # and replace minibatch_size by len(minibatch)
    scale = [data_size / float(minibatch_size)]*2
    epsilons = []
    etas = []
    variances = []

    data = generate_data(theta_true, data_size, data_sigma)
    for epoch in range(num_epochs):
        np.random.shuffle(data)
        if (epoch+1) % 100 == 0:
            print "Epoch %d" % (epoch+1)
            print theta
        for index in range(0, data_size - minibatch_size + 1, minibatch_size):
            minibatch = data[index:index+minibatch_size]
            prior_gradient = compute_prior_gradient(theta, theta_mean,
                                                    theta_sigma)
            likelihood_gradient = compute_likelihood_gradient(minibatch,
                                                              data_sigma,
                                                              theta)
            epsilon = a * (b+iteration)**-gamma
            eta = np.random.normal(0, np.sqrt(epsilon), 2)
            theta += epsilon * 0.5 * (prior_gradient +
                                      scale*likelihood_gradient) + eta
            # Record info to plot later
            epsilons.append(epsilon)
            etas.append(np.mean(eta))
            V = calculate_variance(minibatch, data_sigma, data_size, theta)
            variances.append(V)
            if iteration > 10000:
                theta_draws.append(np.copy(theta))
            iteration += 1

    # Reproduce figure 1
    theta_draws = np.array(theta_draws)
    x = theta_draws[:, 0]
    y = theta_draws[:, 1]
    plt.plot(x, y, 'o', markersize=0.9)
    plt.xlim((-1, 2))
    plt.ylim((-3, 3))
    plt.show()
    # Reproduce figure 2
    plt.figure()
    plt.plot(variances)
    plt.plot(epsilons)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
