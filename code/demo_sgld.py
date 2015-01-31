import numpy as np
import matplotlib.pyplot as plt
from sgld_problems import *
from sgld import SGLDOptimizer
from sgld_spall import SGLDSpallOptimizer


if __name__ == "__main__":
    np.random.seed(2)
    num_epochs = 1000
    data_size = 100
    sgld = SGLDOptimizer(ToyProblem(data_size))
    # sgld = SGLDSpallOptimizer(ToyProblem(data_size), 2, 0.1)
    step_size_gen = SGLDOptimizer.StepSizeGenerator(0.2, 230, 0.55)
    params_draws, variances, epsilons, validation_scores = sgld.optimize(int(num_epochs*1.1), 1, step_size_gen)

    # Reproduce figure 1
    # We could limit the number of points plotted if num_epochs is large
    # params_draws = params_draws[-(num_epochs*data_size/10):]
    x = params_draws[:, 0]
    y = params_draws[:, 1]
    plt.plot(x, y, 'o', markersize=0.9, alpha=0.1)
    plt.xlim((-1, 2))
    plt.ylim((-3, 3))
    plt.show()

    # Reproduce figure 2
    plt.figure()
    theta1 = variances[:, 0]
    theta2 = variances[:, 1]
    plt.plot(theta1)
    plt.plot(theta2)
    plt.plot(epsilons)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
