import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.decomposition import PCA
from sklearn import mixture

def plot_LL(filename):
    file = open(filename, "r")
    data = json.load(file)
    file.close()

    plt.figure()
    plt.plot(data['LLs'])
    plt.show()

def plot_PCA(filename):
    file = open(filename, "r")
    data = json.load(file)
    file.close()
    # W = np.array([np.array(w) for w in data["weights"][-1]])
    W = np.array([np.array(w).flatten() for w in data["weights"]]).T
    K = 2
    pca = PCA(n_components=K)
    W_pca = pca.fit(W).transform(W)
    clf = mixture.GMM(n_components=K, covariance_type='full')
    clf.fit(W_pca)
    x = np.linspace(-100, 100)
    X, Y = np.meshgrid(x, x)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)[0]
    Z = Z.reshape(X.shape)
    plt.figure()
    plt.contour(X, Y, Z)
    plt.scatter(W_pca[:, 0], W_pca[:, 1])
    plt.show()


if __name__ == '__main__':
    # filename = 'sampling-sgld2.json'
    # filename = 'LR2.json'
    filename = 'sampling-MCMC.json'
    plot_PCA(filename)
