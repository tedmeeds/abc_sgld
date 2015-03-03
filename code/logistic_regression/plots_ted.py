import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.decomposition import PCA
from sklearn import mixture

K = 2

def plot_LL(filename):
    file = open(filename, "r")
    data = json.load(file)
    file.close()

    plt.figure()
    plt.plot(data['LLs'])
    plt.show()


def create_PCA(filename):
    file = open(filename, "r")
    data = json.load(file)
    file.close()
    W = np.array([np.array(w).flatten() for w in data["weights"]])
    pca = PCA(n_components=K)
    pca = pca.fit(W)
    return pca

def plot_PCA(filename):
    file = open(filename, "r")
    data = json.load(file)
    file.close()
    # W = np.array([np.array(w) for w in data["weights"][-1]])
    W = np.array([np.array(w).flatten() for w in data["weights"]])
    pca = create_PCA('sampling-true-posterior-mcmc.json')
    W_pca = pca.transform(W)
    # clf = mixture.GMM(n_components=K, covariance_type='full')
    # clf.fit(W_pca)
    # x = np.linspace(-2, 20)
    # y = np.linspace(-2, 2)
    # X, Y = np.meshgrid(x, y)
    # XX = np.array([X.ravel(), Y.ravel()]).T
    # Z = -clf.score_samples(XX)[0]
    # Z = Z.reshape(X.shape)
    plt.figure()
    # plt.contour(X, Y, Z)
    plt.plot(W_pca[:, 0], W_pca[:, 1], 'k-')
    plt.plot(W_pca[:, 0], W_pca[:, 1], 'ko')
    # plt.xlim((-324, -326))
    # plt.ylim((-345, -350))
    plt.show()
    
    return W, pca, W_pca


if __name__ == '__main__':
    filename = 'sampling-sgld-4.json'
    # filename = 'LR2.json'
    #filename = 'sampling-mcmc-q=0.5.json'
    #filename = 'sampling-true-posterior-mcmc.json'
    # filename = 'sampling-mcmc.json'
    # filename = 'sampling-thermo.json'
    # filename = 'sampling-true-posterior-mcmc.json'
    W, pca, W_pca = plot_PCA(filename)
