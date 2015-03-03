import numpy as np
import pylab as pp
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
    # print W.shape
    # m = W.mean(0)
    # W -= m
    # s = W.std(0)
    # ok = pp.find(s>0)
    # W[:, ok] /= s[ok]
    pca = PCA(n_components=K, whiten=True)
    pca = pca.fit(W)
    return pca

def plot_PCA(filename):
    file = open('LR2.json', "r")
    LR_data = json.load(file)
    file.close()
    MAP = np.array(LR_data['weights'][-1]).flatten()
    # MAP -= MAP.mean()
    # s = MAP.std()
    # MAP /= s
    file = open(filename, "r")
    data = json.load(file)
    file.close()
    # W = np.array([np.array(w) for w in data["weights"][-1]])
    W = np.array([np.array(w).flatten() for w in data["weights"]])
    # W -= W.mean(0)
    # s = W.std(0)
    # ok = pp.find(s>0)
    # W[:,ok] /= s[ok]
    pca = create_PCA('sampling-true-posterior-mcmc.json')
    W_pca = pca.transform(W)
    MAP_pca = pca.transform(MAP)
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
    plt.plot(W_pca[:, 0], W_pca[:, 1], 'ro', markersize=5)
    plt.plot(MAP_pca[0][0], MAP_pca[0][1], 'bo', markersize=8)
    # plt.xlim((-324, -326))
    # plt.ylim((-345, -350))
    plt.show()


if __name__ == '__main__':
    filename = 'sampling-sgld-4.json'
    # filename = 'LR2.json'
    # filename = 'sampling-mcmc-q=0.5.json'
    # filename = 'sampling-mcmc.json'
    # filename = 'sampling-thermo2.json'
    # filename = 'sampling-thermo-eta=1e2-C=100.json'
    # filename = 'sampling-true-posterior-mcmc.json'
    plot_PCA(filename)
