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
    pca = PCA(n_components=K, whiten=True)
    pca = pca.fit(W)
    return pca

def plot_PCA(filename, random_proj=None):
    file = open('LR2.json', "r")
    LR_data = json.load(file)
    file.close()
    MAP = np.array(LR_data['weights'][-1]).flatten()

    file = open(filename, "r")
    data = json.load(file)
    file.close()
    W = np.array([np.array(w).flatten() for w in data["weights"]])

    if random_proj is not None:
        W_proj = np.dot(W, random_proj)
        MAP_proj = np.dot(MAP, random_proj)
        W_proj -= MAP_proj
        # So I don't have to alter the code below..
        W_pca = W_proj
        MAP_pca = W_proj
    else:
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
    # plt.figure()
    # plt.contour(X, Y, Z)
    plt.plot(W_pca[:, 0], W_pca[:, 1], 'k-')
    plt.plot(W_pca[:, 0], W_pca[:, 1], 'o', markersize=5)
    plt.plot(MAP_pca[0][0], MAP_pca[0][1], 'wo', markersize=8)
    # plt.xlim((-324, -326))
    # plt.ylim((-345, -350))
    # plt.show()


if __name__ == '__main__':
    file = open('LR2.json', "r")
    LR_data = json.load(file)
    file.close()
    MAP = np.array(LR_data['weights'][-1]).flatten()
    random_proj = np.random.randn( len(MAP),2 )
    plt.figure()
    # filename = 'sampling-sgld-eta=0.01-C=100.json'
    filename = 'sampling-sgld-truegradient-eta=0.01-C=100.json'
    plot_PCA(filename, random_proj)
    # filename = 'LR2.json'
    # filename = 'sampling-mcmc-q=0.5.json'
    # plot_PCA(filename)
    # filename = 'sampling-mcmc.json'
    # filename = 'sampling-thermo2.json'
    # filename = 'sampling-thermo-eta=1e-2-C=20.json'
    filename = 'sampling-thermo-truegradient-eta=0.01-C=20.json'
    # filename = 'sampling-true-posterior-mcmc.json'
    plot_PCA(filename, random_proj)
    plt.show()
