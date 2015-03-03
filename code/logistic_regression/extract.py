import numpy as np
import matplotlib.pyplot as plt
import json

def extract():
    file = open('LR2.json', "r")
    LR_data = json.load(file)
    file.close()
    MAP = np.array(LR_data['weights'][-1]).flatten()
    random_proj = np.random.randn( len(MAP),2 )
    MAP_proj = np.dot(MAP, random_proj)
    extracted_data = {'MAP_proj': MAP_proj, 'MAP': MAP}
    shortcuts = ['true_sgld', 'true_thermo', 'spsa_sgld', 'spsa_thermo']
    filenames = ['sampling-sgld-truegradient-eta=0.01-C=100.json',
                 'sampling-thermo-truegradient-eta=0.01-C=20.json',
                 'sampling-sgld-eta=0.01-C=100.json',
                 'sampling-thermo-eta=0.01-C=20.json']
    for i in range(len(filenames)):
        file = open(filenames[i], "r")
        data = json.load(file)
        file.close()
        W = np.array([np.array(w).flatten() for w in data["weights"]])
        W_proj = np.dot(W, random_proj)
        W_proj -= MAP_proj
        extracted_data[shortcuts[i]] = W_proj

    return extracted_data

if __name__ == '__main__':
    data = extract()
    plt.figure()
    plt.plot(data['true_sgld'][:, 0], data['true_sgld'][:, 1], 'k-')
    plt.plot(data['true_sgld'][:, 0], data['true_sgld'][:, 1], 'o', markersize=5)
    plt.plot(data['MAP_proj'][0], data['MAP'][1], 'wo', markersize=8)
    plt.show()
