from sklearn.cluster import SpectralClustering
import numpy as np
import json

def posMat(mat):
    n, m = len(mat), len(mat[0])
    for i in range(n):
        for j in range(m):
            mat[i][j] = mat[i][j] if mat[i][j] > 0 else 0

def binMat(mat):
    n, m = len(mat), len(mat[0])
    for i in range(n):
        for j in range(m):
            mat[i][j] = 1 if mat[i][j] > 0 else 0

def constructD_abs(mat):
    n, m = len(mat), len(mat[0])
    D = np.zeros((n,m))
    for i in range(n):
        D[i][i] = sum(abs(mat[i,:]))
    return D

n_clusters = 2
n_components = 2
what = 'bin'

n_samples = 10000
filename = str(n_samples) + "/sim" + str(n_samples) + ".json"
f = open(filename, 'r')
simMat = np.array(json.load(f))
print(simMat)
if what == 'pos':
    posMat(simMat)
elif what == 'bin':
    binMat(simMat)

clustering = SpectralClustering(n_clusters=n_clusters, n_components=n_components,
            affinity='precomputed', assign_labels='kmeans')
clustering.fit(simMat)

labels = {'labels': [int(label) for label in clustering.labels_]}

savefile = "labels_" + str(n_samples) + "_" + str(n_clusters) + "_" + str(n_components) + "_"  + what
f = open(savefile+".json", 'w')
json.dump(labels, f)
print(labels)
