from sklearn.cluster import SpectralClustering, KMeans
import numpy as np
import json
import h5py
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs
import time
from statistics import median
from fastKmeans import fast2means_v1, compare, fast2means_v2, randIndex
import scipy.io
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

what = "abs"
mclc = 1
N_samples = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000]
minl = 500
#l = int(np.log2(n_samples/minl))
l = 2
threshold = -0.05

for n_samples in N_samples:

    print("")
    print("=================================================")
    print("type: ", what, " number of samples = ", n_samples)

    n_clusters = 2
    filename = "sparsedata/" + str(n_samples) + "/sparse" + str(n_samples) + what + ".mat"
    #data = scipy.io.loadmat(filename)
    #L = data['A']
    f = h5py.File(filename, 'r')
    L = f.get('A')
    L = csc_matrix((L['data'], L['ir'], L['jc']), shape=(n_samples, n_samples))

    #print(np.linalg.norm(L-L.T), np.linalg.norm(P-P.T))



    #w, v = eigs(L, k=2, which='SR')
    #vec = np.real(v[:,1])
    vec = np.random.rand(n_samples)*2 - 1
    vec = vec/np.linalg.norm(vec)

    start1 = time.time()
    for i in range(5):
        cluster11, cluster12 = fast2means_v1(vec)
    time1 = time.time() - start1
    time1 /= 5


    start2 = time.time()
    for i in range(5):
        cluster21, cluster22 = fast2means_v2(vec, l)
    time2 = time.time() - start2
    time2 /= 5


    start3 = time.time()
    for i in range(5):
        kmeans = KMeans(n_clusters=2, random_state=0).fit(vec.reshape(-1,1))
    time3 = time.time() - start3
    time3 /= 5


    labels = kmeans.labels_
    cluster31, cluster32 = [], []
    for i, label in enumerate(labels):
        if label == 0:
            cluster31.append(i)
        else:
            cluster32.append(i)


    rate11 = randIndex(vec, cluster11, cluster12, cluster31, cluster32)
    rate12 = randIndex(vec, cluster11, cluster12, cluster32, cluster31)

    if rate11 < rate12:
        rate11, rate12 = rate12, rate11
        cluster11, cluster12 = cluster12, cluster11

    rate21 = randIndex(vec, cluster21, cluster22, cluster31, cluster32)
    rate22 = randIndex(vec, cluster21, cluster22, cluster32, cluster31)

    if rate21 < rate22:
        rate21, rate22 = rate22, rate21
        cluster21, cluster22 = cluster22, cluster21

    print("km time = ", round(time3,3))
    print("RAND Index v1 = ", rate11, " v1 time = ", round(time1,3))
    print("RAND Index v2 = ", rate21, " v1 time = ", round(time2,3))
    print("v1 cl0 rate = ", len(cluster11)/n_samples, ", v2 cl1 rate = ", len(cluster12)/n_samples)
    print("v2 cl0 rate = ", len(cluster21)/n_samples, ", v2 cl1 rate = ", len(cluster22)/n_samples)
    print("km cl0 rate = ", len(cluster31)/n_samples, ", km cl1 rate = ", len(cluster32)/n_samples)

