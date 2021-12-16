from sklearn.cluster import SpectralClustering, KMeans
import numpy as np
import json
from scipy.sparse.linalg import eigs
import time
from statistics import median
from fastKmeans import fast2means_v1, compare, fast2means_v2, randIndex
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

what = 'bin'
mclc = 1
n_samples = 1000
minl = 500
#l = int(np.log2(n_samples/minl))
l = 2
threshold = -0.05

print("method: ", mclc, ", type: ", what, ", n samples: ", n_samples, ", l: ", l, ", threshold: ", threshold)

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

def constructD(mat):
    n, m = len(mat), len(mat[0])
    D = np.zeros((n,m))
    for i in range(n):
        D[i,i] = sum(mat[i,:])
    return D

def constructD_abs(mat):
    n, m = len(mat), len(mat[0])
    D = np.zeros((n,m))
    for i in range(n):
        D[i][i] = sum(abs(mat[i,:]))
    return D

def constructL(D, A):
    D2inv = np.zeros(D.shape)
    for i in range(D.shape[0]):
        D2inv[i,i] = 1/np.sqrt(D[i,i])
    L = np.eye(D.shape[0]) - np.matmul(D2inv, np.matmul(A, D2inv))
    return L

n_clusters = 2
filename = str(n_samples) + "/sim" + str(n_samples) + ".json"
f = open(filename)
simMat = np.array(json.load(f))

if what == 'pos':
    posMat(simMat)
    D = constructD(simMat)
elif what == 'bin':
    binMat(simMat)
    D = constructD(simMat)
elif what == 'abs':
    D = constructD_abs(simMat)
elif what == 'org':
    D = constructD(simMat)

L = constructL(D, simMat)
Dinv = np.zeros(D.shape)
for i in range(D.shape[0]):
    Dinv[i,i] = 1/D[i,i]
P = np.matmul(Dinv, simMat)

#print(np.linalg.norm(L-L.T), np.linalg.norm(P-P.T))
if mclc == 1:
    #L = L + 0.1*np.eye(L.shape[0])
    #print(L[0:5,0:5])
    time2 = 0
    start = time.time()
    for i in range(5):
        w, v = eigs(L, k=2, which='SR')
        #w = w[1]
        vec = np.real(v[:,1])
        vec = vec/np.linalg.norm(vec)
        start2 = time.time()
        cluster1, cluster2 = fast2means_v2(vec, l)
        time2 += time.time() - start2
        #idx = np.sign(vec-median(vec))
        #idx = [num if num == 1 else 0 for num in idx]
    time1 = (time.time() - start)/5
    time2 = time2/5
    print(w)
    w = w[1]
elif mclc == 2:
    time2 = 0
    start = time.time()
    for i in range(5):
        w, v = eigs(P, k=1, which='LM')
        #w = w[0]
        vec = np.real(v[:,0])
        vec = vec/np.linalg.norm(vec)
        start2 = time.time()
        cluster1, cluster2 = fast2means_v2(vec, l)
        time2 += time.time() - start2
        #idx = np.sign(vec-median(vec))
        #idx = [num if num == 1 else 0 for num in idx]
    time1 = (time.time() - start)/5
    time2 = time2/5
    #print(w)
    w = w[0]

#print(vec[:200])
w = np.real(w)
vec = np.squeeze(vec)
#print(w, min(vec), max(vec))
labels = {'cluster1': [int(i) for i in cluster1], 'cluster2': [int(i) for i in cluster2], 'time': np.round(time1, 2), 'time2': time2, 'vec': [val for val in vec], 'eig': w}

savefile = "labels_my_v2_" + str(n_samples) + "_" + str(n_clusters) + "_2_" + what + "_" + str(mclc) + "_" + str(l) + ".json"
f = open(savefile, 'w')
json.dump(labels, f)

cluster11, cluster12 = cluster1, cluster2
time11, time12 = time1, time2

#print('what ', len(cluster11)+len(cluster12))



if mclc == 1:
    #L = L + 0.1*np.eye(L.shape[0])
    #print(L[0:5,0:5])
    time2 = 0
    start = time.time()
    for i in range(5):
        w, v = eigs(L, k=2, which='SR')
        #w = w[1]
        vec = np.real(v[:,1])
        vec = vec/np.linalg.norm(vec)
        start2 = time.time()
        kmeans = KMeans(n_clusters=2, random_state=0).fit(vec.reshape(-1,1))
        labels = kmeans.labels_
        #cluster1, cluster2 = [], []
        #for i, label in enumerate(labels):
        #    if label == 0:
        #        cluster1.append(i)
        #    else:
        #        cluster2.append(i)
        time2 += time.time() - start2
        #idx = np.sign(vec-median(vec))
        #idx = [num if num == 1 else 0 for num in idx]
    time1 = (time.time() - start)/5
    time2 = time2/5
    print(w)
    w = w[1]
elif mclc == 2:
    time2 = 0
    start = time.time()
    for i in range(5):
        w, v = eigs(P, k=1, which='LM')
        #w = w[0]
        vec = np.real(v[:,0])
        vec = vec/np.linalg.norm(vec)
        start2 = time.time()
        kmeans = KMeans(n_clusters=2, random_state=0).fit(vec.reshape(-1,1))
        labels = kmeans.labels_
        #cluster1, cluster2 = [], []
        #for i, label in enumerate(labels):
        #    if label == 0:
        #        cluster1.append(i)
        #    else:
        #        cluster2.append(i)
        time2 += time.time() - start2
        #idx = np.sign(vec-median(vec))
        #idx = [num if num == 1 else 0 for num in idx]
    time1 = (time.time() - start)/5
    time2 = time2/5
    #print(w)
    w = w[0]

#print(vec[:200])

cluster1, cluster2 = [], []
for i, label in enumerate(labels):
    if label == 0:
        cluster1.append(i)
    else:
        cluster2.append(i)

w = np.real(w)
vec = np.squeeze(vec)
#print(w, min(vec), max(vec))

rate1 = randIndex(vec, cluster11, cluster12, cluster1, cluster2)
rate2 = randIndex(vec, cluster11, cluster12, cluster2, cluster1)

if rate1 < rate2:
    rate1, rate2 = rate2, rate1
    cluster1, cluster2 = cluster2, cluster1

print("mismatch rate = ", rate1)
print("my time1 = ", time11, ", my time2 = ", time12)
print("km time1 = ", time1, ", km time2 = ", time2)
print("my cl0 rate = ", len(cluster11)/n_samples, ", my cl1 rate = ", len(cluster12)/n_samples)
print("km cl0 rate = ", len(cluster11)/n_samples, ", km cl1 rate = ", len(cluster12)/n_samples)

labels = {'cluster1': [int(i) for i in cluster1], 'cluster2': [int(i) for i in cluster2], 'time': np.round(time1, 2), 'time2': time2, 'vec': [val for val in vec], 'eig': w, 'rate': rate1}

savefile = "labels_km_v2_" + str(n_samples) + "_" + str(n_clusters) + "_2_" + what + "_" + str(mclc) + "_" + str(l)  + ".json"
f = open(savefile, 'w')
json.dump(labels, f)

cluster31, cluster32 = [], []
for i, num in enumerate(vec):
    if num > threshold:
        cluster31.append(i)
    else:
        cluster32.append(i)

rate1 = randIndex(vec, cluster31, cluster32, cluster1, cluster2)
rate2 = randIndex(vec, cluster31, cluster32, cluster2, cluster1)
if rate1 < rate2:
    rate1, rate2 = rate2, rate1
    cluster31, cluster32 = cluster32, cluster31
print("mismatch rate (tv) = ", rate1)
print("tv cl0 rate = ", len(cluster31)/n_samples, ", tv cl1 rate = ", len(cluster32)/n_samples)

if what != 'abs' and mclc == 1:
    start = time.time()
    clustering = SpectralClustering(n_clusters=2, n_components=2, affinity='precomputed', assign_labels='kmeans')
    for i in range(5):
        #clustering = SpectralClustering(n_clusters=2, n_components=2, affinity='precomputed', assign_labels='kmeans')
        clustering.fit(simMat)
    time1 = (time.time() - start)/5

    labels = clustering.labels_
    cluster1, cluster2 = [], []
    for i, label in enumerate(labels):
        if label == 0:
            cluster1.append(i)
        else:
            cluster2.append(i)
    rate1 = randIndex(vec, cluster11, cluster12, cluster1, cluster2)
    rate2 = randIndex(vec, cluster11, cluster12, cluster2, cluster1)

    if rate1 < rate2:
        rate1, rate2 = rate2, rate1
        cluster1, cluster2 = cluster2, cluster1
    
    print("mismatch rate (with bt) = ", rate1)
    print("bt time1 = ", time1, ", bt time2 = unknown", )
    print("bt cl0 rate = ", len(cluster1)/n_samples, ", bt cl1 rate = ", len(cluster2)/n_samples)

    rate1 = randIndex(vec, cluster31, cluster32, cluster1, cluster2)
    rate2 = randIndex(vec, cluster31, cluster32, cluster2, cluster1)
    if rate1 < rate2:
        rate1, rate2 = rate2, rate1
        cluster31, cluster32 = cluster32, cluster31
    print("mismatch rate (tv) = ", rate1)
    print("tv cl0 rate = ", len(cluster31)/n_samples, ", tv cl1 rate = ", len(cluster32)/n_samples)

