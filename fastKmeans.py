from statistics import mean, median_low
from collections import deque
from time import time
import numpy as np

class QSEL:
    def median(self, nums, indices, k):
        ans = self.quickselect(nums, indices, 0, len(indices)-1, k-1)
        return indices[k]
        
    def quickselect(self, nums, indices, lo, hi, k):
        if lo == hi:
           return indices[lo]
        pivot = self.partition(nums, indices, lo, hi)
        if k == pivot:
           return indices[k]
        elif k < pivot:
           return self.quickselect(nums, indices, lo, pivot-1, k)
        else:
           return self.quickselect(nums, indices, pivot+1, hi, k)
           
    def partition(self, nums, indices, lo, hi):
        pivot = nums[indices[hi]]
        i = lo-1
        for j in range(lo, hi+1):
            if nums[indices[j]] <= pivot:
                i += 1
                tmp = indices[i]
                indices[i] = indices[j]
                indices[j] = tmp
        return i

def fast2means_v1(vec):
    #start = time()
    n = len(vec)
    idx = sorted(range(n), key=lambda x:vec[x])
    mid = n // 2
    vec_sorted = vec[idx]
    center1 = mean(vec_sorted[:mid])
    center2 = mean(vec_sorted[mid:])
    n1, n2 = mid, n-mid
    cluster1 = deque(idx[:mid])
    cluster2 = deque(idx[mid:])
    #print(mid, n, len(cluster1), len(cluster2))
    dist1 = abs(center1-vec_sorted[mid])
    dist2 = abs(center2-vec_sorted[mid])
    if dist1 < dist2:
        while dist1 < dist2:
            cluster1.append(idx[mid])
            cluster2.popleft()
            center1 = (center1*n1 + vec_sorted[mid]) / (n1+1)
            center2 = (center2*n2 - vec_sorted[mid]) / (n2-1)
            n1, n2 = n1+1, n2-1
            mid += 1
            dist1 = abs(center1-vec_sorted[mid])
            dist2 = abs(center2-vec_sorted[mid])
    elif dist1 > dist2:
        mid -= 1
        dist1 = abs(center1-vec_sorted[mid])
        dist2 = abs(center2-vec_sorted[mid])
        while dist1 > dist2:
            cluster1.pop()
            cluster2.appendleft(idx[mid])
            center1 = (center1*n1 - vec_sorted[mid]) / (n1-1)
            center2 = (center2*n2 + vec_sorted[mid]) / (n2+1)
            n1, n2 = n1-1, n2+1
            mid -= 1
            dist1 = abs(center1-vec_sorted[mid])
            dist2 = abs(center2-vec_sorted[mid])
    #t = time() - start
    return cluster1, cluster2

def applyQsel(nums, indices, l):
    n = len(indices)
    start, end = 0, n-1
    mid = n // 2
    qsel = QSEL()
    qsel.quickselect(nums, indices, start, end, mid)
    i = 1
    while i < l:
        qsel.quickselect(nums, indices, start, mid-1, (start + mid - 1) // 2)
        qsel.quickselect(nums, indices, mid, end, (mid + end) // 2)
        start = (start + mid - 1) // 2
        end = (mid + end) // 2
        i += 1
    return start, end

def fast2means_v2(vec, l):
    indices = list(range(len(vec)))
    start, end = applyQsel(vec, indices, l)
    subidx = indices[start:end+1]
    subcl1, subcl2 = fast2means_v1(vec[subidx])
    subidx = np.array(subidx, dtype=int)
    cluster1 = indices[:start] + list(subidx[subcl1])
    cluster2 = list(subidx[subcl2]) + indices[end+1:]
    #print(type(cluster1))
    return cluster1, cluster2


def compare(nums, cluster11, cluster12, cluster21, cluster22):
    #print(len(cluster11), len(cluster12), len(cluster11) + len(cluster12), len(cluster21), len(cluster22), len(cluster21) + len(cluster22))
    assert len(cluster11) + len(cluster12) == len(cluster21) + len(cluster22)
    nums11 = set(nums[cluster11])
    nums12 = set(nums[cluster12])
    nums21 = set(nums[cluster21])
    nums22 = set(nums[cluster22])
    mismatch = 0
    for ele in cluster11:
        if nums[ele] not in nums21:
            mismatch += 1
    for ele in cluster12:
        if nums[ele] not in nums22:
            mismatch += 1
    return mismatch/(len(cluster11) + len(cluster12))

def randIndex(nums, predP, predN, grndP, grndN):
    assert len(predP)+len(predN) == len(grndP) + len(grndN)
    N = len(predP) + len(predN)
    TP, TN = 0, 0
    FP, FN = 0, 0
    numsP = set(nums[grndP])
    numsN = set(nums[grndN])
    for ele in predP:
        num = nums[ele]
        if num in numsP:
            TP += 1
        else:
            FP += 1
    for ele in predN:
        num = nums[ele]
        if num in numsP:
            FN += 1
        else:
            TN += 1
    return (TP + TN)/N
