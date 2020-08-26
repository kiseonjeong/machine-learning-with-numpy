import numpy as np
import matplotlib.pyplot as plt
from KNN import KNN as knn

# Do testing on the KNN
k = 3
ratio = 0.10
db_path = 'KNN\\datingTestSet.txt'
mat, labels = knn.get_DB(db_path)
nmat, ranges, mins = knn.scale(mat)
m = nmat.shape[0]
t = int(m * ratio)
error = 0
for i in range(t):
    result = knn.classify(nmat[i, :], nmat[t:m, :], labels[t:m], k)
    print('the classifier came back with: %s, the real answer is: %s' % (result, labels[i]))
    if result != labels[i]:
        error += 1
print('the total error rate is: %f' % (error / float(t)))