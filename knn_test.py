import os
import numpy as np
import matplotlib.pyplot as plt
from knn import knn

def test_dating():
    # Do testing on the KNN
    k = 3
    ratio = 0.10
    db_path = 'KNN\\datingTestSet.txt'
    mat, labels = knn.file2mat(db_path)
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

def test_perseon():
    # Get a new test vector
    result_str = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(input('percentage of time spent playing video games?'))
    ff_miles = float(input('frequent flier miles earned per year?'))
    ice_cream = float(input('liters of ice cream consumed per year?'))

    # Predict a result
    k = 3
    ratio = 0.10
    db_path = 'KNN\\datingTestSet2.txt'
    mat, labels = knn.file2mat(db_path)
    nmat, ranges, mins = knn.scale(mat)
    x = np.array([ff_miles, percent_tats, ice_cream])
    result = knn.classify((x - mins) / ranges, nmat, labels, k)
    print('You will probably like this person: ', result_str[int(result) - 1])

def test_hand_writing_digits():
    # Load train dataset
    labels = []
    tr_folder_path = 'KNN\\trainingDigits'
    tr_file_list = os.listdir(tr_folder_path)
    m, rows, cols = len(tr_file_list), 32, 32
    tr_mat = np.zeros((m, rows * cols))
    for i in range(m):
        path = tr_file_list[i]
        name = path.split('.')[0]
        label_info = int(name.split('_')[0])
        labels.append(label_info)
        tr_mat[i, :] = knn.img2vec(tr_folder_path + '\\' + path)

    # Load test dataset
    k, error = 3, 0
    te_folder_path = 'KNN\\testDigits'
    te_file_list = os.listdir(te_folder_path)
    n = len(te_folder_path)
    for i in range(n):
        path = te_file_list[i]
        name = path.split('.')[0]
        label_info = int(name.split('_')[0])
        te_vec = knn.img2vec(te_folder_path + '\\' + path)
        result = knn.classify(te_vec, tr_mat, labels, k)
        print('the classifier came back with: %s, the real answer is: %s' % (result, label_info))
        if result != label_info:
            error += 1
    print('the total number of errors is: %d' % error)
    print('the total error rate is: %f' % (error / float(n)))

# Do test
test_dating()
#test_perseon()
#test_hand_writing_digits()