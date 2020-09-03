import operator
import numpy as np

def create_dataset():
    """
    (function) create_dataset
    -------------------------
    Creating the dataset

    Parameter
    ---------
    - None

    Return
    ------
    - test dataset and labels
    """
    # Create test dataset
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify(x, DB, labels, k):
    """
    (function) classify
    -------------------
    Classify the input vector 'x'

    Parameter
    ---------
    - x : input vector
    - DB : database for KNN
    - labels : labels for KNN
    - k : hyperparameter K

    Return
    ------
    - predicted label
    """
    # Calculate euclidean distances
    size = DB.shape[0]
    diff = np.tile(x, (size, 1)) - DB
    dist = np.sqrt(np.sum(diff ** 2, axis=1))
    sorted_dist_idx = dist.argsort()

    # Classify the results
    cout = {}
    for i in range(k):
        vote = labels[sorted_dist_idx[i]]
        cout[vote] = cout.get(vote, 0) + 1
    sorted_count = sorted(cout.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_count[0][0]

def file2mat(path):
    """
    (function) file2mat
    -------------------
    Convert file to matrix

    Parameter
    ---------
    - path : input file path

    Return
    ------
    - converted matrix from the file
    """
    # Read all dataset
    fr = open(path)
    lines = fr.readlines()
    num_line = len(lines)

    # Set a matrix
    ndim = 3
    mat = np.zeros((num_line, ndim))
    labels = []
    for idx, line in enumerate(lines):
        temp = line.strip()
        vals = temp.split('\t')
        mat[idx, :] = vals[0:ndim]
        labels.append(vals[-1])

    return mat, labels

def scale(mat):
    """
    (function) scale
    ----------------
    Do scaling on the matrix

    Parameter
    ---------
    - mat : input matrix

    Return
    ------
    - scaled matrix
    """
    # Do scaling on the matrix
    min_vals = mat.min(0)
    max_vals = mat.max(0)
    ranges = max_vals - min_vals
    nmat = (mat - min_vals) / (max_vals - min_vals)

    return nmat, ranges, min_vals

def img2vec(path):
    """
    (function) img2vec
    ------------------
    Convert image to vector

    Parameter
    ---------
    - path : input image path

    Return
    ------
    - converted vector from the image
    """
    # Convert image data to vector
    rows, cols = 32, 32
    vec = np.zeros((1, rows * cols))
    fr = open(path)
    for i in range(rows):
        line = fr.readline()
        for j in range(cols):
            vec[0, i * cols + j] = int(line[j])

    return vec
