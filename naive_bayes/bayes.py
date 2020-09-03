import numpy as np

def load_dataset():
    """
    (function) load_dataset
    -----------------------
    Load test dataset

    Parameter
    ---------
    - None

    Return
    ------
    - dataset
    """
    # Create test dataset
    post_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]

    return post_list, class_vec

def create_voca_list(dataset):
    """
    (function) create_voca_list
    ---------------------------
    Create voca dataset

    Parameter
    ---------
    - None

    Return
    ------
    - voca list
    """
    # Create a voca list
    voca_set = set([])
    for doc in dataset:
        voca_set = voca_set | set(doc)

    return list(voca_set)

def word2vec(voca_list, input_set):
    """
    (function) word2vec
    -------------------
    Set a vector from the words

    Parameter
    ---------
    - None

    Return
    ------
    - word vector
    """
    # Convert words to a vector
    vec = [0] * len(voca_list)
    for word in input_set:
        if word in voca_list:
            vec[voca_list.index(word)] = 1
        else:
            print('the word: %s is not in my vocabulary!' % word)

    return vec

def train(mat, cat):
    """
    (function) train
    ----------------
    Train on input matrix

    Parameter
    ---------
    - mat : input matrix
    - cat : category information

    Return
    ------
    - probabilities
    """
    # Calculate probabilties
    num_docs = len(mat)
    num_words = len(mat[0])
    pr_abusive = sum(cat) / float(num_docs)
    #p0_num, p1_num = np.zeros(num_words), np.zeros(num_words)
    #p0_denom, p1_denom = 0.0, 0.0
    p0_num, p1_num = np.ones(num_words), np.ones(num_words)    # zero 확률 방지
    p0_denom, p1_denom = 2.0, 2.0
    for i in range(num_docs):
        if cat[i] == 1:
            p1_num += mat[i]
            p1_denom += sum(mat[i])
        else:
            p0_num += mat[i]
            p0_denom += sum(mat[i])
    #p0_vec, p1_vec = p0_num / p0_denom, p1_num / p1_denom
    p0_vec, p1_vec = np.log(p0_num / p0_denom), np.log(p1_num / p1_denom)    # underflow 방지

    return p0_vec, p1_vec, pr_abusive

def classify(vec2class, p0_vec, p1_vec, p1_class):
    """
    (function) classify
    -------------------
    Classify the input vector

    Parameter
    ---------
    - mat : input matrix
    - cat : category information

    Return
    ------
    - classify results
    """
    # Find an argmax
    p0 = sum(vec2class * p0_vec) + np.log(1.0 - p1_class)
    p1 = sum(vec2class * p1_vec) + np.log(p1_class)
    if p1 > p0:
        return 1
    else:
        return 0

def bag_of_words(voca_list, input_set):
    """
    (function) bag_of_words
    -----------------------
    Create bag of words

    Parameter
    ---------
    - None

    Return
    ------
    - bag of words vector
    """
    # Convert words to a vector
    vec = [0] * len(voca_list)
    for word in input_set:
        if word in voca_list:
            vec[voca_list.index(word)] += 1

    return vec