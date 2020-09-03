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