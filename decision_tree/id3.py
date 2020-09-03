from math import log

def calc_entropy(dataset):
    """
    (function) calc_entropy
    -----------------------
    Calculate an entropy on the dataset

    Parameter
    ---------
    - dataset : input dataset

    Return
    ------
    - entropy value
    """
    # Create a label count table
    label_count = {}
    for vec in dataset:
        curr_label = vec[-1]
        if not curr_label in label_count:
            label_count[curr_label] = 0
        label_count[curr_label] += 1

    # Calculate an entropy
    num_entries = len(dataset)
    ent = 0.0
    for key in label_count:
        prob = float(label_count[key]) / num_entries
        ent -= prob * log(prob, 2)

    return ent

def create_dataset():
    """
    (function) create_dataset
    -------------------------
    Create test dataset

    Parameter
    ---------
    - None

    Return
    ------
    - test dataset
    """
    dataset = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataset, labels

def split_dataset(dataset, axis, value):
    """
    (function) split_dataset
    ------------------------
    Split input dataset

    Parameter
    ---------
    - dataset : input dataset
    - axis : split axis
    - value : separator

    Return
    ------
    - split dataset
    """
    # Split input dataset
    split = []
    for vec in dataset:
        if vec[axis] == value:
            reduced = vec[:axis]
            reduced.extend(vec[axis + 1:])
            split.append(reduced)

    return split

def choose_best_feature(dataset):
    """
    (function) choose_best_feature
    ------------------------------
    Choose best feature by entropy

    Parameter
    ---------
    - dataset : input dataset

    Return
    ------
    - best feature index
    """
    # Calculate information gain and choose the best feature
    dims = len(dataset[0]) - 1
    base = calc_entropy(dataset)
    best_info_gain, best_feature = 0.0, -1
    for i in range(dims):
        feat = [example[i] for example in dataset]
        uniq = set(feat)
        new = 0.0
        for value in uniq:
            sub_dataset = split_dataset(dataset, i, value)
            prob = len(sub_dataset) / float(len(dataset))
            new += prob * calc_entropy(sub_dataset)
        info_gain = base - new
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i

    return best_feature

def get_majority_count(class_list):
    """
    (function) get_majority_count
    -----------------------------
    Get majority count value

    Parameter
    ---------
    - class_list : class list

    Return
    ------
    - majority count
    """
    # Create a count table
    class_count = {}
    for vote in class_list:
        if not vote in class_count:
            class_count[vote] = 0
        class_count[vote] += 1

    # Sort the table by values
    sorted_class_count = sorted(class_count.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)

    return sorted_class_count

def create_tree(dataset, labels):
    """
    (function) create_tree
    ----------------------
    Create a tree

    Parameter
    ---------
    - dataset : input dataset
    - labels : label information

    Return
    ------
    - decision tree
    """
    # Check the class list
    class_list = [example[-1] for example in dataset]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(dataset[0]) == 1:
        return get_majority_count(class_list)

    # Create a subtree
    best_feat = choose_best_feature(dataset)
    best_feat_label = labels[best_feat]
    tree = {best_feat_label: {}}
    del(labels[best_feat])
    feat_values = [example[best_feat] for example in dataset]
    uniq_values = set(feat_values)
    for value in uniq_values:
        sub_labels = labels[:]
        tree[best_feat_label][value] = create_tree(split_dataset(dataset, best_feat, value), sub_labels)

    return tree

def retrieve_tree(i):
    """
    (function) retrieve_tree
    ------------------------
    Retrieve the tree

    Parameter
    ---------
    - i : tree index

    Return
    ------
    - selected tree
    """
    tree_list = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]

    return tree_list[i]

def classify(tree_model, labels, vec):
    """
    (function) classify
    -------------------
    Classify the input vector 'x'

    Parameter
    ---------
    - tree_model : modeling tree
    - labels : label information
    - vec : test vector

    Return
    ------
    - predicted label
    """
    first = [*tree_model.keys()][0]
    second = tree_model[first]
    index = labels.index(first)
    for key in second.keys():
        if vec[index] == key:
            if type(second[key]).__name__ == 'dict':
                class_label = classify(second[key], labels, vec)
            else:
                class_label = second[key]

    return class_label