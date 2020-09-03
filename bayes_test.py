import numpy as np
from naive_bayes import bayes

def test_words():
    # Set a train vector
    post_list, class_list = bayes.load_dataset()
    voca_list = bayes.create_voca_list(post_list)
    train_mat = []
    for doc in post_list:
        train_mat.append(bayes.word2vec(voca_list, doc))

    # Train the dataset
    p0_vec, p1_vec, pr_abusive = bayes.train(np.array(train_mat), np.array(class_list))

    # Test the dataset
    test_entry = ['love', 'my', 'dalmation']
    test_doc = np.array(bayes.word2vec(voca_list, test_entry))
    print(test_entry, 'classified as: ', bayes.classify(test_doc, p0_vec, p1_vec, pr_abusive))

    # Test the dataset
    test_entry = ['stupid', 'garbage']
    test_doc = np.array(bayes.word2vec(voca_list, test_entry))
    print(test_entry, 'classified as: ', bayes.classify(test_doc, p0_vec, p1_vec, pr_abusive))

# Do test
test_words()