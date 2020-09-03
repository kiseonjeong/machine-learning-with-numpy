from naive_bayes import bayes

post_list, class_list = bayes.load_dataset()
voca_list = bayes.create_voca_list(post_list)
word_vec = bayes.word2vec(voca_list, post_list[0])
print(word_vec)