from decision_tree import id3

# 미 구현, 구현 진행 중
dataset, labels = id3.create_dataset()
#tree = id3.create_tree(dataset, labels)
tree = id3.retrieve_tree(0)
print(id3.classify(tree2, labels, [1, 1]))