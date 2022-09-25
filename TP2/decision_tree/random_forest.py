import sys

sys.path.append("..")

from decision_tree.tree import DecisionTree


# -------------------- DEFINITIONS --------------------

def random_forest(df, attributes, target_attr, sample_size, max_depth=4, min_samples=0, n_trees=64):
    trees = []

    train_set = df.sample(frac=1).reset_index(drop=True)

    for n in range(n_trees):
        # print(f"Tree {n + 1}/{n_trees}")
        trees.append(DecisionTree(max_depth=max_depth, min_samples=min_samples))

        # get random sample from train_set with replacement
        aux = train_set.sample(n=sample_size, replace=True)

        # train tree
        trees[n].train(aux, attributes, target_attr)

    return trees
