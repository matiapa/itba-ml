import sys

sys.path.append("..")

from decision_tree.tree import DecisionTree


# -------------------- DEFINITIONS --------------------

def random_forest(train_set, attributes, target_attr, sample_size, max_depth=4, min_samples=0, n_trees=64):
    trees = []

    for n in range(n_trees):
        # print(f"Tree {n + 1}/{n_trees}")
        trees.append(DecisionTree(max_depth=max_depth, min_samples=min_samples))

        # get random sample from train_set with replacement
        aux = train_set.sample(n=sample_size, replace=True)

        # train tree
        trees[n].train(aux, attributes, target_attr)

    return trees


def random_forest_evaluate(trees, test_set):
    results = {}
    for tree in trees:
        result = tree.evaluate(test_set)
        if result not in results:
            results[str(result)] = 0

        results[str(result)] += 1

    return max(results, key=results.get)


def random_forest_precision(trees, test_set, target_attr):
    correct = 0
    for index, row in test_set.iterrows():
        obtained = random_forest_evaluate(trees, row)
        if str(obtained) == str(row[target_attr.label]):
            correct += 1

    return correct / len(test_set) * 100

