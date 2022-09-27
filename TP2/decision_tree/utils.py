from decision_tree.tree import DecisionTree


def tab(depth):
    return f"{'|'.join([' ' for _ in range(depth)])}|_"


def cprint(string, c, d):
    colors = {'g': '\033[92m', 'b': '\033[94m', 'y': '\033[93m', 'r': '\033[91m'}
    print(f'{tab(d)}{colors[c]}{string}\033[0m')


def srange(s, e):
    return list(map(lambda n: f'{n}', range(s, e + 1)))


def precision(tree, df, target_attr):
    return round(sum([1 if tree.evaluate(s) == s[target_attr.label] else 0 for _, s in df.iterrows()]) / len(df) * 100)


def cross_validation(df, attributes, target_attr, max_depth=8, min_samples=100, k=5):
    block_size = len(df) // k
    best_acc = 0
    best_tree = None
    best_test_set = None
    best_train_set = None
    for i in range(k):
        test_set = df.iloc[i * block_size: (i + 1) * block_size]
        train_set = df.drop(test_set.index)
        tree = DecisionTree(max_depth, min_samples)
        tree.train(train_set, attributes, target_attr)
        acc = precision(tree, test_set, target_attr)
        if acc > best_acc:
            best_acc = acc
            best_tree = tree
            best_test_set = test_set
            best_train_set = train_set
    return best_acc, best_tree, best_train_set, best_test_set
