import matplotlib.patches as mpatches
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

sys.path.append("..")
from decision_tree.random_forest import random_forest
from decision_tree.tree import DecisionTree
from decision_tree.utils import precision
from main.attributes import attributes, target_attr


def cross_validation(df, max_depth=8, min_samples=100, k=5):
    block_size = len(df) // k
    best_acc = 0
    best_train_block = []
    best_test_block = []
    for i in range(k):
        test_set = df.iloc[i * block_size: (i + 1) * block_size]
        train_set = df.drop(test_set.index)
        tree = DecisionTree(max_depth, min_samples)
        tree.train(train_set, attributes, target_attr)
        acc = precision(tree, test_set, target_attr)
        if acc > best_acc:
            best_acc = acc
            best_train_block = train_set
            best_test_block = test_set
    return best_acc, best_train_block, best_test_block


def cross_validation_analysis():
    k = [4, 5, 6, 7, 8, 9, 10, 11]
    df = pd.read_csv('../data/german_credit_proc.csv', dtype=object)
    precisions = []

    for block in k:
        print("Block: " + str(block))
        # ! Guarda porque la cantidad de min samples afecta el resultado
        best_acc, _, _ = cross_validation(df, k=block, min_samples=100, max_depth=8)
        precisions.append(best_acc)

    plt.plot(k, precisions)
    plt.xlabel('k')
    plt.ylabel('Precisión (%)')
    plt.savefig('out/cross_validation.png')
    plt.show()


cross_validation_analysis()


def tree_height_precision():
    max_depth = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    df = pd.read_csv('../data/german_credit_proc.csv', dtype=object)
    k = 5

    train_precisions = []
    test_precisions = []
    for depth in max_depth:
        print("Depth: " + str(depth))
        tree = DecisionTree(max_depth=depth, min_samples=0)

        _, train_set, test_set = cross_validation(df, depth, 0, k)

        tree.train(train_set, attributes, target_attr)
        test_precisions.append(precision(tree, test_set, target_attr))
        train_precisions.append(precision(tree, train_set, target_attr))

    plt.plot(max_depth, test_precisions, label='Test Set')
    plt.plot(max_depth, train_precisions, label='Train Set')
    plt.xlabel('Profundidad máxima')
    plt.ylabel('Precisión (%)')
    plt.legend()
    plt.savefig('out/height_precision.png')
    plt.show()


# tree_height_precision()


def min_samples_analysis():
    df = pd.read_csv('../data/german_credit_proc.csv', dtype=object)
    min_samples = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    k = 5

    train_precisions = []
    test_precisions = []

    for size in min_samples:
        tree = DecisionTree(max_depth=3, min_samples=size)
        _, train_set, test_set = cross_validation(df, 3, size, k)
        tree.train(train_set, attributes, target_attr)
        train_precisions.append(precision(tree, train_set, target_attr))
        test_precisions.append(precision(tree, test_set, target_attr))

    plt.plot(min_samples, train_precisions, label='Train Set')
    plt.plot(min_samples, test_precisions, label='Test Set')
    plt.xlabel('Mínimo de muestras')
    plt.ylabel('Precisión (%)')
    plt.legend()
    plt.savefig('out/min_samples.png')
    plt.show()


# min_samples_analysis()


def sample_size_analysis():
    df = pd.read_csv('../data/german_credit_proc.csv', dtype=object)
    samples = 10
    sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    avg_acc = []
    std_acc = []

    for size in sizes:
        print("Size: " + str(size))
        acc = []
        for _ in range(samples):
            acc.append(
                random_forest(df, attributes, target_attr, int(len(df) * size), max_depth=5, min_samples=0, n_trees=70))
        avg_acc.append(np.mean(acc))
        std_acc.append(np.std(acc))

    with open('results.txt', 'w') as f:
        for i in range(len(sizes)):
            f.write(str(sizes[i]) + ' ' + str(avg_acc[i]) + ' ' + str(std_acc[i]) + '\n')

    plt.errorbar(sizes, avg_acc, yerr=std_acc, fmt='o')


def random_forest_analysis():
    df = pd.read_csv('../data/german_credit_proc.csv', dtype=object)
    df = df.sample(frac=1).reset_index(drop=True)

    train_set = df.iloc[0:900]
    test_set = df.iloc[900:1000]
    trees = random_forest(train_set, attributes, target_attr, 700, max_depth=8, min_samples=100, n_trees=64)
    print("Finished training")

    matches = 0
    i = 0
    not_found = 0

    print("Testing...")
    for index, row in test_set.iterrows():
        print(i)
        results = {}
        for tree in trees:
            result = tree.evaluate(row)
            if result not in results:
                results[str(result)] = 0

            results[str(result)] += 1

        obtained = max(results, key=results.get)

        if str(obtained) == '?':
            not_found += 1

        if str(obtained) == str(row['Creditability']):
            matches += 1

        print("The majority vote is: " + str(obtained) + " expected: " + str(row['Creditability']) + " - 1: " + str(
            results.get('1')) + " - 0: " + str(results.get('0')) + " - ?: " + str(results.get('?')))
        i += 1

    print("Matched: " + str(matches) + " not found " + str(not_found))
    print("Precision of the random forest: " + str(matches / len(test_set)))


# random_forest_analysis()


def variables_analysis():
    df = pd.read_csv('../data/german_credit.csv')

    variable = 'Age (years)'

    groups = [4, 5, 6, 7, 8, 9]

    fig, ax = plt.subplots(len(groups))
    i = 0

    for group in groups:
        labels = []
        df_copy = df.copy()

        df_copy[variable] = pd.cut(df_copy[variable], group, labels=[i for i in range(group)])

        # get the values from df that have 'Creditability' == 1
        df1 = df_copy[df_copy['Creditability'] == 0]
        df2 = df_copy[df_copy['Creditability'] == 1]

        violin = ax[i].violinplot(df1[variable], vert=False)
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), 'Creditability = 0'))

        violin = ax[i].violinplot(df2[variable], vert=False)
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), 'Creditability = 1'))

        # ax[i].legend(*zip(*labels), loc=2)

        i += 1
    plt.show()


def violin_analysis():
    df = pd.read_csv('../data/german_credit_proc.csv')

    columns = df.columns.values.tolist()
    datas = {}
    data_columns = {}

    for i in range(len(columns)):
        if columns[i] == 'Creditability':
            continue

        max_val = df[columns[i]].max()

        if max_val not in datas:
            datas[max_val] = {}
            datas[max_val][0] = []
            datas[max_val][1] = []
            data_columns[max_val] = []

        datas[max_val][0].append(df[df['Creditability'] == 0][columns[i]])
        datas[max_val][1].append(df[df['Creditability'] == 1][columns[i]])
        data_columns[max_val].append(columns[i])

    for key in datas:
        labels = []
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_yticklabels(data_columns[key])
        pos = [i + 1 for i in range(len(data_columns[key]))]

        violin = ax.violinplot(datas[key][0], pos, vert=False)
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), 'Creditability = 0'))

        violin = ax.violinplot(datas[key][1], pos, vert=False)
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), 'Creditability = 1'))

        plt.yticks(pos, data_columns[key])
        plt.legend(*zip(*labels), loc=2)
        plt.tight_layout()
        plt.show()

# violin_analysis()
# test()
# parallel_analysis()
