import sys

sys.path.append("..")
sys.path.append("../..")

import pandas as pd
import seaborn as sn

from matplotlib import pyplot as plt
from decision_tree.random_forest import random_forest, random_forest_evaluate, random_forest_precision
from decision_tree.utils import precision, cross_validation
from attributes import attributes, target_attr


def metric_to_percent(metric, total):
    return str(metric) + " (" + str(round(metric / total * 100, 2)) + "%)"


def create_confusion_matrix(metrics, total, name='confusion_matrix', title='Matriz de Confusión'):
    matrix = [[metrics['tp'], metrics['fn']], [metrics['fp'], metrics['tn']]]
    labels = [[metric_to_percent(metrics['tp'], total), metric_to_percent(metrics['fn'], total)],
              [metric_to_percent(metrics['fp'], total), metric_to_percent(metrics['tn'], total)]]
    df_cm = pd.DataFrame(matrix, index=['Positivo', 'Negativo'], columns=['Positivo', 'Negativo'])
    plt.figure(figsize=(5, 3))
    sn.heatmap(df_cm, annot=labels, fmt='')

    # increase font size
    sn.set(font_scale=1.4)

    # save the figure
    plt.title(title)
    plt.savefig('out/' + name + '.png')
    plt.show()


def algorithms_confusion_matrix_analysis():
    df = pd.read_csv('in//german_credit_proc.csv', dtype=object)
    df = df.sample(frac=1).reset_index(drop=True)
    k = 10
    max_depth = 3
    min_samples = 55

    metrics = {
        'normal': {
            'tp': 0,
            'tn': 0,
            'fp': 0,
            'fn': 0
        },
        'random_forest': {
            'tp': 0,
            'tn': 0,
            'fp': 0,
            'fn': 0
        }
    }

    # Cross validation
    _, tree, train_set, test_set = cross_validation(df, attributes, target_attr, max_depth, min_samples, k)
    for index, row in test_set.iterrows():
        result = tree.evaluate(row)
        if str(result) == str(row['Creditability']):
            if str(result) == '1':
                metrics['normal']['tp'] += 1
            else:
                metrics['normal']['tn'] += 1
        else:
            if str(result) == '1':
                metrics['normal']['fp'] += 1
            else:
                metrics['normal']['fn'] += 1

    create_confusion_matrix(metrics['normal'], len(test_set), title='Matriz de Confusión: Validación Cruzada')

    # Random Forest
    sample_size = int(len(train_set) * 0.7)
    trees = random_forest(train_set, attributes, target_attr, sample_size, max_depth=max_depth, min_samples=min_samples,
                          n_trees=30)
    for index, row in test_set.iterrows():
        results = {}
        for tree in trees:
            result = tree.evaluate(row)
            if result not in results:
                results[str(result)] = 0

            results[str(result)] += 1

        obtained = max(results, key=results.get)
        if str(obtained) == str(row['Creditability']):
            if str(obtained) == '1':
                metrics['random_forest']['tp'] += 1
            else:
                metrics['random_forest']['tn'] += 1
        else:
            if str(obtained) == '1':
                metrics['random_forest']['fp'] += 1
            else:
                metrics['random_forest']['fn'] += 1

    create_confusion_matrix(metrics['random_forest'], len(test_set), name='random_forest_matrix',
                            title='Matriz de Confusión: Random Forest')


def min_samples_analysis():
    df = pd.read_csv('in//german_credit_proc.csv', dtype=object)
    min_samples = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    k = 10

    train_precisions = []
    test_precisions = []

    for size in min_samples:
        _, tree, train_set, test_set = cross_validation(df, attributes, target_attr, 3, size, k)
        train_precisions.append(precision(tree, train_set, target_attr))
        test_precisions.append(precision(tree, test_set, target_attr))

    plt.plot(min_samples, train_precisions, label='Train Set')
    plt.plot(min_samples, test_precisions, label='Test Set')
    plt.xlabel('Mínimo de muestras')
    plt.ylabel('Precisión (%)')
    plt.legend()
    plt.savefig('out/min_samples.png')
    plt.show()


def tree_height_precision():
    max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    df = pd.read_csv('in//german_credit_proc.csv', dtype=object)
    df = df.sample(frac=1).reset_index(drop=True)
    k = 10

    train_precisions = []
    test_precisions = []
    for depth in max_depth:
        print("Depth: " + str(depth))

        _, tree, train_set, test_set = cross_validation(df, attributes, target_attr, depth, 0, k)
        test_precisions.append(precision(tree, test_set, target_attr))
        train_precisions.append(precision(tree, train_set, target_attr))

    plt.plot(max_depth, test_precisions, label='Test Set')
    plt.plot(max_depth, train_precisions, label='Train Set')
    plt.xlabel('Cantidad de nodos')
    plt.ylabel('Precisión (%)')
    plt.legend()
    plt.savefig('out/height_precision_50.png')
    plt.show()


def cross_validation_analysis():
    k = [4, 5, 6, 7, 8, 9, 10, 11]
    df = pd.read_csv('in//german_credit_proc.csv', dtype=object)
    df = df.sample(frac=1).reset_index(drop=True)
    precisions = []

    for block in k:
        print("Block: " + str(block))
        best_acc, _, _, _ = cross_validation(df, attributes, target_attr, k=block, min_samples=50, max_depth=8)
        precisions.append(best_acc)

    plt.plot(k, precisions)
    plt.xlabel('k')
    plt.ylabel('Precisión (%)')
    plt.savefig('out/cross_validation.png')
    plt.show()


def random_forest_analysis():
    df = pd.read_csv('in//german_credit_proc.csv', dtype=object)
    # df = df.sample(frac=1).reset_index(drop=True)
    train_set = df[:int(len(df) * 0.9)]
    test_set = df[int(len(df) * 0.9):]
    sample_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # train_precisions = []
    test_precisions = []
    for size in sample_sizes:
        print("Size: " + str(size))

        trees = random_forest(train_set, attributes, target_attr, int(len(train_set) * size), max_depth=3,
                              min_samples=50, n_trees=30)

        test_precisions.append(random_forest_precision(trees, test_set, target_attr))

    plt.plot(sample_sizes, test_precisions)
    plt.xlabel('Tamaño de la muestra')
    plt.ylabel('Precisión (%)')
    plt.savefig('out/random_forest.png')
    plt.show()


def dt_node_amount_analysis():
    df = pd.read_csv('in//german_credit_proc.csv', dtype=object)
    df = df.sample(frac=1).reset_index(drop=True)
    k = 10
    max_depths = [1, 2, 3, 4, 5, 6, 7]

    train_precisions = []
    test_precisions = []
    nodes_amount = []

    for depth in max_depths:
        print("Depth: " + str(depth))
        _, tree, train_set, test_set = cross_validation(df, attributes, target_attr, depth, 0, k)
        train_precisions.append(precision(tree, train_set, target_attr))
        test_precisions.append(precision(tree, test_set, target_attr))
        nodes_amount.append(tree.get_node_amount())
        print(nodes_amount[-1])

    plt.plot(nodes_amount, test_precisions, label='Test Set', marker='o')
    plt.plot(nodes_amount, train_precisions, label='Train Set', marker='o')

    # add annotation based on max_depth
    for i, txt in enumerate(max_depths):
        plt.annotate('h='+str(txt), (nodes_amount[i], test_precisions[i]))
        plt.annotate('h='+str(txt), (nodes_amount[i], train_precisions[i]))

    plt.xlabel('Cantidad de nodos')
    plt.ylabel('Precisión (%)')
    plt.legend()
    plt.savefig('out/nodes_amount.png')
    plt.show()


def rf_node_amount_analysis():
    df = pd.read_csv('in//german_credit_proc.csv', dtype=object)
    df = df.sample(frac=1).reset_index(drop=True)
    max_depths = [1, 2, 3, 4, 5, 6, 7]
    train_precisions = []
    test_precisions = []
    nodes_amount = []
    train_set = df[:int(len(df) * 0.9)]
    test_set = df[int(len(df) * 0.9):]
    sample_size = int(len(train_set) * 0.7)

    for depth in max_depths:
        print("Depth: " + str(depth))

        trees = random_forest(train_set, attributes, target_attr, sample_size, max_depth=depth, min_samples=0, n_trees=30)
        amounts = []
        for tree in trees:
            amounts.append(tree.get_node_amount())
        nodes_amount.append(int(sum(amounts) / len(amounts)))
        print(nodes_amount[-1])
        train_precisions.append(random_forest_precision(trees, test_set, target_attr))
        test_precisions.append(random_forest_precision(trees, train_set, target_attr))

    plt.plot(nodes_amount, test_precisions, label='Test Set', marker='o')
    plt.plot(nodes_amount, train_precisions, label='Train Set', marker='o')

    for i, txt in enumerate(max_depths):
        plt.annotate('h='+str(txt), (nodes_amount[i], test_precisions[i]))
        plt.annotate('h='+str(txt), (nodes_amount[i], train_precisions[i]))

    plt.xlabel('Cantidad de nodos')
    plt.ylabel('Precisión (%)')
    plt.legend()
    plt.savefig('out/nodes_amount_rf.png')
    plt.show()


rf_node_amount_analysis()

# node_amount_analysis()

# random_forest_analysis()

# algorithms_confusion_matrix_analysis()

# cross_validation_analysis()

# tree_height_precision()

# min_samples_analysis()
