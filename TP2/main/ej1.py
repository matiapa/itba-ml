import pandas as pd
import seaborn as sn
import sys
from matplotlib import pyplot as plt
sys.path.append("..")
from decision_tree.random_forest import random_forest
from decision_tree.utils import precision, cross_validation
from main.attributes import attributes, target_attr


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
    plt.savefig('out/'+name+'.png')
    plt.show()


def algorithms_confusion_matrix_analysis():
    df = pd.read_csv('../data/german_credit_proc.csv', dtype=object)
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
    trees = random_forest(train_set, attributes, target_attr, sample_size, max_depth=max_depth, min_samples=min_samples, n_trees=64)
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

    create_confusion_matrix(metrics['random_forest'], len(test_set), name='random_forest_matrix', title='Matriz de Confusión: Random Forest')


algorithms_confusion_matrix_analysis()


def min_samples_analysis():
    df = pd.read_csv('../data/german_credit_proc.csv', dtype=object)
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


# min_samples_analysis()


def tree_height_precision():
    max_depth = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    df = pd.read_csv('../data/german_credit_proc.csv', dtype=object)
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


# tree_height_precision()


def cross_validation_analysis():
    k = [4, 5, 6, 7, 8, 9, 10, 11]
    df = pd.read_csv('../data/german_credit_proc.csv', dtype=object)
    df = df.sample(frac=1).reset_index(drop=True)
    precisions = []

    for block in k:
        print("Block: " + str(block))
        # ! Guarda porque la cantidad de min samples afecta el resultado
        best_acc, _, _, _ = cross_validation(df, attributes, target_attr, k=block, min_samples=50, max_depth=8)
        precisions.append(best_acc)

    plt.plot(k, precisions)
    plt.xlabel('k')
    plt.ylabel('Precisión (%)')
    plt.savefig('out/cross_validation.png')
    plt.show()


# cross_validation_analysis()

