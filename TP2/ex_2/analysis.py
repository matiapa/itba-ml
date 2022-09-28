import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from ex_2.knn import get_nearest_neighbors, normal_knn, weighted_knn
from ex_2.main import cross_validation


def metric_to_percent(metric, total):
    return str(metric) + " (" + str(round(metric / total * 100, 2)) + "%)"


def create_confusion_matrix(metrics, total, name='confusion_matrix', title='Matriz de Confusión'):
    df_cm = pd.DataFrame(metrics, index=[str(i + 1) for i in range(5)], columns=[str(i + 1) for i in range(5)])
    labels = [[metric_to_percent(value, total) for value in metrics[i]] for i in range(5)]
    plt.figure()
    sn.heatmap(df_cm, annot=labels, fmt='')

    # increase font size
    sn.set(font_scale=1.4)

    # save the figure
    plt.title(title)
    plt.savefig('out/' + name + '.png')
    plt.show()


def algorithms_confusion_matrix_analysis():
    df = pd.read_csv('reviews_sentiment_parsed.csv', sep=';')
    df = df.sample(frac=1).reset_index(drop=True)
    k = 6

    metrics = {
        'normal': [[0 for _ in range(5)] for _ in range(5)],
        'weighted': [[0 for _ in range(5)] for _ in range(5)],
    }

    _, train_set, test_set = cross_validation(df, knn_k=k, knn_method='normal', validation_k=5)
    train_set = train_set.to_numpy()
    test_set = test_set.to_numpy()

    filtered_train_set = []
    for i in range(len(train_set)):
        filtered_train_set.append([train_set[i][2], train_set[i][3], train_set[i][6], train_set[i][5]])

    for i in range(len(test_set)):
        nearest_neighbors = get_nearest_neighbors(filtered_train_set, test_set[i], k)

        normal_prediction = normal_knn(nearest_neighbors)
        weighted_prediction = weighted_knn(nearest_neighbors, test_set[i])

        expected = test_set[i][5]

        metrics['normal'][expected - 1][normal_prediction - 1] += 1
        metrics['weighted'][expected - 1][weighted_prediction - 1] += 1

    create_confusion_matrix(metrics['normal'], len(test_set), title='Matriz de Confusión: K-NN')
    create_confusion_matrix(metrics['weighted'], len(test_set), name='random_forest_matrix',
                            title='Matriz de Confusión: K-NN Ponderado')


# algorithms_confusion_matrix_analysis()
