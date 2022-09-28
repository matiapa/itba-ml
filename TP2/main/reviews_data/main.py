from knn import get_nearest_neighbors, normal_knn, weighted_knn, evaluate

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import datetime


def cross_validation(df, knn_k, knn_method, validation_k=5):
    block_size = len(df) // validation_k

    best_acc = 0
    best_test_set = None
    best_train_set = None


    for i in range(1,validation_k):
        test_set = df.iloc[i * block_size: (i + 1) * block_size]
        train_set = df.drop(test_set.index)
        
        acc = evaluate(train_set.to_numpy(), test_set.to_numpy(), knn_k, knn_method)


        if acc > best_acc:
            best_acc = acc
            best_test_set = test_set
            best_train_set = train_set


 
    return best_acc, best_train_set, best_test_set


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


def accuracy_vs_knn_k(df):
    #for values of k from 1 to 10, get the accuracy of the knn algorithm
    #graph the accuracy vs k

    accuracies = []
    accuracies_normal = []
    for k_knn in range(1, 20):
        acc,train_set, test_set = cross_validation(df, k_knn, "weighted")
        accuracies_normal.append(acc)

        acc,train_set, test_set = cross_validation(df, k_knn, "normal")
        accuracies.append(acc)


    #graph the accuracy vs k
    #plot with dots and lines
    plt.plot(range(1, 20), accuracies, 'bo-')
    plt.plot(range(1, 20), accuracies_normal, 'ro-')
    plt.legend(['weighted', 'normal'])
    plt.xlabel('knn k')
    plt.ylabel('accuracy')
    plt.xticks(range(1, 20))

    #get timestamp
    timestamp = str(datetime.datetime.now())
    timestamp = timestamp.replace(" ", "_")
    timestamp = timestamp.replace(":", "-")
    plt.savefig("out/accuracy_vs_k_" + timestamp + ".png")
    
    plt.show()


def accuracy_vs_validation_k(df, knn_k, validation_k_max=10):
    accuracies = []
    accuracies_normal = []
    for i in range(2, validation_k_max):
        acc, _, _ = cross_validation(df, knn_k, "weighted", i)
        accuracies.append(acc)

    for i in range(2, validation_k_max):
        acc, _, _ = cross_validation(df, knn_k, "normal", i)
        accuracies_normal.append(acc)

    plt.plot(range(2, validation_k_max), accuracies, 'bo-')
    plt.xlabel('ammount of blocks in cross validation')
    plt.ylabel('accuracy')

    plt.plot(range(2, validation_k_max), accuracies_normal, 'ro-')

    #set legend to read that red is weighted and blue is normal
    plt.legend(['weighted', 'normal'])
    #get timestamp
    timestamp = str(datetime.datetime.now())
    timestamp = timestamp.replace(" ", "_")
    timestamp = timestamp.replace(":", "-")
    plt.savefig("out/accuracy_vs_validation_k_" + timestamp + ".png")

    plt.show()


df = pd.read_csv('in/reviews_sentiment_parsed.csv', sep=';')
df = df.sample(frac=1).reset_index(drop=True)

accuracy_vs_knn_k(df)