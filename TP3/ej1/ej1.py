import pandas as pd
import matplotlib.pyplot as plt
from SVM import SVM
from SimplePerceptron import SimplePerceptron
import numpy as np
from utils import plot_results

def ej1():
    df = pd.read_csv('data_2.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    train_set = df[:80]
    test_set = df[80:]

    X = train_set[['x', 'y']].values
    labels = train_set['class'].values
    perceptron = SimplePerceptron(2)
    perceptron.train(X, labels, 1)

    test_set['prediction'] = test_set.apply(lambda row: perceptron.predict(row[['x', 'y']].values, 1), axis=1)

    correct = test_set[test_set['class'] == test_set['prediction']].shape[0]
    print(f'Accuracy: {correct / test_set.shape[0]}')
    plot_results(test_set)

    x = np.linspace(0, 5, 100)
    gradient = -perceptron.weights[1] / perceptron.weights[2]
    c = -perceptron.weights[0] / perceptron.weights[2]
    y = gradient * x + c
    plt.plot(x, y, c='orange', label='Perceptron')
    # y_1, y_2, y_3 = optimum_separation(test_set, x)
    # plt.plot(x, y_1, c='lightblue', linestyle='dashed')
    # plt.plot(x, y_2, c='lightblue', linestyle='dashed')
    # plt.plot(x, y_3, c='green', label='Optimum separation')
    plot_results(test_set, file_name=f'results1_{round(accuracy, 3)}')
    

def ej3():
    df = pd.read_csv('data_2.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    train_set = df[:80]
    test_set = df[80:]

    X = train_set[['x', 'y']].values
    labels = train_set['class'].values
    perceptron = SimplePerceptron(2)
    perceptron.train(X, labels, 1)

    test_set['prediction'] = test_set.apply(lambda row: perceptron.predict(row[['x', 'y']].values, 1), axis=1)

    correct = test_set[test_set['class'] == test_set['prediction']].shape[0]
    accuracy = correct / test_set.shape[0]
    print(f'Accuracy: {accuracy}')
    x = np.linspace(0, 5, 100)
    gradient = -perceptron.weights[1] / perceptron.weights[2]
    c = -perceptron.weights[0] / perceptron.weights[2]
    y = gradient * x + c
    plt.plot(x, y, c='orange', label='Perceptron')
    plot_results(test_set, file_name=f'results2_{round(accuracy, 3)}')


def ej4():
    df = pd.read_csv('data_2.csv')
    df = df.sample(frac=1).reset_index(drop=True)
    train_size = int(0.6*df.shape[0])
    train_set = df[:train_size]
    test_set = df[train_size:]

    X = train_set[['x', 'y']].values
    labels = train_set['class'].values

    svm = SVM(C=1, input_size=2, b=0, Aw=0.1, Ab=0.1, k=0.01)
    svm.train(X, labels, iter=1000)

    x = np.linspace(0, 5, 100)
    gradient = -svm.weights[0] / svm.weights[1]
    c = -svm.b / svm.weights[1]
    y = gradient * x + c
    plt.plot(x, y, c='orange', label='SVM')

    # perceptron = SimplePerceptron(2)
    # perceptron.train(X, labels, 1)
    # gradient = -perceptron.weights[1] / perceptron.weights[2]
    # c = -perceptron.weights[0] / perceptron.weights[2]
    # y = gradient * x + c
    # plt.plot(x, y, c='purple', label='Perceptron')

    test_set['prediction'] = test_set.apply(lambda row: svm.evaluate(row[['x', 'y']].values), axis=1)

    correct = test_set[test_set['class'] == test_set['prediction']].shape[0]
    accuracy = correct / test_set.shape[0]
    print(f'Accuracy: {accuracy}')

    # y_1, y_2, y_3 = optimum_separation(test_set, x)
    # plt.plot(x, y_1, c='lightblue', linestyle='dashed')
    # plt.plot(x, y_2, c='lightblue', linestyle='dashed')
    # plt.plot(x, y_3, c='green', label='Optimum separation')

    plot_results(test_set)

# create_dataset(size=200, dist=0.2, bad_points=0, file_name='data_3')
# ej1()
# ej3()
ej4()

