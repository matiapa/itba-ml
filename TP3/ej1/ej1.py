import pandas as pd
import matplotlib.pyplot as plt
from SVM import SVM
from SimplePerceptron import SimplePerceptron
import numpy as np
from utils import plot_results, create_dataset

def optimum_separation(df, x):
    gradient = -1
    c = 5
    df_1 = df[df['class'] == 1][['x', 'y']].values
    distances = np.abs(gradient * df_1[:, 0] - df_1[:, 1] + c) / np.sqrt(gradient ** 2 + 1)
    idx1 = np.argsort(distances)

    df_2 = df[df['class'] == -1][['x', 'y']].values
    distances = np.abs(gradient * df_2[:, 0] - df_2[:, 1] + c) / np.sqrt(gradient ** 2 + 1)
    idx2 = np.argsort(distances)
    
    x1, x2 = df_1[idx1[0], 0], df_1[idx1[1], 0]
    y1, y2 = df_1[idx1[0], 1], df_1[idx1[1], 1]
    gradient = (y2 - y1) / (x2 - x1)
    c_1 = y1 - gradient * x1
    y_1 = gradient * x + c_1

    x1 = df_2[idx2[0], 0]
    y1 = df_2[idx2[0], 1]
    c_2 = y1 - gradient * x1
    y_2 = gradient * x + c_2

    distance = np.abs(c_2 - c_1) / np.sqrt(gradient ** 2 + 1)
    y_3 = (y_1 + y_2) / 2

    return y_1, y_2, y_3

def cross_validation(df, k=10):
    df = df.sample(frac=1).reset_index(drop=True)
    train_block = None
    test_block = None
    block_size = df.shape[0] // k
    best_accuracy = 0
    for i in range(k):
        train_set = df[:i * block_size]
        test_set = df[i * block_size:(i + 1) * block_size]

        X = train_set[['x', 'y']].values
        labels = train_set['class'].values
        perceptron = SimplePerceptron(2)
        perceptron.train(X, labels, 1)

        test_set['prediction'] = test_set.apply(lambda row: perceptron.predict(row[['x', 'y']].values, 1), axis=1)

        correct = test_set[test_set['class'] == test_set['prediction']].shape[0]
        accuracy = correct / test_set.shape[0]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            train_block = train_set
            test_block = test_set
    return best_accuracy, train_block, test_block

def ej1():
    df = pd.read_csv('data_1.csv')
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
    y_1, y_2, y_3 = optimum_separation(test_set, x)
    plt.plot(x, y_1, c='lightblue', linestyle='dashed')
    plt.plot(x, y_2, c='lightblue', linestyle='dashed')
    plt.plot(x, y_3, c='green', label='Optimum separation')
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
    df = pd.read_csv('data_1.csv')
    df = df.sample(frac=1).reset_index(drop=True) # shuffle data
    train_set = df[:80]
    test_set = df[80:]

    X = train_set[['x', 'y']].values
    labels = train_set['class'].values

    svm = SVM(C=0.5, input_size=2, b=0)
    svm.train(X, labels)
    print("Weights: ", svm.weights)
    print("B: ", svm.b)

    x = np.linspace(0, 5, 100)
    y = (-svm.weights[0] * x + svm.b) / svm.weights[1]
    plt.plot(x, y, c='orange')
    print("y = ", -svm.weights[0] / svm.weights[1], "x + ", svm.b / svm.weights[1])

    test_set['prediction'] = test_set.apply(lambda row: svm.evaluate(row[['x', 'y']].values), axis=1)

    correct = test_set[test_set['class'] == test_set['prediction']].shape[0]
    print(f'Accuracy: {correct / test_set.shape[0]}')

    plot_results(test_set)

# create_dataset(size=200, dist=0.2, bad_points=5, file_name='data_2')
# ej1()
ej3()

