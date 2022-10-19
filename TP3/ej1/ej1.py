import pandas as pd
import matplotlib.pyplot as plt
from SVM import SVM
from SimplePerceptron import SimplePerceptron
import numpy as np
from utils import plot_results

def ej1():
    df = pd.read_csv('data.csv')
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

# ej1()
    

def ej3():
    df = pd.read_csv('data.csv')
    df = df.sample(frac=1).reset_index(drop=True) # shuffle data
    train_set = df[20:]
    test_set = df[:20]

    X = train_set[['x', 'y']].values
    labels = train_set['class'].values

    svm = SVM(C=0.2, input_size=2, b=1)
    svm.train(X, labels)
    print(svm.weight)

    test_set['prediction'] = test_set.apply(lambda row: svm.evaluate(row[['x', 'y']].values), axis=1)

    correct = test_set[test_set['class'] == test_set['prediction']].shape[0]
    print(f'Accuracy: {correct / test_set.shape[0]}')

    plot_results(test_set)

ej3()