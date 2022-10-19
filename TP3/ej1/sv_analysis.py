import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SVM import SVM

def optimum_C():
    df = pd.read_csv('data_1.csv')
    X = df[['x', 'y']].values
    labels = df['class'].values

    C = [0.1, 1, 3, 5, 7, 10, 12]
    samples = 50
    error_avg = []
    error_std = []
    
    for i in range(len(C)):
        print('C =', C[i])
        errors = []
        for _ in range(samples):
            svm = SVM(C[i], 2, 0, 0.01, 0.01, 0.01)
            svm.train(X, labels, 1000, 0.01)
            errors.append(svm.error(X, labels))
        error_avg.append(np.mean(errors))
        error_std.append(np.std(errors))

    plt.errorbar(C, error_avg, yerr=error_std)

    plt.xlabel('C')
    plt.ylabel('Error')
    plt.show()

def error_per_iteration():
    df = pd.read_csv('data_1.csv')
    X = df[['x', 'y']].values
    labels = df['class'].values

    svm = SVM(C=10, input_size=2, b=0, Aw=0.01, Ab=0.01, k=0.01)
    svm.train(X, labels, 1000, 0.01, write_errors=True)

    df1 = pd.read_csv('error.csv')
    plt.plot(df1['iteration'], df1['error'])
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.show()

optimum_C()
# error_per_iteration()
