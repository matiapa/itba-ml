import pandas as pd
import matplotlib.pyplot as plt
from SimplePerceptron import SimplePerceptron

def best_learning_rate():
    df = pd.read_csv('data.csv')
    df = df.sample(frac=1).reset_index(drop=True) # shuffle data
    train_set = df[20:]
    test_set = df[:20]

    X = train_set[['x', 'y']].values
    labels = train_set['class'].values

    learning_rates = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5]
    errors = []
    for learning_rate in learning_rates:
        perceptron = SimplePerceptron(2, learning_rate=learning_rate)
        perceptron.train(X, labels, 1)
        errors.append(perceptron.error(test_set[['x', 'y']].values, test_set['class'].values, 1, use_min_weights=True))

    plt.plot(learning_rates, errors)
    plt.show()

def error_per_iteration():
    df = pd.read_csv('data_1.csv')
    df = df.sample(frac=1).reset_index(drop=True) # shuffle data
    train_size = int(len(df) * 0.8)
    train_set = df[:train_size]
    test_set = df[train_size:]
    samples = 50

    X = train_set[['x', 'y']].values
    labels = train_set['class'].values

    errors = []

    # for s in range(samples):
    #     print('Sample', s)
    perceptron = SimplePerceptron(2, learning_rate=0.001)
    perceptron.train(X, labels, 1000)
    errors.append(perceptron.errors_per_iteration)

    # average error per iteration
    # errors_avg = [sum([errors[i][j] for i in range(samples)]) / samples for j in range(len(errors[0]))]
    # errors_std = [np.std([errors[i][j] for i in range(samples)]) for j in range(len(errors[0]))]

    print(perceptron.errors_per_iteration)
    plt.errorbar(range(len(perceptron.errors_per_iteration)), perceptron.errors_per_iteration)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.show()

# best_learning_rate()
error_per_iteration()