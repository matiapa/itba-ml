import pandas as pd
from NaiveBayes import predict
import matplotlib.pyplot as plt
import numpy as np

# read csv file and create dataframe
df = pd.read_csv('data/PreferenciasBritanicos.csv')

def pre_analysis(data):
    fig,ax=plt.subplots(edgecolor='k')
    w=0.15
    a=np.arange(2)
    i = 0
    
    total1 = data[data['Nacionalidad'] == 'E'].shape[0]
    total2 = data[data['Nacionalidad'] == 'I'].shape[0]
    totals = [total1, total2]

    for attr in data.columns:
        if attr == 'Nacionalidad':
            continue

        # get the count of rows that match attr based on the class
        count = data.groupby('Nacionalidad')[attr].sum()

        # calculate the probability of attr given class
        p_attr_per_class = count / totals

        # plot the probability
        ax.bar(a + i*w, p_attr_per_class, width=w, label=attr)
        i += 1

    plt.xticks(a + 2*w, ['E', 'I'])
    plt.xlabel('Nacionalidad')
    plt.ylabel('Probabilidad')
    plt.legend()
    plt.show()


def cross_validation(data, k = 5):
    data = data.sample(frac=1).reset_index(drop=True)
    block = data.shape[0] // k
    best_block = -1
    best_block_accuracy = 0
    for i in range(k):
        test = data[i*block:(i+1)*block]
        train = data.drop(test.index)

        correct = 0
        for _, row in test.iterrows():
            max_class, p_class = predict(row, train, 'Nacionalidad', laplace=True)
            if max_class == row['Nacionalidad']:
                correct += 1

        if correct / block > best_block_accuracy:
            best_block_accuracy = correct / block
            best_block = train

    return best_block, best_block_accuracy

def best_block_size(data):
    block_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    aux = []
    for i in range(1, len(block_sizes)):
        if data.shape[0] // block_sizes[i] != data.shape[0] // block_sizes[i-1] and data.shape[0] // block_sizes[i] > 1:
            aux.append(block_sizes[i])
    block_sizes = aux

    samples = 10
    avg_accuracy = []
    svd_accuracy = []

    for block_size in block_sizes:
        accuracies = []
        for s in range(samples):
            _, accuracy = cross_validation(data, k=block_size)
            accuracies.append(accuracy)
        avg_accuracy.append(np.mean(accuracies))
        svd_accuracy.append(np.std(accuracies))

    # set y limits for graph
    plt.errorbar(block_sizes, avg_accuracy, yerr=svd_accuracy, fmt='o')
    plt.xlabel('Tamaño de bloque (k)')
    plt.ylabel('Precisión')
    plt.ylim(0, 1)
    plt.show()

x1 = {
    'scones': 1,
    'cerveza': 0,
    'wiskey': 1,
    'avena': 1,
    'futbol': 0,
}

x2 = {
    'scones': 0,
    'cerveza': 1,
    'wiskey': 1,
    'avena': 0,
    'futbol': 1
}

pre_analysis(df)
# best_block_size(df)

# train, _ = cross_validation(df, k=5)

train = df

max_prob, mult_prob = predict(x1, train, 'Nacionalidad', laplace=False)
print('For the values {}\n the class with the greatest probability is: {} with a probability of {}%'.format(x1, max_prob, round(mult_prob[max_prob]*100, 2)))

max_prob, mult_prob = predict(x2, train, 'Nacionalidad', laplace=False)
print('For the values {}\n the class with the greatest probability is: {} with a probability of {}%'.format(x2, max_prob, round(mult_prob[max_prob]*100, 2)))