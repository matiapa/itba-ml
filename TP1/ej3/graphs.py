import pandas
import matplotlib.pyplot as plt
import numpy as np


df = pandas.read_csv('../data/binary.csv')

def p(var_pred, cond_pred, class_count):
    return (len(df[(var_pred) & (cond_pred)]) + 1) / (len(df[cond_pred]) + class_count)

def pre_analysis(data):
    lab1 = ['GPA >= 3', 'GRE >= 500', 'Rank >= 2']
    x_pos1 = np.arange(len(lab1))

    prob1 = [
        p(data['admit'] == 1, data['gpa'] >= 3, 2),
        p(data['admit'] == 1, data['gre'] >= 500, 2),
        p(data['admit'] == 1, data['rank'] >= 2, 2),
    ]

    plt.bar(x_pos1, prob1, align='center')
    plt.xticks(x_pos1, lab1)

    plt.xlabel('Condition')
    plt.ylabel('P(A=1 | ...)')

    plt.legend()
    plt.show()

pre_analysis(df)