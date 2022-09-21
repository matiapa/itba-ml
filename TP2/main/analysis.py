import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from random_forest import random_forest
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def sample_size_analysis():
    df = pd.read_csv('../data/german_credit_proc.csv', dtype=object)
    test_frac = 0.1
    samples = 10
    sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    avg_acc = []
    std_acc = []

    for size in sizes:
        print("Size: " + str(size))
        acc = []
        for _ in range(samples):
            acc.append(random_forest(df, int(len(df) * size), test_frac, max_depth=5, min_samples=0, n_trees=70))
        avg_acc.append(np.mean(acc))
        std_acc.append(np.std(acc))

    with open('results.txt', 'w') as f:
        for i in range(len(sizes)):
            f.write(str(sizes[i]) + ' ' + str(avg_acc[i]) + ' ' + str(std_acc[i]) + '\n')

    plt.errorbar(sizes, avg_acc, yerr=std_acc, fmt='o')


def random_forest_analysis(df):
    df = df.sample(frac=1).reset_index(drop=True)

    train_set = df.iloc[0:900]
    test_set = df.iloc[900:1000]
    trees = random_forest(train_set, 700, 0.05, max_depth=8, min_samples=100, n_trees=64)
    print("Finished training")

    matches = 0
    i = 0
    not_found = 0

    print("Testing...")
    for index, row in test_set.iterrows():
        print(i)
        results = {}
        for tree in trees:
            result = tree.evaluate(row)
            if result not in results:
                results[str(result)] = 0

            results[str(result)] += 1

        obtained = max(results, key=results.get)

        if str(obtained) == '?':
            not_found += 1

        if str(obtained) == str(row['Creditability']):
            matches += 1

        print("The majority vote is: " + str(obtained) + " expected: " + str(row['Creditability']) + " - 1: " + str(
            results.get('1')) + " - 0: " + str(results.get('0')) + " - ?: " + str(results.get('?')))
        i += 1

    print("Matched: " + str(matches) + " not found " + str(not_found))
    print("Precision of the random forest: " + str(matches / len(test_set)))


# df = pd.read_csv('../data/german_credit_proc.csv', dtype=object)
# random_forest_analysis(df)


def variables_analysis():
    df = pd.read_csv('../data/german_credit.csv')
    #     divide df for column 'Credit Amount' in 4 groups
    df['Credit Amount'] = pd.qcut(df['Credit Amount'], 4, labels=[0, 1, 2, 3])

    fig, ax = plt.subplots()
    ax.violinplot(df[df['Creditability' == 1]]['Credit Amount'], vert=False)
    ax.violinplot(df[df['Creditability' == 0]]['Credit Amount'], vert=False)
    plt.show()


variables_analysis()


def violin_analysis():
    df = pd.read_csv('../data/german_credit_proc.csv')

    columns = df.columns.values.tolist()
    datas = {}
    data_columns = {}

    for i in range(len(columns)):
        if columns[i] == 'Creditability':
            continue

        max_val = df[columns[i]].max()

        if max_val not in datas:
            datas[max_val] = {}
            datas[max_val][0] = []
            datas[max_val][1] = []
            data_columns[max_val] = []

        datas[max_val][0].append(df[df['Creditability'] == 0][columns[i]])
        datas[max_val][1].append(df[df['Creditability'] == 1][columns[i]])
        data_columns[max_val].append(columns[i])

    for key in datas:
        labels = []
        fig, ax = plt.subplots()
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_yticklabels(data_columns[key])
        pos = [i + 1 for i in range(len(data_columns[key]))]

        violin = ax.violinplot(datas[key][0], pos, vert=False)
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), 'Creditability = 0'))

        violin = ax.violinplot(datas[key][1], pos, vert=False)
        color = violin["bodies"][0].get_facecolor().flatten()
        labels.append((mpatches.Patch(color=color), 'Creditability = 1'))

        plt.yticks(pos, data_columns[key])
        plt.legend(*zip(*labels), loc=2)
        plt.tight_layout()
        plt.show()


def parallel_analysis():
    df = pd.read_csv('../data/german_credit_proc.csv')

    # columns = df.columns.values.tolist()
    # columns.remove('Creditability')
    # fig, ax = plt.subplots(1, len(columns))
    #
    # i = 0
    # for row in df.iterrows():
    #     # remove the creditability column
    #     row = row[1].tolist()
    #     row.pop(0)
    #     print(columns)
    #     ax[i].plot(columns, row)
    #     # ax[i].set_title(column)
    #     i += 1
    #     if i == 5:
    #         break
    #
    # plt.subplots_adjust(wspace=0)
    # plt.show()

    # create a dictionary with the columns name
    columns = df.columns.values.tolist()
    columns.remove('Creditability')

    # fig = px.parallel_coordinates(df, color="Creditability", labels=labels,
    #                               color_continuous_scale=px.colors.diverging.Tealrose,
    #                               color_continuous_midpoint=2)

    # fig = px.parallel_coordinates(df, color="Creditability",
    #                               dimensions=columns[:5],
    #                               color_continuous_scale=px.colors.diverging.Tealrose,
    #                               color_continuous_midpoint=2)

    dimensions = []
    for column in columns:
        # get max and min value of the column
        max_val = df[column].max()
        min_val = df[column].min()
        dimensions.append(
            dict(range=[min_val, max_val], constraintrange=[min_val, max_val], label=column, values=df[column]))

    fig = go.Figure(data=
    go.Parcoords(
        line=dict(color=df['Creditability'],
                  colorscale=[[0, 'purple'], [1, 'gold']]),
        dimensions=dimensions
    )
    )

    fig.show()

# violin_analysis()
# test()
# parallel_analysis()
