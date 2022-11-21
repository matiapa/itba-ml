from typing import Tuple
import numpy as np
import pandas
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from network import Kohonen
from scipy import stats


# Read the data

df = pandas.read_csv('../data/movie_data_proc.csv')
genres = df["genres"].to_numpy()

df.drop("imdb_id", axis=1, inplace=True)
df.drop("genres", axis=1, inplace=True)

# Explore hyperparameters

def avg_neur_dist(k, r0, n0) -> float:

    kohonen = Kohonen(k, r0, n0)
    kohonen.train(df.values)

    avg_distances = np.zeros((k, k))
    for x in range(k):
        for y in range(k):
            own_weights = kohonen.network[y][x].weights
            avg_neigh_dist = 0

            valid_neighs = 0
            for nx in range(x-1, x+2):
                for ny in range(y-1, y+2):
                    if nx>0 and nx<k and ny>0 and ny<k:
                        avg_neigh_dist += kohonen.network[ny][nx].distance(own_weights)
                        valid_neighs += 1
        
            avg_distances[y][x] = avg_neigh_dist / valid_neighs
    

    return sum(avg_distances[y][x] for y in range(k) for x in range(k))/k**2


def genre_accuracy(k, r0, n0) -> float:
    neur_genres = [[[] for _ in range(k*k)] for _ in range(k*k)]

    # Train and run the network

    kohonen = Kohonen(k, r0, n0)
    kohonen.train(df.values)

    for i in range(len(df.values)):
        x, y = kohonen.get_coords(df.values[i])
        neur_genres[x][y].append(genres[i])

    # Calculate the neurons genre modes

    heatmap = np.zeros((k, k))
    genre_code = {'None': 0, 'Action': 1, 'Comedy': 2, 'Drama': 3}
    for x in range(k):
        for y in range(k):
            genre_mode = stats.mode(neur_genres[x][y]).mode
            genre_mode = genre_mode[0] if len(genre_mode) > 0 else 'None'
            neur_genres[x][y] = genre_mode
            heatmap[x,y] =genre_code[genre_mode]

    sns.heatmap(data=heatmap)
    plt.show()

    # Calculate the accuracies
    
    corrects = {'Action': 0, 'Comedy': 0, 'Drama': 0}
    totals = {'Action': 0, 'Comedy': 0, 'Drama': 0}

    for i in range(len(df.values)):
        totals[genres[i]] += 1
        x, y = kohonen.get_coords(df.values[i])

        if genres[i] == neur_genres[x][y]:
            corrects[genres[i]] += 1
    
    for genre in ['Action', 'Comedy', 'Drama']:
        print(f'{genre} accuracy: {round(corrects[genre] / totals[genre] * 100, 2)}%')
    print(f'General accuracy: {round(sum(corrects.values()) / sum(totals.values()) * 100, 2)}%')

    return sum(corrects.values()) / sum(totals.values())


def explore_hyper():

    for k in [5, 10, 20]:
        df = pandas.DataFrame(columns={'r0','n0','value'})

        for r0 in [1, k//2, k]:
            for n0 in [0.1, 0.5, 1]:
                # value = avg_neur_dist(k, r0, n0)
                value = genre_accuracy(k, r0, n0)
                df = df.append({'r0': r0, 'n0': n0, 'value': value}, ignore_index=True)
                print(f'k={k}, r0={r0}, n0={n0} | value = {value}')

        sns.heatmap(df.pivot('r0', 'n0', 'value'), annot=True)

        plt.show()

# explore_hyper()

genre_accuracy(k=20, r0=1, n0=1)


# Show the results

def count_plot(k, r0, n0):
    kohonen = Kohonen(k, r0, n0)
    kohonen.train(df.values)

    heatmap = np.zeros((k, k))
    for input in df.values:
        x, y = kohonen.get_coords(input)
        heatmap[y][x] += 1
    plt.imshow(heatmap)
    plt.colorbar()
    plt.show()


def var_avg_plot(k, r0, n0):
    kohonen = Kohonen(k, r0, n0)
    kohonen.train(df.values)
    numeric_columns = ["budget","production_countries","revenue","runtime","spoken_languages","vote_average","vote_count"]

    count_matrix = np.ones((k, k))
    vars_avg_matrix = np.zeros((len(numeric_columns), k, k))

    # Sum the values of variables
    for input in df.values:
        x, y = kohonen.get_coords(input)
        count_matrix[y][x] += 1
        for v in range(len(numeric_columns)):
            vars_avg_matrix[v][y][x] += kohonen.network[y][x].weights[v]

    # Divide by the cell count
    for v in range(len(numeric_columns)):
        vars_avg_matrix[v] /= count_matrix

    # Display all matrixes as heatmaps
    _, axes = plt.subplots(2,5)
    for v in range(len(numeric_columns)):
        axes[v//5][v%5].set_title(numeric_columns[v])
        axes[v//5][v%5].imshow(vars_avg_matrix[v], cmap=cm.gray)
    plt.show()


def u_matrix_plot(k, r0, n0):
    kohonen = Kohonen(k, r0, n0)
    kohonen.train(df.values)

    heatmap = np.zeros((k, k))
    for x in range(k):
        for y in range(k):
            own_weights = kohonen.network[y][x].weights
            avg_neigh_dist = 0

            valid_neighs = 0
            for nx in range(x-1, x+2):
                for ny in range(y-1, y+2):
                    if nx>0 and nx<k and ny>0 and ny<k:
                        avg_neigh_dist += kohonen.network[ny][nx].distance(own_weights)
                        valid_neighs += 1
        
            heatmap[y][x] = avg_neigh_dist / valid_neighs
    print(f"Avg {sum(heatmap[y][x] for y in range(k) for x in range(k))/k**2}")

    plt.imshow(heatmap, cmap=cm.gray)
    plt.colorbar()
    plt.show()

# count_plot(k=20, r0=20, n0=0.1)
# u_matrix_plot(k=20, r0=20, n0=0.1)
# var_avg_plot(k=20, r0=20, n0=0.1)