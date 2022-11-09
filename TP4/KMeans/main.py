import pandas as pd
from KMeans import KMeans
import matplotlib.pyplot as plt

def variation(classes):
    class_variation = {}
    for c in range(len(classes)):
        sum = 0
        for i in range(len(classes[c])):
            for j in range(i+1, len(classes[c])):
                row1 = classes[c][i][:-1]
                row2 = classes[c][j][:-1]
                for p in range(len(row1)):
                    sum += (row1[p] - row2[p])**2
        sum *= 1/len(classes[c])
        print("Class", c, "variation:", sum)

        class_variation[c] = sum

    sum = 0
    for c in class_variation:
        sum += class_variation[c]
    return sum


def main(columns=None, id_column='imdb_id', k=3, max_iter=100, write_to_file=False):
    df_proc = pd.read_csv('../data/movie_data_proc.csv')
    df = pd.read_csv('../data/movie_data.csv', sep=';')
    if columns is None:
        columns = df_proc.columns.drop(id_column).to_list()
    columns.append(id_column)
    df_proc = df_proc[columns]

    X = df_proc.to_numpy()
    kmeans = KMeans(k=k, max_iter=max_iter)
    kmeans.fit(X)

    centroid = kmeans.centroids
    classes = kmeans.classes

    if write_to_file:
        res_file = open('../data/results.csv', 'w')
        res_file.write('class,'+','.join(df.columns) + '\n')
        for i in range(len(classes)):
            for row in classes[i]:
                res_file.write(str(i)+','+df[df['imdb_id'] == row[-1]].to_csv(header=None, index=False).split('\n')[0] + '\n')

        res_file.close()

    print("Calculating variation...")
    print(variation(classes))

main(k=20, write_to_file=True)
