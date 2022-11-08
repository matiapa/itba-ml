import pandas as pd
from KMeans import KMeans
import matplotlib.pyplot as plt

def main(columns=None, id_column='imdb_id', k=3, max_iter=100):
    df_proc = pd.read_csv('data/movie_data_proc.csv')
    df = pd.read_csv('data/movie_data.csv', sep=';')
    if columns is None:
        columns = df_proc.columns.drop(id_column).to_list()
    columns.append(id_column)
    df_proc = df_proc[columns]

    X = df_proc.to_numpy()
    kmeans = KMeans(k=k, max_iter=max_iter)
    kmeans.fit(X)

    centroid = kmeans.centroids
    classes = kmeans.classes

    res_file = open('data/results.csv', 'w')
    res_file.write('class,'+','.join(df.columns) + '\n')
    for i in range(len(classes)):
        for row in classes[i]:
            res_file.write(str(i)+','+df[df['imdb_id'] == row[-1]].to_csv(header=None, index=False).split('\n')[0] + '\n')

    res_file.close()


    # for classification in classes:
    #     color = 'r'
    #     if classification == 0:
    #         color = 'b'
    #     elif classification == 1:
    #         color = 'g'
    #     elif classification == 2:
    #         color = 'y'
    #     for features in classes[classification]:
    #         plt.scatter(features[0], features[1], color=color, s=10)

    # for c in centroid:
    #     plt.scatter(centroid[c][0], centroid[c][1], color='k', marker='x', s=100)

    # plt.show()


main(k=3)
# clean_data()




    



