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
        # print("Class", c, "variation:", sum)

        class_variation[c] = sum

    sum = 0
    for c in class_variation:
        sum += class_variation[c]

    return sum

def k_similarity_analysis():
    id_column = 'imdb_id'
    df_proc = pd.read_csv('../data/movie_data_proc.csv')
    df = pd.read_csv('../data/movie_data.csv')
    columns = df_proc.columns.drop(id_column).to_list()
    columns.append(id_column)
    df_proc = df_proc[columns]
    X = df_proc.to_numpy()

    ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    similarities = []
    cant = 10
    for k in ks:
        print("-------- K:", k)
        kmeans = KMeans(k=k, max_iter=100)
        aux = []
        for i in range(cant):
            kmeans.fit(X)
            centroid = kmeans.centroids
            classes = kmeans.classes

            similarity = variation(classes)/len(classes)
            aux.append(similarity)
            print("Similarity:", similarity)
        # get the min aux value
        print("k =", k, "similarity:", min(aux))
        similarities.append(min(aux))

    print("Similarities:", similarities)
    plt.plot(ks, similarities, marker='o')
    plt.xlabel('k')
    plt.ylabel('Similitud')
    plt.show()

def accuracy(classes, df):
    correct = 0
    total = 0
    for c in classes:
        genres = []
        for row in classes[c]:
            id = row[-1]
            genres.append(df[df['imdb_id'] == id]['genres'].to_list()[0])
        mode = max(set(genres), key=genres.count)
        correct += genres.count(mode)
        total += len(genres)

    return correct/total



def k_accuracy_analysis():
    id_column = 'imdb_id'
    df_proc = pd.read_csv('../data/movie_data_proc.csv')
    df = pd.read_csv('../data/movie_data.csv')
    columns = df_proc.columns.drop(id_column).to_list()
    columns.append(id_column)
    df_proc = df_proc[columns]
    X = df_proc.to_numpy()

    ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    accuracies = []
    cant = 10
    for k in ks:
        print("-------- K:", k)
        kmeans = KMeans(k=k, max_iter=100)
        aux = []
        for i in range(cant):
            kmeans.fit(X)
            centroid = kmeans.centroids
            classes = kmeans.classes
            aux.append(accuracy(classes, df))
            print("Accuracy:", aux[-1])
        print("k =", k, "accuracy:", max(aux))
        accuracies.append(max(aux))

    print("Accuracies:", accuracies)
    plt.plot(ks, accuracies, marker='o')
    plt.xlabel('k')
    plt.ylabel('Accuracy')
    plt.show()



def main(columns=None, id_column='imdb_id', k=3, max_iter=100, cant=15, type='variation', write_to_file=False):
    if type not in ['variation', 'accuracy']:
        print("Invalid type")
        return

    df_proc = pd.read_csv('../data/movie_data_proc.csv')
    df = pd.read_csv('../data/movie_data.csv')
    if columns is None:
        columns = df_proc.columns.drop(id_column).to_list()
    columns.append(id_column)
    df_proc = df_proc[columns]
    X = df_proc.to_numpy()

    kmeans = KMeans(k=k, max_iter=max_iter)

    min_variation = None
    min_accuracy = None
    min_centroids = None
    min_classes = None
    for i in range(cant):
        kmeans.fit(X)
        centroid = kmeans.centroids
        classes = kmeans.classes

        if type == 'variation':
            var = variation(classes)
            if (min_variation is None) or (var < min_variation):
                min_accuracy = accuracy(classes, df)
                min_variation = var
                min_centroids = centroid
                min_classes = classes
        else:
            acc = accuracy(classes, df)
            if (min_accuracy is None) or (acc > min_accuracy):
                min_variation = variation(classes)
                min_accuracy = acc
                min_centroids = centroid
                min_classes = classes

    print("Accuracy:", min_accuracy)
    print("Variation:", min_variation)

    if write_to_file:
        res_file = open('../data/results_'+type+'.csv', 'w')
        res_file.write('class,'+','.join(df.columns) + '\n')
        for i in range(len(classes)):
            for row in min_classes[i]:
                res_file.write(str(i)+','+df[df['imdb_id'] == row[-1]].to_csv(header=None, index=False).split('\n')[0] + '\n')
        res_file.close()

columns=['budget','production_countries','revenue','runtime','spoken_languages','vote_average','vote_count']
main(k=5, max_iter=1000, type='accuracy', columns=columns, write_to_file=True)
# k_similarity_analysis()
# k_accuracy_analysis()

