import numpy as np


class KMeans:
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = {}
        self.classes = {}


    def fit(self, X, stop=10e-6):
        random_indexes = np.random.choice(len(X), self.k, replace=False)
        index = 0
        # calculate initial centroids
        for i in random_indexes:
            self.centroids[index] = np.array(X[i][:-1])
            index += 1

        for i in range(self.max_iter):
            # clean classes
            self.classes = {}
            for r in range(self.k):
                self.classes[r] = []

            # assign each row to a class based the nearest euclidian distance
            for features in X:
                # remove the first element of the features array, which is the imdb_id
                aux = np.array(features[:-1])
                distances = [np.linalg.norm(aux - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)

            # calculate new centroids
            prev_centroids = dict(self.centroids)
            for classification in self.classes:
                # remove imdb_id from the features
                aux = np.array([np.array(row[:-1]) for row in self.classes[classification]])
                # caculate the mean of the features
                self.centroids[classification] = np.average(aux, axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.linalg.norm(current_centroid - original_centroid) > stop:
                    optimized = False

            if optimized:
                break
