import numpy as np


class KMeans:
    def __init__(self, k=3, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.centroids = {}
        self.classes = {}


    def fit(self, X):
        for i in range(self.k):
            self.centroids[i] = np.array(X[i][:-1])

        for i in range(self.max_iter):
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = []

            for features in X:
                # remove the first element of the features array, which is the imdb_id
                aux = np.array(features[:-1])
                distances = [np.linalg.norm(aux - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)

            prev_centroids = dict(self.centroids)

            for classification in self.classes:
                # create a numpy array with the first element of each row of the classes[classification] array
                # this is the vote_average column
                aux = np.array([np.array(row[:-1]) for row in self.classes[classification]])
                self.centroids[classification] = np.average(aux, axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.linalg.norm(current_centroid - original_centroid) > 0.0001:
                    optimized = False

            if optimized:
                break
