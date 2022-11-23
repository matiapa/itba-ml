import numpy as np

# implementation of Hierarchical grouping algorithm
class HGroup:
    def __init__(self):
        self.levels = []
        self.distance = self.average_distance
        

    def fit(self, X):
        # create a group for each row in X
        groups = []
        groups_ids = []
        for row in X:
            groups.append([row])
            groups_ids.append([row[-1]])

        # file = open('groups.csv', 'w')
        # file.write('level,size,elements\n')

        # create a level for each group
        self.levels.append(groups_ids)

        num = 0

        distances = np.zeros((len(groups), len(groups)))
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                distances[i][j] = self.distance(groups[i], groups[j])
                distances[j][i] = distances[i][j]

        # while there is more than one group
        while len(groups) > 1:
            if num % 10 == 0:
                print("Level", num)

            min_coords = self.min_distance_coordinates(distances)

            x = min(min_coords)
            y = max(min_coords)
            groups[x] = groups[x] + groups[y]
            groups_ids[x] = groups_ids[x] + groups_ids[y]
            groups.pop(y)
            groups_ids.pop(y)
            distances = np.delete(distances, y, 0)
            distances = np.delete(distances, y, 1)

            # print('\n'.join([' '.join([str(round(item, 3)) for item in row]) for row in distances]))

            # update the distances of the group i
            for k in range(len(groups)):
                if k != x:
                    distances[x][k] = self.distance(groups[x], groups[k])
                    distances[k][x] = distances[x][k]

            
            # print('\n'.join([' '.join([str(round(item, 3)) for item in row]) for row in distances]))
            self.levels.append(groups_ids.copy())

            # file.write("{},{},{}\n".format(num, len(groups_ids), groups_ids[min_distance[0]]))

            num+=1

        # file.close()

    def min_distance_coordinates(self, distances):
        min_distance = float('inf')
        min_distance_coordinates = None
        for i in range(len(distances)):
            for j in range(i+1, len(distances)):
                if distances[i][j] < min_distance:
                    min_distance = distances[i][j]
                    min_distance_coordinates = [i, j]
        return min_distance_coordinates

    def euclidian_distance(self, x, y):
        print(x, y)
        return np.linalg.norm(np.array(x[:-1]) - np.array(y[:-1]))

    def average_distance(self, X, Y):
        return np.average([self.euclidian_distance(x, y) for x in X for y in Y])

