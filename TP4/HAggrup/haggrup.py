from __future__ import annotations
import numpy as np
from tqdm import tqdm

class HAggrup:

    #   A B C
    # A 0 1 2
    # B 1 0 1
    # C 2 1 0

    def __init__(self):
        self.aggrupations = []

    def fit(self, X):
        # Create a group for each sample
        
        groups = [Group(X=[x]) for x in X]

        # Calculate the initial group distances

        print('Calculating initial distances...')

        dists = []
        for i in tqdm(range(0, len(X))):
            for j in range(i+1, len(X)):
                dist = groups[i].dist(groups[j])
                dists.append((groups[i], groups[j], dist))

        # Iterate until only one group is left

        while len(groups) > 1:
            print(len(groups))

            # Store current aggrupation

            self.aggrupations.append(groups.copy())

            # Find the min group distance

            g1, g2, _ = min(dists, key = lambda d : d[2])

            # print(groups)
            # print(g1)
            # print(g2)
            # print('-------------')

            # Remove the groups and their associated distances

            groups = [g for g in groups if g!=g1 and g!=g2]

            dists = [d for d in dists if d[0] not in [g1,g2] and d[1] not in [g1,g2]]

            # Create a new group by joining them

            new_group = g1.join(g2)

            # Add the new group and its distances

            groups.append(new_group)

            for i in range(len(groups) - 1):
                dist = new_group.dist(groups[i])
                dists.append((new_group, groups[i], dist))
    
        self.aggrupations.append(groups)


class Group:

    def __init__(self, X) -> None:
        self.X = X

    def dist(self, g: Group) -> float:
        return np.average([np.linalg.norm(x1-x2) for x1, x2 in zip(self.X, g.X)])

    def join(self, g: Group) -> Group:
        return Group(X = self.X + g.X)

    def __str__(self) -> str:
        return str(self.X)

    def __repr__(self):
       return self.__str__()