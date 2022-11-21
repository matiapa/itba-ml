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

        min_dist = float('inf')
        g1, g2 = None, None
        
        for i in tqdm(range(0, len(X))):
            for j in range(i+1, len(X)):
                dist = groups[i].dist(groups[j])
                if dist < min_dist:
                    min_dist = dist
                    g1, g2 = groups[i], groups[j]

        # Iterate until only one group is left

        while len(groups) > 1:
            print(len(groups))

            # print('Groups')
            # print(groups)

            # print(f'Min dist - {g1p}-{g2p}: {min_dist}')

            # print('--------------------')

            # Store current aggrupation

            self.aggrupations.append(groups.copy())

            # Remove the nearest groups

            print(groups)
            print(g1)
            print(g2)

            groups.remove(g1)
            groups.remove(g2)

            # Add the new joined group

            new_group = g1.join(g2)

            groups.append(new_group)

            # Update the minimum distance

            for i in range(len(groups) - 1):
                dist = new_group.dist(groups[i])
                if dist < min_dist:
                    min_dist = dist
                    g1, g2 = new_group, groups[i]
    
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