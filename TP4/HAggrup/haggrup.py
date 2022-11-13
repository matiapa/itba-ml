from __future__ import annotations
import numpy as np
from tqdm import tqdm

class HAggrup:

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
                g1, g2 = groups[i], groups[j]
                dists.append((g1, g2, g1.dist(g2)))

        groups = set(groups)

        # Iterate until only one group is left

        while len(groups) > 1:
            print(len(groups))

            # Store current aggrupation

            self.aggrupations.append(groups.copy())
            
            # Get the nearest groups

            g1, g2, j = min(dists, key = lambda d: d[2])

            # Remove those groups

            # print('Groups')
            # print(groups)

            # print('Dists')
            # for dist in dists:
            #     print(dist)

            # print('--------------------')

            groups.remove(g1)
            groups.remove(g2)
            
            dists = [d for d in dists if d[0] not in [g1,g2] and d[1] not in [g1,g2]]

            # Add the new joined group

            new_group = g1.join(g2)
            
            for g in groups:
                dists.append((new_group, g, new_group.dist(g)))
           
            groups.add(new_group)
    
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