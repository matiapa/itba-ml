from math import hypot
from random import random
from typing import Tuple
import numpy as np
from neuron import Neuron

class Kohonen:

    def __init__(self, k: int, r0: int, n0: float):
        self.network = [[None for _ in range(k)] for _ in range(k)]
        self.r0 = r0
        self.n0 = n0
        self.k = k

    def train(self, inputs: list[list[float]]) -> None:

        # Fill initial weights

        for y in range(self.k):
            for x in range(self.k):
                # rand_input = inputs[int(random()*(len(inputs)-1))]
                rand_input = inputs[y*x % len(inputs)]
                self.network[y][x] = Neuron(weights=list(rand_input))
        
        T = 500 * len(inputs[0])

        # Start iteration

        t = 0
        r = self.r0
        n = self.n0
        while t < T:
            # print(f'Iteration {t}/{T}')

            # i = int(random()*(len(inputs)-1))
            i = t % len(inputs)
            rand_input = inputs[i]

            # Find the best neuron

            bx, by = self.get_coords(rand_input)
            
            # Update neighbours weights

            for x in range(bx-r, bx+r+1, 1):
                for y in range(by-r, by+r+1, 1):
                    if x>=0 and x<self.k and y>=0 and y<self.k and hypot(x-bx, y-by) <= r:
                        self.network[x][y].weights += n * (rand_input - self.network[x][y].weights)
            
            # Update n, r and t

            t += 1
            n = (0.7-self.n0)/T * t + self.n0
            r = int((1-self.r0)/T * t) + self.r0
        
    def get_coords(self, input: list[float]) -> Tuple[int, int]:
        best_dist, best_coords = np.Infinity, None
        for x in range(self.k):
            for y in range(self.k):
                distance = self.network[x][y].distance(input)
                if distance < best_dist:
                    best_dist = distance
                    best_coords = [x,y]
            # print(f"({x},{y})={self.network[x][y].weights}")
        return best_coords