import numpy as np
import matplotlib.pyplot as plt


def create_dataset(size=50, max=0.5, dist=0.025):
    f = open('data.csv', 'w')
    f.write('x,y,class\n')

    actual_size = 0
    while True:
        data = np.random.rand(size, 2) * max
        for i in range(size):
            v = data[i,0] + data[i,1]

            if v > max+dist:
                f.write(f'{data[i,0]},{data[i,1]},{1}\n')
                actual_size += 1
            elif v < max-dist:
                f.write(f'{data[i,0]},{data[i,1]},{-1}\n')
                actual_size += 1
            
            if actual_size == size:
                f.close()
                return


def plot_dataset():
    data = np.genfromtxt('data.csv', delimiter=',', skip_header=1)
    class1 = data[data[:, 2] == 1]
    class2 = data[data[:, 2] == -1]
    plt.scatter(class1[:, 0], class1[:, 1], marker='o', c='red', label='1')
    plt.scatter(class2[:, 0], class2[:, 1], marker='x', c='blue', label='-1')
    plt.legend()
    plt.savefig('points.png')


def plot_results(data):
    errors = data[data['prediction'] != data['class']]
    class1 = data[(data['prediction'] == 1) & (data['class'] == 1)]
    class2 = data[(data['prediction'] == -1) & (data['class'] == -1)]
    plt.scatter(class1['x'], class1['y'], marker='o', c='red', label='1')
    plt.scatter(class2['x'], class2['y'], marker='x', c='blue', label='-1')
    plt.scatter(errors['x'], errors['y'], marker='1', c='green', label='wrong')

    x = np.linspace(0, 0.5, 100)
    y = 0.5 - x
    plt.plot(x, y, c='black')
    plt.legend()
    plt.show()


# create_dataset(size=100)
# plot_dataset()