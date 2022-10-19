import numpy as np
import matplotlib.pyplot as plt

def plot_dataset(input='data.csv', file_name='points'):
    data = np.genfromtxt(input, delimiter=',', skip_header=1)
    class1 = data[data[:, 2] == 1]
    class2 = data[data[:, 2] == -1]
    plt.scatter(class1[:, 0], class1[:, 1], marker='o', c='red', label='1')
    plt.scatter(class2[:, 0], class2[:, 1], marker='x', c='blue', label='-1')
    plt.legend()
    x = np.linspace(0, 5, 100)
    y = 5 - x
    plt.plot(x, y, c='black')
    plt.savefig('out/'+file_name+'.png')

def create_dataset(size=50, max=5, dist=1, file_name='data', bad_points=0):
    f = open(file_name+'.csv', 'w')
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
            elif bad_points > 0 and v > max-dist and v < max+dist:
                if v > max:
                    f.write(f'{data[i,0]},{data[i,1]},{-1}\n')
                else:
                    f.write(f'{data[i,0]},{data[i,1]},{1}\n')
                actual_size += 1
                bad_points -= 1
            
            if actual_size == size + bad_points:
                f.close()
                plot_dataset(input=file_name+'.csv', file_name=file_name)
                return




def plot_results(data, file_name='results'):
    errors = data[data['prediction'] != data['class']]
    class1 = data[(data['prediction'] == 1) & (data['class'] == 1)]
    class2 = data[(data['prediction'] == -1) & (data['class'] == -1)]
    plt.scatter(class1['x'], class1['y'], marker='o', c='red', label='1')
    plt.scatter(class2['x'], class2['y'], marker='x', c='blue', label='-1')
    plt.scatter(errors['x'], errors['y'], marker='1', c='green', label='wrong')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
    plt.savefig('out/'+file_name+'.png')
    plt.show()