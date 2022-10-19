import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


def add_to_list(pixels, file, label):
    new_pixels = list(Image.open(file).getdata())
    for np in new_pixels:
        pixels.append((np[0], np[1], np[2], label))


def create_dataset():
    df = pd.DataFrame(columns=['R', 'G', 'B', 'label'])

    pixels = []
    add_to_list(pixels, 'data/cielo.jpg', 'cielo')
    add_to_list(pixels, 'data/pasto.jpg', 'pasto')
    add_to_list(pixels, 'data/vaca.jpg', 'vaca')

    df = pd.DataFrame(pixels, columns =['R', 'G', 'B', 'label'])
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv('data\dataset.csv', index=False)


def scatter(df, label, color):
    _df = df[df['label'] == label]
    ax.scatter(_df['R'].to_list(), _df['G'].to_list(), _df['B'].to_list(), c=color, label=label)

# def plot_dataset():
df = pd.read_csv('data/dataset.csv')
df = df.sample(n = 10000)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')

scatter(df, 'cielo', 'blue')
scatter(df, 'pasto', 'green')
scatter(df, 'vaca', 'red')

plt.legend()
plt.show()

# create_dataset()