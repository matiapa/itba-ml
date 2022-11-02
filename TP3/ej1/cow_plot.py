import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from SVM import SVM


# Read and split the data

print('Reading file...')

df = pd.read_csv('../ej2/data/dataset.csv')

class_num = {'vaca': -1, 'cielo': 0, 'pasto': 1}
df["label"] = df["label"].apply(lambda l : class_num[l])

X_train, X_test, y_train, y_test = train_test_split(df.drop(["label"], axis=1), df["label"], test_size=0.3)


# def full_image_plot(img_path):

img = Image.open('../ej2/data/imagen3.jpg')

w, h = img.size

# Train with best parameters

print('Training...')

svm = SVM(C=1, input_size=2, b=0, Aw=0.1, Ab=0.1, k=0.01)

svm.train(X_train.to_numpy(), y_train.to_numpy(), iter=500)

# Classify the whole image

print('Evaluating...')

X_test = list(img.getdata())

y_pred = [svm.evaluate(r) for r in X_test]

# Plot an image of the labels

print('Plotting...')

pixels = np.zeros((h, w, 3))

colors = {-1: [102/255, 57/255, 30/255], 0: [66/255, 181/255, 201/255], 1: [106/255, 161/255, 39/255]}

for i in range(len(y_pred)):
    label = y_pred[i]
    pixels[i // w, i % w] = colors[label]

plt.imshow(pixels)

plt.show()

# full_image_plot('../ej2/data/imagen3.jpg')