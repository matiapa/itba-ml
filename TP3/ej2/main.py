import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Read and split the data

print('Reading file...')

df = pd.read_csv('data/dataset.csv')

X_train, X_test, y_train, y_test = train_test_split(df.drop(["label"], axis=1), df["label"], test_size=0.3)


def accuracy_vs_c_k():
    # Define hyperparameters

    c_vals = [0.1, 1, 10]
    kernels = ['poly', 'rbf', 'linear']
    labels = ['vaca', 'pasto', 'cielo']

    for k in kernels:
        for c in c_vals:
            # Train the SVM model

            print(f'Training with C={c}, K={k}...')

            clf = svm.SVC(kernel=k, C=c)

            clf.fit(X_train, y_train)

            # Evaluate it

            print('Evaluating...')

            y_pred = clf.predict(X_test)

            conf_mat = metrics.confusion_matrix(y_test, y_pred, labels=labels)

            accuracy = metrics.accuracy_score(y_test, y_pred)

            print(f'Accuracy: {accuracy}')

            sns.heatmap(data=conf_mat, xticklabels=labels, yticklabels=labels, annot=True, fmt='g')

            plt.title(f'C={c}, K={k}')

            plt.show()

def accuracy_vs_gamma():
    # Define hyperparameters

    g_vals = [1e-4, 1e-5, 1e-3]
    labels = ['vaca', 'pasto', 'cielo']

    for g in g_vals:
        # Train the SVM model

        print(f'Training with G={g}...')

        clf = svm.SVC(kernel='rbf', C=10, gamma=g)

        clf.fit(X_train, y_train)

        # Evaluate it

        print('Evaluating...')

        y_pred = clf.predict(X_test)

        conf_mat = metrics.confusion_matrix(y_test, y_pred, labels=labels)

        accuracy = metrics.accuracy_score(y_test, y_pred)

        print(f'Accuracy: {accuracy}')

        sns.heatmap(data=conf_mat, xticklabels=labels, yticklabels=labels, annot=True, fmt='g')

        plt.title(f'Gamma={g}')

        plt.show()


def full_image_plot(img_path):

    img = Image.open(img_path)

    w, h = img.size

    # Train with best parameters

    print('Training...')

    clf = svm.SVC(kernel='poly', C=1)

    clf.fit(X_train, y_train)

    # Classify the whole image

    print('Evaluating...')

    X_test = list(img.getdata())

    y_pred = clf.predict(X_test)

    # Plot an image of the labels

    print('Plotting...')

    pixels = np.zeros((h, w, 3))

    colors = {'vaca': [102/255, 57/255, 30/255], 'cielo': [66/255, 181/255, 201/255], 'pasto': [106/255, 161/255, 39/255]}

    for i in range(len(y_pred)):
        label = y_pred[i]
        pixels[i // w, i % w] = colors[label]

    plt.imshow(pixels)

    plt.show()

# full_image_plot('data/imagen3.jpg')
accuracy_vs_gamma()