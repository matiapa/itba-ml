from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
from imblearn.over_sampling import SMOTE

def load_data(undersample=False, oversample=False):
    le = LabelEncoder()
    df = pd.read_csv('data/train_LZdllcl.csv')
    df.dropna(inplace=True)

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col].values)

    df = df.sample(frac=1).reset_index(drop=True)
    if undersample:
        df_0 = df[df['is_promoted'] == 0]
        df_1 = df[df['is_promoted'] == 1]
        df = pd.concat([df_0[:len(df_1)], df_1], axis=0)
        df = df.sample(frac=1).reset_index(drop=True)

    y = df['is_promoted']
    X = df.drop(['employee_id', 'is_promoted'], axis=1)

    if oversample:
        smote = SMOTE()
        X, y = smote.fit_resample(X, y)

    print(X.shape, y.shape)
    print(y.value_counts())


    return X, y

def cross_validation(X, y, k=5, n_estimators=50, learning_rate=1):
    best_accuracy = 0
    results = None
    block_size = len(X) // k
    y_pred = None
    y_test = None
    for i in range(k):
        X_test = X.iloc[i*block_size:(i+1)*block_size]
        X_train = X.drop(X_test.index)
        y_test = y.iloc[i*block_size:(i+1)*block_size]
        y_train = y.drop(y_test.index)

        abc = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        model = abc.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            results = confusion_matrix(y_pred, y_test, plot=False)
            y_pred = y_pred
            y_test = y_test

    return results, best_accuracy, metrics(y_pred, y_test)


def metrics(y_pred, y_test):
    metrics = {'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_test.iloc[i] == 1:
            metrics['tp'] += 1
        if y_pred[i] == 1 and y_test.iloc[i] == 0:
            metrics['fp'] += 1
        if y_pred[i] == 0 and y_test.iloc[i] == 0:
            metrics['tn'] += 1
        if y_pred[i] == 0 and y_test.iloc[i] == 1:
            metrics['fn'] += 1
    result = {}
    # add accuracy to results
    result['accuracy'] = (metrics['tp'] + metrics['tn']) / (metrics['tp'] + metrics['tn'] + metrics['fp'] + metrics['fn'])
    if metrics['tp'] + metrics['fp'] == 0:
        result['precision'] = 0
    else:
        result['precision'] = metrics['tp'] / (metrics['tp'] + metrics['fp'])
    if metrics['tp'] + metrics['fn'] == 0:
        result['recall'] = 0
    else:
        result['recall'] = metrics['tp'] / (metrics['tp'] + metrics['fn'])
    if result['precision'] + result['recall'] == 0:
        result['f1'] = 0
    else:
        result['f1'] = 2 * (result['precision'] * result['recall']) / (result['precision'] + result['recall'])
    if metrics['tp'] + metrics['fn'] == 0:
        result['tp_rate'] = 0
    else:
        result['tp_rate'] = metrics['tp'] / (metrics['tp'] + metrics['fn'])
    if metrics['fp'] + metrics['tn'] == 0:
        result['fp_rate'] = 0
    else:
        result['fp_rate'] = metrics['fp'] / (metrics['fp'] + metrics['tn'])
    return result


def confusion_matrix(y_pred, y_test, plot=True):
    class_match = {
        0: 0,
        1: 0
    }
    for i in range(0, len(y_test)):
        if (y_test.iloc[i] == y_pred[i]):
            class_match[y_test.iloc[i]] += 1

    results = [[class_match[0]/y_test.value_counts()[0], (y_test.value_counts()[0] - class_match[0])/y_test.value_counts()[0]], [(y_test.value_counts()[1] - class_match[1])/y_test.value_counts()[1], class_match[1]/y_test.value_counts()[1]]]    
    if plot:
        plt.figure(figsize=(5, 5))
        plt.title("Confusion matrix")
        plt.xlabel("Predicted class")
        plt.ylabel("True class")
        plt.imshow(results, cmap='Blues')
        plt.colorbar()
        plt.xticks([0, 1], ['0', '1'])
        plt.yticks([0, 1], ['0', '1'])
        for i in range(2):
            for j in range(2):
                t = plt.text(j, i, str(round(results[i][j]*100, 2)) + "%", ha="center", va="center")
                t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
        plt.show()

    return results


def learning_rate_analysis(undersample=False):
    # print("Learning rate analysis...")
    learning_rates = [0.02, 0.2, 0.4, 0.6, 0.8, 1]
    samples = 10
    avg_accuracies = {
        'total': [],
        0: [],
        1: []
    }
    std_accuracies = {
        'total': [],
        0: [],
        1: []
    }
    for lr in learning_rates:
        # print("Learning rate:", lr)
        accuracy = {
            'total': [],
            0: [],
            1: []
        }
        for _ in range(samples):
            X, y = load_data(undersample=undersample)
            results, best_accuracy, _ = cross_validation(X, y, learning_rate=lr)
            accuracy['total'].append(best_accuracy)
            accuracy[0].append(results[0][0])
            accuracy[1].append(results[1][1])

        avg_accuracies['total'].append(np.mean(accuracy['total']))
        avg_accuracies[0].append(np.mean(accuracy[0]))
        avg_accuracies[1].append(np.mean(accuracy[1]))
        std_accuracies['total'].append(np.std(accuracy['total']))
        std_accuracies[0].append(np.std(accuracy[0]))
        std_accuracies[1].append(np.std(accuracy[1]))

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].plot(learning_rates, avg_accuracies['total'], label='Total accuracy', marker='o')
    ax[0].plot(learning_rates, avg_accuracies[0], label='Class 0 accuracy', marker='o')
    ax[0].plot(learning_rates, avg_accuracies[1], label='Class 1 accuracy', marker='o')
    ax[0].set_xlabel("Learning Rate")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[1].plot(learning_rates, avg_accuracies['total'], label='Total accuracy', marker='o')
    ax[1].set_xlabel("Learning Rate")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    ax[2].plot(learning_rates, avg_accuracies[1], label='Class 1 accuracy', marker='o')
    ax[2].set_xlabel("Learning Rate")
    ax[2].set_ylabel("Accuracy")
    ax[2].legend()
    fig.tight_layout(pad=1.0)
    plt.show()


def accuracy(X, y):
    results, best_accuracy, _ = cross_validation(X, y)

    print("Accuracy:", round(best_accuracy*100, 2), "%")
    plt.figure(figsize=(5, 5))
    plt.title("Accuracy")
    plt.xlabel("Clase predicha")
    plt.ylabel("Clase verdadera")
    plt.imshow(results, cmap='Greens')
    plt.colorbar()
    plt.xticks([0, 1], ['0', '1'])
    plt.yticks([0, 1], ['0', '1'])
    for i in range(2):
        for j in range(2):
            t = plt.text(j, i, str(round(results[i][j]*100, 2)) + "%", ha="center", va="center")
            t.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white'))
    plt.show()


def estimators_analysis(undersample=False):
    # print("Estimators analysis...")
    samples = 10
    estimators = [10, 20, 40, 60, 80, 100]
    avg_accuracies = {
        'total': [],
        0: [],
        1: []
    }
    std_accuracies = {
        'total': [],
        0: [],
        1: []
    }
    for n in estimators:
        # print("Estimators:", n)
        accuracy = {
            'total': [],
            0: [],
            1: []
        }
        for _ in range(samples):
            X, y = load_data(undersample=undersample)
            results, best_accuracy, _ = cross_validation(X, y, n_estimators=n)
            accuracy['total'].append(best_accuracy)
            accuracy[0].append(results[0][0])
            accuracy[1].append(results[1][1])

        avg_accuracies['total'].append(np.mean(accuracy['total']))
        avg_accuracies[0].append(np.mean(accuracy[0]))
        avg_accuracies[1].append(np.mean(accuracy[1]))
        std_accuracies['total'].append(np.std(accuracy['total']))
        std_accuracies[0].append(np.std(accuracy[0]))
        std_accuracies[1].append(np.std(accuracy[1]))

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].plot(estimators, avg_accuracies['total'], label='Total accuracy', marker='o')
    ax[0].plot(estimators, avg_accuracies[0], label='Class 0 accuracy', marker='o')
    ax[0].plot(estimators, avg_accuracies[1], label='Class 1 accuracy', marker='o')
    ax[0].set_xlabel("Estimators")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[1].plot(estimators, avg_accuracies['total'], label='Total accuracy', marker='o')
    ax[1].set_xlabel("Estimators")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    ax[2].plot(estimators, avg_accuracies[1], label='Class 1 accuracy', marker='o')
    ax[2].set_xlabel("Estimators")
    ax[2].set_ylabel("Accuracy")
    ax[2].legend()
    fig.tight_layout(pad=1.0)
    plt.show()

def precision(undersample=False):
    X, y = load_data(undersample=undersample)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

    abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)
    model = abc.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    metrics = {
        'tp': 0,
        'fp': 0,
        'tn': 0,
        'fn': 0
    }

    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_test.iloc[i] == 1:
            metrics['tp'] += 1
        if y_pred[i] == 1 and y_test.iloc[i] == 0:
            metrics['fp'] += 1
        if y_pred[i] == 0 and y_test.iloc[i] == 0:
            metrics['tn'] += 1
        if y_pred[i] == 0 and y_test.iloc[i] == 1:
            metrics['fn'] += 1

    print("Precision:", metrics['tp'] / (metrics['tp'] + metrics['fp']))

def metrics_results(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 

    abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)
    model = abc.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    result = metrics(y_pred, y_test)  

    print("Accuracy:", round(result['accuracy']*100, 2), "%")
    print("Precision:", round(result['precision']*100, 2), "%")
    print("Recall:", round(result['recall']*100, 2), "%")
    print("F1:", round(result['f1']*100, 2), "%")
    print("True positive rate:", round(result['tp_rate']*100, 2), "%")
    print("False positive rate:", round(result['fp_rate']*100, 2), "%")

def metrics_learning_rate(types, samples=10, undersample=False):
    learning_rates = [0.02, 0.2, 0.4, 0.6, 0.8, 1]
    avg = {}
    std = {}
    for type in types:
        avg[type] = []
        std[type] = []
    
    for lr in tqdm(learning_rates):
        metric = {}
        for type in types:
            metric[type] = []
        for _ in range(samples):
            X, y = load_data(undersample=undersample)
            _, _, results = cross_validation(X, y, learning_rate=lr)
            for type in types:
                metric[type].append(results[type]*100)

        for type in types:
            avg[type].append(np.mean(metric[type]))
            std[type].append(np.std(metric[type]))
    for type in types:
        plt.errorbar(learning_rates, avg[type], yerr=std[type], label=type)
        plt.xlabel("Learning rate")
        plt.ylabel(type + " (%)")
        plt.show()

def metrics_estimators(types, samples=10, undersample=False):
    estimators = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    avg = {}
    std = {}
    for type in types:
        avg[type] = []
        std[type] = []
    
    for n in tqdm(estimators):
        metric = {}
        for type in types:
            metric[type] = []
        for _ in range(samples):
            X, y = load_data(undersample=undersample)
            _, _, results = cross_validation(X, y, n_estimators=n)
            for type in types:
                metric[type].append(results[type]*100)

        for type in types:
            avg[type].append(np.mean(metric[type]))
            std[type].append(np.std(metric[type]))
    for type in types:
        plt.errorbar(estimators, avg[type], yerr=std[type], label=type)
        plt.xlabel("Cantidad de estimadores")
        plt.ylabel(type + " (%)")
        plt.show()
