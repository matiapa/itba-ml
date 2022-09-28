import pandas as pd
from knn import evaluate
import matplotlib.pyplot as plt
import datetime

def cross_validation(df, knn_k, knn_method, validation_k=5):
    block_size = len(df) // validation_k

    best_acc = 0
    best_test_set = None
    best_train_set = None


    for i in range(1,validation_k):
        test_set = df.iloc[i * block_size: (i + 1) * block_size]
        train_set = df.drop(test_set.index)
        
        acc = evaluate(train_set.to_numpy(), test_set.to_numpy(), knn_k, knn_method)


        if acc > best_acc:
            best_acc = acc
            best_test_set = test_set
            best_train_set = train_set


 
    return best_acc, best_train_set, best_test_set


def accuracy_vs_k(df):
    #for values of k from 1 to 10, get the accuracy of the knn algorithm
    #graph the accuracy vs k

    accuracies = []
    accuracies_normal = []
    for k_knn in range(1, 20):
        acc,train_set, test_set = cross_validation(df, k_knn, "weighted")
        accuracies_normal.append(acc)

        acc,train_set, test_set = cross_validation(df, k_knn, "normal")
        accuracies.append(acc)


    #graph the accuracy vs k
    #plot with dots and lines
    plt.plot(range(1, 20), accuracies, 'bo-')
    plt.plot(range(1, 20), accuracies_normal, 'ro-')
    plt.legend(['weighted', 'normal'])
    plt.xlabel('knn k')
    plt.ylabel('accuracy')
    plt.xticks(range(1, 20))

    #get timestamp
    timestamp = str(datetime.datetime.now())
    timestamp = timestamp.replace(" ", "_")
    timestamp = timestamp.replace(":", "-")
    plt.savefig("accuracy_vs_k_" + timestamp + ".png")
    
    plt.show()

def graph_accuracy_vs_validation_k(df, knn_k, validation_k_max=10):
    accuracies = []
    accuracies_normal = []
    for i in range(2, validation_k_max):
        acc, _, _ = cross_validation(df, knn_k, "weighted", i)
        accuracies.append(acc)

    for i in range(2, validation_k_max):
        acc, _, _ = cross_validation(df, knn_k, "normal", i)
        accuracies_normal.append(acc)

    plt.plot(range(2, validation_k_max), accuracies, 'bo-')
    plt.xlabel('ammount of blocks in cross validation')
    plt.ylabel('accuracy')

    plt.plot(range(2, validation_k_max), accuracies_normal, 'ro-')

    #set legend to read that red is weighted and blue is normal
    plt.legend(['weighted', 'normal'])
    #get timestamp
    timestamp = str(datetime.datetime.now())
    timestamp = timestamp.replace(" ", "_")
    timestamp = timestamp.replace(":", "-")
    plt.savefig("accuracy_vs_validation_k_" + timestamp + ".png")

    plt.show()

df = pd.read_csv('reviews_sentiment_parsed.csv', sep=';')
df = df.sample(frac=1).reset_index(drop=True)

#best_acc, _, _ = cross_validation(df, knn_k = 10, knn_method = 'normal', validation_k = 10)
#graph_accuracy_vs_validation_k(df, knn_k = 6, validation_k_max = 10)
accuracy_vs_k(df)

#print(best_acc)