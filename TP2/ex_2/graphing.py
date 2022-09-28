from knn import get_accuracy
import matplotlib.pyplot as plt
import pandas as pd
from dividing_dataset import divide_test_train

def accuracy_vs_k(train_set, test_set):
    #for values of k from 1 to 10, get the accuracy of the knn algorithm
    #graph the accuracy vs k

    #get the accuracy for each k
    accuracies = []
    for k in range(1, 11):
        accuracy = get_accuracy(train_set, test_set, k, "normal")
        accuracies.append(accuracy)

    print("hola")

    #graph the accuracy vs k
    #plot with dots and lines
    plt.plot(range(1, 11), accuracies, 'bo-')
    plt.xlabel('k')
    plt.ylabel('accuracy')
    
    plt.show()


list_of_data = pd.read_csv('reviews_sentiment_parsed.csv', sep=';')
list_of_data.sample(frac=1).reset_index(drop=True)
list_of_data = list_of_data.to_numpy()

#257 es un numero primo, así que redondeemos a 258. Los divisores son: 1,2,3,6,43,86,129,258. 
#tiene sentido probar con k de 3,6,43,86. mas o menos de eso es demasiado chico o grande. 
train_set, test_set = divide_test_train(list_of_data, 43, 0)
accuracy_vs_k(train_set, test_set)


