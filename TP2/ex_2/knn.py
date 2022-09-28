import math
from dividing_dataset import divide_test_train
import numpy as np
import pandas as pd


def main(train_set, test_set, k, which_knn):
    #which knn can be two values: normal or weighted

    #return the accuracy of the knn algorithm

    #create structure of three dimensions were the points of the train set will be stored
    #first dimension: wordcount, second dimension: title sentiment, third dimension: sentiment value
    #the objective value is the star rating

    filtered_train_set = []
    for i in range(len(train_set)):
        filtered_train_set.append([train_set[i][2], train_set[i][3], train_set[i][6], train_set[i][5]])

    correct_predictions = 0

    for i in range(len(test_set)):
        nearest_neighbors = get_nearest_neighbors(filtered_train_set, test_set[i], k)

        #decide which function to call
        if which_knn == "normal":
            prediction = normal_knn(nearest_neighbors)
        if which_knn == "weighted":
            prediction = weighted_knn(nearest_neighbors, test_set[i])
        
        if int(prediction) == int(test_set[i][5]):
            correct_predictions += 1

    accuracy = correct_predictions / len(test_set)
    print("accuracy: " + str(accuracy))

    return accuracy


def normal_knn(nearest_neighbors):
    ammount_of_neighbors_per_rating = [0,0,0,0,0]

    for i in range(len(nearest_neighbors)):
        ammount_of_neighbors_per_rating[int(nearest_neighbors[i][3])-1] += 1
    
    return np.argmax(ammount_of_neighbors_per_rating) + 1


def weighted_knn(nearest_neighbors, test_point):
    ammount_of_neighbors_per_rating = [0,0,0,0,0]

    for i in range(len(nearest_neighbors)):
        distance = get_distance(nearest_neighbors[i], [test_point[2], test_point[3], test_point[6]])

        if distance == 0:
            ammount_of_neighbors_per_rating[int(nearest_neighbors[i][3])-1] += 1
        else:
            ammount_of_neighbors_per_rating[int(nearest_neighbors[i][3])-1] +=  1/pow(distance, 2)

    return np.argmax(ammount_of_neighbors_per_rating) + 1


def get_distance(point1, point2):
    distance_wordcount = pow(float(point1[0]) - float(point2[0]),2)
    distance_title_sentiment = pow(float(point1[1]) - float(point2[1]),2)
    distance_sentiment_value = pow(float(point1[2]) - float(point2[2]),2)

    # print("distance_wordcount: " + str(distance_wordcount) + "   distance_title_sentiment: " + str(distance_title_sentiment) + "   distance_sentiment_value: " + str(distance_sentiment_value))

    return math.sqrt(distance_wordcount + distance_title_sentiment + distance_sentiment_value)


def get_nearest_neighbors(filtered_train_set, test_point, k):
    distances = []
    for i in range(len(filtered_train_set)):
        distance = get_distance(filtered_train_set[i], [test_point[2], test_point[3], test_point[6]])
        distances.append([distance, i])

    distances.sort()

    nearest_neighbors = []
    for i in range(k):
        nearest_neighbors.append(filtered_train_set[distances[i][1]])
    
    return nearest_neighbors


list_of_data = pd.read_csv('reviews_sentiment_parsed.csv', sep=';')
list_of_data.sample(frac=1).reset_index(drop=True)
list_of_data = list_of_data.to_numpy()

#257 es un numero primo, así que redondeemos a 258. Los divisores son: 1,2,3,6,43,86,129,258. 
#tiene sentido probar con k de 3,6,43,86. mas o menos de eso es demasiado chico o grande. 
train_set, test_set = divide_test_train(list_of_data, 43, 0)

accuracy = main(train_set, test_set, 6, "normal")