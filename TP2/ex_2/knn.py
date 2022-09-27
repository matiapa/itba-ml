import math
from parsing import parsing_file
from dividing_dataset import divide_test_train

def main(train_set, test_set, k):
    #return the accuracy of the knn algorithm

    #create structure of three dimensions were the points of the train set will be stored
    #first dimension: wordcount, second dimension: title sentiment, third dimension: sentiment value
    #the objective value is the star rating

    filtered_train_set = []
    for i in range(len(train_set)):
        filtered_train_set.append([train_set[i][2], train_set[i][3], train_set[i][6], train_set[i][5]])

    #get predictions
    predictions = []
    for i in range(len(test_set)):
        nearest_neighbors = get_nearest_neighbors(filtered_train_set, test_set[i], k)
        prediction = normal_knn(nearest_neighbors)
        predictions.append(prediction)
    
    print("predictions: " + str(predictions))

    #get accuracy
    correct_predictions = 0
    for i in range(len(predictions)):
        #print("prediction: " + str(predictions[i]) + "   actual: " + str(test_set[i][5]))

        if predictions[i] == test_set[i][5]:
            correct_predictions += 1

    accuracy = correct_predictions / len(predictions)
    print("accuracy: " + str(accuracy))
    return accuracy




def normal_knn(nearest_neighbors):
    ammount_of_neighbors_per_rating = [0,0,0,0,0]
    for i in range(len(nearest_neighbors)):
        ammount_of_neighbors_per_rating[int(nearest_neighbors[i][3])-1] += 1
    
    #get the most common rating
    other_list = ammount_of_neighbors_per_rating.copy()
    print("ammount_of_neighbors_per_rating: " + str(ammount_of_neighbors_per_rating))
    ammount_of_neighbors_per_rating.sort()
    highest = ammount_of_neighbors_per_rating[4]
    for i in range(len(other_list)):
        if other_list[i] == highest:
            return i+1


def ponderated_distance_knn(nearest_neighbors):
    #get the prediction of the rating using the ponderated distance method
    pass


def get_distance(point1, point2):
    #return the distance between two points

    #get the distance in the wordcount dimension
    distance_wordcount = pow(abs(int(point1[0]) - int(point2[0])),2)
    #get the distance in the title sentiment dimension
    distance_title_sentiment = pow(abs(int(point1[1]) - int(point2[1])),2)
    #get the distance in the sentiment value dimension
    distance_sentiment_value = pow(abs(float(point1[2]) - float(point2[2])),2)

    #get the total distance
    distance = math.sqrt(distance_wordcount + distance_title_sentiment + distance_sentiment_value)
    return distance

def get_nearest_neighbors(filtered_train_set, test_point, k):
    #return the k nearest neighbors of the test_point

    #get the distances of the test_point to all the points in the filtered_train_set
    distances = []
    for i in range(len(filtered_train_set)):
        filtered_test_point = [test_point[2], test_point[3], test_point[6]]
        distances.append([get_distance(filtered_train_set[i], filtered_test_point), i])
    #sort the distances
    distances.sort()
    #get the k nearest neighbors
    nearest_neighbors = []
    for i in range(k):
        nearest_neighbors.append(filtered_train_set[distances[i][1]])
    return nearest_neighbors



list_of_data = parsing_file('./ex_2/reviews_sentiment.csv')
#257 es un numero primo, así que redondeemos a 258. Los divisores son: 1,2,3,6,43,86,129,258. 
#tiene sentido probar con k de 3,6,43,86. mas o menos de eso es demasiado chico o grande. 
train_set, test_set = divide_test_train(list_of_data, 43, 0)
accuracy = main(train_set, test_set, 43)