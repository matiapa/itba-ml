from parsing import parsing_file

def divide_test_train(data, k, test_block_index):
   #divide data into train and test set. 
   #k is the block size, and block number is the number of the block to be used as test set
    #return train_set, test_set
    train_set = []
    test_set = []
    train_indexes = []
    test_indexes = []

    #get the test set
    for i in range(len(data)):
        if i >= test_block_index*k and i < (test_block_index*k + k) :
            test_set.append(data[i])
            test_indexes.append(i)
        else:
            train_set.append(data[i])
            train_indexes.append(i)

    return train_set, test_set






