import pandas as pd
from knn import evaluate

def cross_validation(df, knn_k, knn_method, validation_k=5):
    block_size = len(df) // validation_k

    best_acc = 0
    best_test_set = None
    best_train_set = None

    for i in range(validation_k):
        test_set = df.iloc[i * block_size: (i + 1) * block_size]
        train_set = df.drop(test_set.index)
        
        acc = evaluate(train_set.to_numpy(), test_set.to_numpy(), knn_k, knn_method)

        if acc > best_acc:
            best_acc = acc
            best_test_set = test_set
            best_train_set = train_set
    
    return best_acc, best_train_set, best_test_set

df = pd.read_csv('reviews_sentiment_parsed.csv', sep=';')
df.sample(frac=1).reset_index(drop=True)

best_acc, _, _ = cross_validation(df, knn_k = 6, knn_method = 'weighted', validation_k = 5)

print(best_acc)