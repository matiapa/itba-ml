
import pandas as pd
import json
import seaborn as sn
import matplotlib.pyplot as plt

def laplace_correction(n, N, k):
    return (n + 1) / (N + k)

def noticias_argentinas_parser(path_to_file, categories):
    #parse the file indicated in the parameter path_to_file. its an csv file where the first column is the date, 
    # the second column is the title of the news, the third column is the source of the news and the fourth column is the category of the news.
    #return a dataframe with the parsed data.
    df = pd.read_csv(path_to_file, sep=';', header=0, names=['date', 'title', 'source', 'category'])

    #remove the rows that are not in the categories list.
    df = df[df['category'].isin(categories)]
    
    return df


def divide_dataset(df, train_percentage):
    #divide the dataframe to create two dataframes, one for training and one for testing.
    #return the two dataframes.
    train_size = int(len(df) * train_percentage)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    return train, test


def get_categories(df):
    #return a list with the categories of the news in the dataframe.
    categories = df['category'].unique()
    #remove category nan from the list.
    print("The categories are: " + str(categories))
    print("The number of categories is: " + str(len(categories)))
    return categories

def get_all_words(df):
    #words is a set in order to not have repeated words.
    words = set()
    for index, row in df.iterrows():
        words.update(row['title'].split(' '))
    return words
        
def get_titles_by_category(df, category):
    #return a list with the titles of the news in the dataframe that are in the category indicated in the parameter category.
    titles = df[df['category'] == category]['title'].tolist()
    return titles


def get_words_in_category(df, category):
    titles = get_titles_by_category(df, category)
    words = set()
    for title in titles:
        words.update(title.split(' '))



def separate_by_class(df):
    #separate the dataframe by class.
    for word in get_all_words(df):
        df.loc[df['title'] == word, word] = int(1)
        df.loc[ df['title'] != word, word] = int(0)
    print(df)



def fill_dictionary(df):
   #hash with keys being pair word category
   #and values being the number of times the word appears in the category.
   #for example: {('hola', 'Deportes'): 2, ('hola', 'Economía'): 1} 
    dictionary = {} 
    total_words_by_category = {} #Deportes: 100, Economía: 50
    total_words = 0
    for index, row in df.iterrows():
        words = row['title'].split(' ')
        total_words += len(words)
        for word in words:
            #handling aparitions of words by specific category
            if word in dictionary:
                if row['category'] in dictionary[word]:
                    dictionary[word][row['category']] += 1
                else:
                    dictionary[word][row['category']] = 1
            else:
                dictionary[word] = {}
                dictionary[word][row['category']] = 1

            #Handling the total words by category.
            if row['category'] in total_words_by_category:
                total_words_by_category[row['category']] += 1
            else:
                total_words_by_category[row['category']] = 1
    print("Finished filling the dictionary.")
    return dictionary, total_words_by_category, total_words


def analyze_title(all_categories, title, dictionary, total_words_by_category, total_words):
    #return the probability of the title being in the category.
    valor_de_clasificacion_por_categoria = {}
    title_words = title.split(' ')
    for category in all_categories:
        valor_de_clasificacion_por_categoria[category] = 1

        # iterate through all the words in dictionary
        for word in dictionary:
            # if word is in the title, then we add the probability of the word being in the category
            if word in title_words:
                if category in dictionary[word]:
                    valor_de_clasificacion_por_categoria[category] *= laplace_correction(dictionary[word][category], total_words_by_category[category], len(all_categories))
                else:
                    valor_de_clasificacion_por_categoria[category] *= laplace_correction(0, total_words_by_category[category], len(all_categories))
            else:
                if category in dictionary[word]:
                    valor_de_clasificacion_por_categoria[category] *= 1-laplace_correction(dictionary[word][category], total_words_by_category[category], len(all_categories))
                else:
                    valor_de_clasificacion_por_categoria[category] *= 1-laplace_correction(0, total_words_by_category[category], len(all_categories))

    # print(title)
    # print("valor_de_clasificacion_por_categoria: " + str(valor_de_clasificacion_por_categoria))
    denominator = 0
    for value in valor_de_clasificacion_por_categoria.values():
        denominator += value
    
    probabilities_by_category = {}
    for category in all_categories: 
        probabilities_by_category[category] = valor_de_clasificacion_por_categoria[category]/denominator
        # print("The probability of the title being in the category {} is: {}%".format(category, round(probabilities_by_category[category]*100, 2)))
    # print("probabilities_by_category: " + str(probabilities_by_category))
    
    # print("sum of probabilities: " + str(sum(probabilities_by_category.values())))
    
    #get the category with the highest probability.
    max_probability = 0
    for category in all_categories:
        if probabilities_by_category[category] > max_probability:
            max_probability = probabilities_by_category[category]
            category_with_max_probability = category
    # print("The category with the highest probability is: {} with a probability of {}".format(category_with_max_probability, round(max_probability*100, 2)))
    return category_with_max_probability


def get_metrics(all_categories, dictionary, total_words_by_category, total_words, test):

    tp_metrics = {}
    fp_metrics = {}
    fn_metrics = {}
    tn_metrics = {}

    for category in all_categories:
        tp_metrics[category] = 0
        fp_metrics[category] = 0
        fn_metrics[category] = 0
        tn_metrics[category] = 0

    for index, row in test.iterrows():
        category = analyze_title(all_categories, row['title'], dictionary, total_words_by_category, total_words)
        if category == row['category']:
            tp_metrics[category] += 1
            for c in all_categories:
                if c != category:
                    tn_metrics[c] += 1
        else:
            fp_metrics[category] += 1
            fn_metrics[row['category']] += 1
            for c in all_categories:
                if c != category and c != row['category']:
                    tn_metrics[c] += 1
        
    return tp_metrics, fp_metrics, fn_metrics, tn_metrics

def cross_validation(data, k, all_categories):
    data = data.sample(frac=1).reset_index(drop=True)
    block = data.shape[0] // k
    best_block = -1
    test = []
    best_block_accuracy = 0
    for i in range(k):
        print("Block: " + str(i))
        test = data[i*block:(i+1)*block]
        train = data.drop(test.index)
        dictionary, total_words_by_category, total_words = fill_dictionary(train)

        correct = 0
        for _, row in test.iterrows():
            category = analyze_title(all_categories, row['title'], dictionary, total_words_by_category, total_words)
            if category == row['category']:
                correct += 1

        if correct / block > best_block_accuracy:
            best_block_accuracy = correct / block
            best_block = train
            test = test

    return best_block, test, best_block_accuracy

