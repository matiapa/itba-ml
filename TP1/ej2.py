
import pandas as pd


def noticias_argentinas_parser(path_to_file):
    #parse the file indicated in the parameter path_to_file. its an csv file where the first column is the date, 
    # the second column is the title of the news, the third column is the source of the news and the fourth column is the category of the news.
    #return a dataframe with the parsed data.
    df = pd.read_csv(path_to_file, sep=';', header=0, names=['date', 'title', 'source', 'category'])
    
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
    total_words_by_category = {}
    total_words = 0
    for index, row in df.iterrows():
        for word in row['title'].split(' '):
            total_words += 1
            #handling aparitions of words by specific category
            if (word, row['category']) in dictionary:
                dictionary[(word, row['category'])] += 1
            else:
                dictionary[(word, row['category'])] = 1
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
    for category in all_categories:
        valor_de_clasificacion_por_categoria[category] = 0
        for word in title.split(' '):
            if (word, category) in dictionary:
                #using laplace smoothing
                if valor_de_clasificacion_por_categoria[category] == 0:
                    valor_de_clasificacion_por_categoria[category] = (dictionary[(word, category)] + 1) / (total_words_by_category[category] + all_categories.size)
                else:
                    valor_de_clasificacion_por_categoria[category] *= ((dictionary[(word, category)]+1)/(total_words_by_category[category] + all_categories.size)) 
        valor_de_clasificacion_por_categoria[category] *= (total_words_by_category[category]/total_words)

    print("valor_de_clasificacion_por_categoria: " + str(valor_de_clasificacion_por_categoria))
    denominator = 0
    for value in valor_de_clasificacion_por_categoria.values():
        denominator += value
    
    probabilities_by_category = {}
    for category in all_categories: 
        probabilities_by_category[category] = valor_de_clasificacion_por_categoria[category]/denominator
    print("probabilities_by_category: " + str(probabilities_by_category))
    
    print("sum of probabilities: " + str(sum(probabilities_by_category.values())))
    
    #get the category with the highest probability.
    max_probability = 0
    for category in all_categories:
        if probabilities_by_category[category] > max_probability:
            max_probability = probabilities_by_category[category]
            category_with_max_probability = category
    print("The category with the highest probability is: " + str(category_with_max_probability) + " with a probability of " + str(max_probability))
    return category_with_max_probability




def get_accuracy(all_categories, dictionary, total_words_by_category, total_words, test):
    #return the accuracy of the classifier.
    correct_predictions = 0
    total_predictions = 0
    for index, row in test.iterrows():
        total_predictions += 1
        if analyze_title(all_categories, row['title'], dictionary, total_words_by_category, total_words) == row['category']:
            correct_predictions += 1
    return correct_predictions/total_predictions





df = noticias_argentinas_parser("data/Noticias_argentinas_no_extra_columns.csv")
train, test = divide_dataset(df, 0.8)
dictionary, total_words_by_category, total_words = fill_dictionary(train)
#analyze_title(get_categories(df), "Santiago Bal expuso a Carmen Barbieri en vivo contándole por qué le fue infiel", dictionary, total_words_by_category, total_words)
get_accuracy(get_categories(df), dictionary, total_words_by_category, total_words, test)

