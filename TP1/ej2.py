
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



df = noticias_argentinas_parser("data/Noticias_argentinas_no_extra_columns.csv")
separate_by_class(df)
