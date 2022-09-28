import pandas as pd

def parse_file():
    df = pd.read_csv('reviews_sentiment.csv', sep=';')

    columns = ['wordcount', 'titleSentiment', 'textSentiment', 'sentimentValue']

    # in column 'textSentiment' replace 'positive' with 1, 'negative' with -1 and '' with 0
    df['textSentiment'] = df['textSentiment'].replace('positive', 1)
    df['textSentiment'] = df['textSentiment'].replace('negative', -1)
    df['textSentiment'] = df['textSentiment'].replace('', 0)

    df['titleSentiment'] = df['titleSentiment'].replace('positive', 1)
    df['titleSentiment'] = df['titleSentiment'].replace('negative', -1)
    df['titleSentiment'] = df['titleSentiment'].replace('', 0)

    # replace nan values in column 'titleSentiment' with 0
    df['titleSentiment'] = df['titleSentiment'].fillna(0)

    for column in columns:
        max_value = df[column].max()
        min_value = df[column].min()
        df[column] = (df[column] - min_value) / (max_value - min_value)

    # save the parsed data to a new csv file
    df.to_csv('reviews_sentiment_parsed.csv', sep=';', index=False)

    return df

parse_file()