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

# def parsing_file(file_name,  normalize_variables = True):
#     #parse reviews_sentiment.csv file 
#     #return a list of tuples (review_title, review_text, wordcount, titleSentiment, textSentiment, starRating, sentimentValue)
#     # values wordcount and sentimentValue are already normalized 
#     list = []
#     #open file
#     file = open(file_name, 'r')
#     #read file
#     lines = file.readlines()

#     #skip the first line
#     lines = lines[1:]
#     max_wordCount = None 
#     max_sentimentValue = None
#     min_wordCount = None 
#     min_sentimentValue = None

#     for line in lines: 
#         #split line
#         line = line.split(';')
#         #get title, text, wordcount, titleSentiment, textSentiment, starRating, sentimentValue
#         title = line[0]
#         text = line[1]

#         wordcount = int(line[2])
#         if normalize_variables:
#             if min_wordCount == None or min_wordCount > wordcount:
#                 min_wordCount = wordcount
#             if max_wordCount == None or max_wordCount < wordcount:
#                 max_wordCount = wordcount

#         titleSentiment = line[3]
#         if normalize_variables:
#             if(titleSentiment == ''):
#                 titleSentiment = 0.5
#             if(titleSentiment == "positive"):
#                 titleSentiment = 1
#             if(titleSentiment == "negative"):
#                 titleSentiment = 0
#         else:
#             if(titleSentiment == ''):
#                 titleSentiment = 0
#             if(titleSentiment == "positive"):
#                 titleSentiment = 1
#             if(titleSentiment == "negative"):
#                 titleSentiment = -1
#         textSentiment = line[4]
#         starRating = line[5]
       
#         sentimentValue = float(line[6])
#         if normalize_variables:
#             if min_sentimentValue == None or min_sentimentValue > sentimentValue:
#                 min_sentimentValue = sentimentValue
#             if max_sentimentValue == None or max_sentimentValue < sentimentValue:
#                 max_sentimentValue = sentimentValue
#         #create tuple
#         tuple = [title, text, wordcount, titleSentiment, textSentiment, starRating, sentimentValue]
#         #append tuple to list
#         list.append(tuple)
#     #close file
#     file.close()

#     print("min wordcount: " + str(min_wordCount))
#     print("max wordcount: " + str(max_wordCount))
#     print("min sentiment: " + str(min_sentimentValue))
#     print("max sentiment: " + str(max_sentimentValue))

#     #normalize the values of wordcount, and sentimentValue
#     range_wordcount = max_wordCount - min_wordCount
#     range_sentimentValue = max_sentimentValue - min_sentimentValue
#     for i in range(len(list)):
#         list[i][2] = (int(list[i][2]) - min_wordCount) / range_wordcount
#         list[i][6] = (float(list[i][6]) - min_sentimentValue) / range_sentimentValue
#         #print("wordcount: " + str(list[i][2]) + "   sentiment: " + str(list[i][6]))

#     return list
