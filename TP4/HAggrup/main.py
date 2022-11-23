# from haggrup import HAggrup
from Hgroup import HGroup
import numpy as np
import pandas as pd


df_proc = pd.read_csv('../data/movie_data_proc.csv')
# remove duplicates
df_proc = df_proc.drop_duplicates()
columns = df_proc.columns.drop('imdb_id').to_list()
columns.append('imdb_id')
df_proc = df_proc[columns]

hg = HGroup()
hg.fit(df_proc.to_numpy())

# print(hg.levels)

file = open('groups.csv', 'w')
file.write('level,size,elements\n')
num = 0
for ids in hg.levels:
    file.write("{},{},{}\n".format(num, len(ids), ids))
    num+=1