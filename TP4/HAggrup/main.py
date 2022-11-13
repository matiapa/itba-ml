from haggrup import HAggrup
import numpy as np
import pandas as pd


df_proc = pd.read_csv('../data/movie_data_proc.csv')
df_proc = df_proc.drop(columns=['imdb_id'])

hg = HAggrup()
hg.fit(df_proc.to_numpy())

# hg.fit(np.array([[1], [2], [7], [9]]))

for aggrupation in hg.aggrupations:
    for group in aggrupation:
        print(group)
    print('--------------------')