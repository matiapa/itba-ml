# from haggrup import HAggrup
import pandas as pd
from Hgroup import HGroup


def run():
    df_proc = pd.read_csv('../data/movie_data_proc.csv')
    df_proc = df_proc.drop_duplicates()

    columns = df_proc.columns.drop('imdb_id').to_list()
    columns.append('imdb_id')
    df_proc = df_proc[columns]

    hg = HGroup()
    hg.fit(df_proc.to_numpy())

    file = open('groups.csv', 'w')
    file.write('level,size,elements\n')
    num = 0
    for ids in hg.levels:
        file.write("{},{},{}\n".format(num, len(ids), ids))
        num+=1


def plot_similarity_vs_k():

    movies_df = pd.read_csv('../data/movie_data_proc.csv')
    movies_df = movies_df.drop_duplicates()
    movies_df.set_index('imdb_id')

    clusters_df = pd.read_csv('../data/groups.csv')

    for i in range(0, len(clusters_df), 100):
        pass