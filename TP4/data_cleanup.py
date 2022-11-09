import pandas as pd

df = pd.read_csv('data/movie_data.csv', sep=';')
df_proc = df.copy()

# Clean columns
df_proc = df_proc.drop(columns=['original_title', 'overview', 'release_date'])
imdb_id_index = df_proc.columns.get_loc('imdb_id')
cols = df_proc.columns.tolist()
cols = cols[2:3] + cols[0:2] + cols[3:]
df_proc = df_proc[cols]

# Clean Genres
genres = df['genres'].unique()
genre_index = {genre:i for i, genre in enumerate(genres)}
df_proc['genres'] = df['genres'].map(genre_index)

# Normalize columns
cols = df_proc.columns
for col in df_proc.columns:
    if col == 'imdb_id':
        continue
    df_proc[col] = (df_proc[col] - df_proc[col].min()) / (df_proc[col].max() - df_proc[col].min())

# Clean rows with nan values with mode of column
for col in df_proc.columns:
    if col == 'imdb_id':
        continue
    df_proc[col] = df_proc[col].fillna(df_proc[col].mode()[0])

# Populate nan ids with next id
ids = df['imdb_id'].sort_values().values
last_id = ids[~pd.isnull(ids)][-1]
last_id = last_id[:2] + str(int(last_id[2:]) + 1)
for i, id in enumerate(df_proc['imdb_id']):
    if pd.isnull(id):
        last_id = last_id[:2] + str(int(last_id[2:]) + 1)
        df_proc['imdb_id'][i] = last_id

# Remove repeated rows
# df_proc = df_proc.drop_duplicates()

df_proc.to_csv('data/movie_data_proc.csv', index=False)