import pandas as pd

df = pd.read_csv('data/movie_data.csv', sep=';')
df_proc = df.copy()

# Filter Genres
df_proc = df_proc[df_proc['genres'].isin(['Action', 'Comedy', 'Drama'])]

# Clean columns
df_proc = df_proc.drop(columns=['genres', 'original_title', 'overview', 'release_date', 'production_companies', 'popularity'])
cols = df_proc.columns.tolist()
cols = cols[2:3] + cols[0:2] + cols[3:]
df_proc = df_proc[cols]

# Normalize columns
cols = df_proc.columns
for col in df_proc.columns:
    if col == 'imdb_id' or col == 'genres':
        continue
    df_proc[col] = (df_proc[col] - df_proc[col].mean()) / df_proc[col].std()

# Clean rows with nan values with mode of column
for col in df_proc.columns:
    if col == 'imdb_id' or col == 'genres':
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