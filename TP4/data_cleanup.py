import pandas as pd

df = pd.read_csv('data/movie_data.csv', sep=';')
df_proc = df.copy()

# Clean columns
df_proc = df_proc.drop(columns=['original_title', 'overview', 'release_date', 'genres'])
imdb_id_index = df_proc.columns.get_loc('imdb_id')
cols = df_proc.columns.tolist()
cols = cols[2:3] + cols[0:2] + cols[3:]
df_proc = df_proc[cols]

# Normalize columns
cols = df_proc.columns
for col in df_proc.columns:
    if col == 'imdb_id':
        continue
    df_proc[col] = (df_proc[col] - df_proc[col].min()) / (df_proc[col].max() - df_proc[col].min())

# Remove rows with nan values
df_proc = df_proc.dropna() # TODO change this, add mean values

# Remove repeated rows
# df_proc = df_proc.drop_duplicates()

# save the processed dataframe
df_proc.to_csv('data/movie_data_proc.csv', index=False)