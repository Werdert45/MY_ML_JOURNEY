# This is a movie classifier, combining ... with ..., insert a movie query and output the movies, most related to this movie,
# based on the experience and reviews of other users (This only comprises of ratings 1-5 and not sentiment analysis or NLP

# Libraries
import pandas as pd
import numpy as np


# Get the data

# Users and rating
column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=column_names)

# Movie titles
movie_titles = pd.read_csv("Movie_Id_Titles")


# Merge the datasets
df = pd.merge(df, movie_titles, on='item_id')
df.drop('timestamp', axis=1, inplace=True)

# Set ratings to mean ratings to the movie
ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())

moviemat = df.pivot_table(index='user_id',columns='title',values='rating')

# Input for the wanted movie:

movie_recommend = 'Star Wars (1977)'

movie_user_rating = moviemat[movie_recommend]

similar_to_movie = moviemat.corrwith(movie_user_rating)

corr_movie = pd.DataFrame(similar_to_movie, columns=['Correlation'])
corr_movie.dropna(inplace=True)
corr_movie = corr_movie.join(ratings['num of ratings'])

# Print the result of the best correlated movies, with 50 or more reviews

corr_movie = corr_movie[corr_movie['num of ratings']>50].sort_values('Correlation',ascending=False)

print(corr_movie.head())