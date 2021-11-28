import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

movie_dff = pd.read_csv("C:\\Users\\adm\\Documents\\Datasets\\Recommendations on MovieLens\\movie.csv")
rating_dff = pd.read_csv("C:\\Users\\adm\\Documents\\Datasets\\Recommendations on MovieLens\\rating.csv")

movies_dff.shape

movie_dff.head()
# Using regular expressions  to find a year stored between parenthesis
# specify  the parentheses to avoid conflict with movies that have years in their titles.
movie_dff['year'] = movie_dff.title.str.extract('(\(\d\d\d\d\))', expand =False)
# Removing the parenthesis
movie_dff['title'] = movie_dff.title.str.replace('(\(\d\d\d\d\))', '')
# Applying the strip function to get rid of any ending  whitespace characters that may have appeared.
movie_dff['title'] = movie_dff['title'].apply(lambda x:x.strip())
movie_dff.head()
# Removing the| that separetes each genre using split function
movie_dff['genres'] = movie_dff.genres.str.split('|')
movie_dff.head()

# copying movie dataframe into a  new  since we need not use  genre information in our first  case.
moviesWithGenres_df = movie_dff.copy()

# For every row in the dataframe iterate through ye first genre and place a  1 to the corresponding column
for index, row in movie_dff.iterrows():
    for genre in row ['genres']:
        moviesWithGenres_df.at[index,genre] = 1

# Filling in the Nan values  with 0 to show  that a movie doesnt have that column's genre
moviesWithGenres_df = moviesWithGenres_df.fillna(0)
moviesWithGenres_df.head()

# Content -Based recommendation System.
 userInput = [
                {'title':'Breakfast  Club, The','rating':5},
               {'title':'Toy Story', 'rating':3.5},
               {'title': 'Jumanji', 'rating' : 2},
               {'title': 'Pulp Fiction', 'rating' : 5},
               {'title': 'Akira' , 'rating' : 4.5}
                ]
inputMovies = pd.DataFrame(userInput)
inputMovies

#Filtering out the movies by title .  tolist()  converts data elements of an  array into a list
inputId = movie_dff[movie_dff['title'].isin(inputMovies['title'].tolist())]
#Then merging it so we can get the movieId. It's implicitly merging it by title.
inputMovies = pd.merge(inputId, inputMovies)
#Dropping information we won't use from the input dataframe
inputMovies = inputMovies.drop('genres', 1).drop('year', 1)
#Final input dataframe
inputMovies

#Filtering out the movies from the input
userMovies = moviesWithGenres_df[moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]
userMovies
 # Resetting the index to avoid future issues
userMovies = userMovies.reset_index(drop = True)
userMovies

# Dropping unnecessary issues to save memory and avoid issues
userGenreTable = userMovies.drop('movieId' , 1).drop('title',1).drop('genres',1).drop('year',1)
userGenreTable

inputMovies['rating']
# Dot products to get weights
userProfile =  userGenreTable.transpose().dot(inputMovies['rating'])
userProfile

# Recommendation table
# Getting the genres of every movie in the original dataframe
genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
genreTable = genreTable.drop('movieId' , 1).drop('title',1).drop('genres',1).drop('year',1)
genreTable.head()
genreTable.shape

# Multiplying the genres by the weighted  and then the weighted averages
recommendationTable_df =  ((genreTable*userProfile).sum(axis=1))/(userProfile.sum())
recommendationTable_df.head()

# Sorting recommendation in descending order
recommendationTable_df = recommendationTable_df.sort_values(ascending=False)
recommendationTable_df.head()

# Final recommendation table
movie_dff.loc[movie_dff['movieId'].isin(recommendationTable_df.head(20).keys())]

# adv and disadv of contenta -Based filtering
# adv
# Learn users preferences
# Highly personalized for the user.

# Disadv
# Doesn't take into account what other user think of the item, so low quality recommendation may happeen.
# Extracting the data is not always intuative
# Determining what characteristic of the item the users dislikes or likes is not always abvious.


### User based collaborative filtering (User- User filtering)
movie_dff.head()
# dropping the genres column
movie_dff = movie_dff.drop('genres',1)
movie_dff.head()
rating_dff.head()
rating_dff = rating_dff.drop('timestamp' , 1)
rating_dff.head()

# Filtering movie by title
inputId = movie_dff[movie_dff['title'].isin(inputMovies['title'].tolist())]