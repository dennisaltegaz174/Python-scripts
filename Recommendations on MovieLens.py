import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

movie_dff = pd.read_csv("C:\\Users\\adm\\Documents\\Datasets\\Recommendations on MovieLens\\movie.csv")
rating_dff = pd.read_csv("C:\\Users\\adm\\Documents\\Datasets\\Recommendations on MovieLens\\rating.csv")

movie_dff.shape

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

userInput = [
                {'title':'Breakfast  Club, The','rating':5},
               {'title':'Toy Story', 'rating':3.5},
               {'title': 'Jumanji', 'rating' : 2},
               {'title': 'Pulp Fiction', 'rating' : 5},
               {'title': 'Akira' , 'rating' : 4.5}
                ]
inputMovies = pd.DataFrame(userInput)u
inputMovies
# Filtering movie by title
inputId = movie_dff[movie_dff['title'].isin(inputMovies['title'].tolist())]
# Then merging it  so as  to get movieId. its implicitly merging it by title.
inputMovies = pd.merge(inputId,inputMovies)
# Dropping information we  won't use from the imput dataframe
inputMovies = inputMovies.drop('year',1)
# Final input dataframe
inputMovies

# Similar users
# filtering out users that have watched movies that the input has watched and storing it
userSubset = rating_dff[rating_dff['movieId'].isin(inputMovies['movieId'].tolist())]
userSubset.head()

# Groupby creates several sub dataframes where they all have same value in the column specified as the parameter.
userSubsetGroup = userSubset.groupby(['userId'])
# Looking at one of the users e.g userID = 5
userSubsetGroup.get_group(5)
userSubsetGroup.head()

# sorting it so that  users  with movie most in common with input will have priority.
userSubsetGroup = sorted(userSubsetGroup,key=lambda x:len(x[1]),reverse=True)
# checking the first 3
userSubsetGroup[0:3]

# comparing users
# Using a subset of  top 10 users
userSubsetGroup = userSubsetGroup[0:100]
# store the pearson correlation in a  dictonary where the key  is the user id and the value is the coefficient.
pearsonCorrelationDict =  {}
# for  every user group in our subset
for name, group in userSubsetGroup:
    #starting by sorting input and current user group so that the values are not mixed up later on
group =  group.sort_values(by='movieId')
    inputMovies = inputMovies.sort_values(by = 'movieId')
    # Getting the  N for the formula
    nRatings = len(group)
    # Getting the review scores for the movies that they  have in common
    temp_df = inputMovies[inputMovies['movieId'].isin(group['movieId'].tolist())]
    # Then store then in a temporary buffer variable in a list format to facilitate future calculation
    tempRatingList = temp_df['rating'].tolist()
    # Putting the  current user groups reviews in a list format
    tempGroupList = group['rating'].tolist()
    # now calculating the pearson correlation between two users so called x and y
    Sxx = sum([1**2 for i in tempRatingList]) - pow(sum(tempGroupList),2)/float(nRatings)
        Sxy = sum(i*j for i , j in zip (tempRatingList,tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)