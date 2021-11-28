import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
# Loading the dataset.
movies = pd.read_csv("C:/Users/adm/Documents/Datasets/Movie Recommendation datasets/movie.csv")
rating = pd.read_csv("C:/Users/adm/Documents/Datasets/Movie Recommendation datasets/ratings.csv")

movies.head()
rating.head()

movies.shape
rating.shape
movies.describe()
rating.describe()

# making a list of list of movies
genres=[]
for genre in movies.genres:
    x = genre.split('|')
    for i in x:
         if i not in genres:
            genres.append(str(i))
genres = str(genres)
movie_title = []
movie_title = [title[0:-7] for title in movies.title]
movie_title = str(movie_title)

# Merging both dataframes.
df = pd.merge(rating,movies,how="left",on = "movieId")
df.head()
df.columns

# Creating a data frame of movies with the highest rating is descending order.
df1 = df.groupby(['title'])[['rating']].sum()
high_rated = df1.nlargest(20,'rating')
high_rated.head()

# Plotting the data
plt.figure(figsize=(30,10))
plt.title("Top 20 Movies with Highest Rating", fontsize = 40)
plt.ylabel('Rating', fontsize = 30)
plt.xlabel("Movie Title" fontsize = 30)
plt.xticks(fontsize = 25 ,rotation = 90)
plt.yticks(fontsize = 25)
plt.bar(high_rated.index,high_rated['rating'],linewidth = 3,color= 'skyblue')

# Creating dataframe of movies with highest number of rating in descending order.
df2 = df.groupby('title')[['rating']].count()
rating_count_20 = df2.nlargest(20,'rating')
rating_count_20.head()


tv  = TfidfVectorizer()
tfidf_matrix = tv.fit_transform(movies['genres'])

movie_user = df.pivot_table(index='userId',columns='title',values='rating')
movie_user.head()