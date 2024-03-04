import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
#import seaborn as sns
import warnings
import os 
from scipy.sparse import csr_matrix
warnings.simplefilter(action='ignore', category=FutureWarning)

ratings = "rating.csv"
movies = "movie.csv"

dataset_path = "/Users/matteo/Desktop/Recommender-System/movielens-20m-dataset/"

movies_df = pd.read_csv(os.path.join(dataset_path, movies), usecols=["movieId", "title"], dtype={"movieId": "int32", "title":"str"})
ratings_df = pd.read_csv(os.path.join(dataset_path, ratings), usecols=["userId", "movieId", "rating"], dtype={"userId":"int32", "movieId":"int32", "rating":"float32"})


n_ratings = len(ratings_df)
n_movies = len(ratings_df['movieId'].unique())
n_users = len(ratings_df['userId'].unique())

print("numero di ratings: {}".format(n_ratings))
print("numero di film: {}".format(n_movies))
print("numero di users: {}".format(n_users))
print(f"Average ratings per user: {round(n_ratings/n_users, 2)}")
print(f"Average ratings per movie: {round(n_ratings/n_movies, 2)}")

user_freq = ratings_df[['userId', 'movieId']].groupby('userId').count().reset_index()
user_freq.columns = ['userId', 'n_ratings']


# trova il film con il rate piu alto e quello con il rate più basso
mean_rating = ratings_df.groupby('movieId')[['rating']].mean()
# film con rate piu basso
lowest_rated = mean_rating['rating'].idxmin()
movies_df.loc[movies_df['movieId'] == lowest_rated]
# film con rate piu alto 
highest_rated = mean_rating['rating'].idxmax()
movies_df.loc[movies_df['movieId'] == highest_rated]
# mostra il numero di persone che hanno valutato il film con il punteggio piu alto 
ratings_df[ratings_df['movieId']==highest_rated]
# mostra il numero di persone che hanno valutato il film con il punteggio piu basso
ratings_df[ratings_df['movieId']==lowest_rated]
 
# media del punteggio  
movie_stats = ratings_df.groupby('movieId')[['rating']].agg(['count', 'mean'])
movie_stats.columns = movie_stats.columns.droplevel()

# creazione matrice user-item usando scipy csr matrix MxN: M = film distinti, N = user distinti 
def create_matrix(df):
     
    N = len(df['userId'].unique())
    M = len(df['movieId'].unique())
     
    # Mappa gli id agli indici 
    user_mapper = dict(zip(np.unique(df["userId"]), list(range(N))))
    movie_mapper = dict(zip(np.unique(df["movieId"]), list(range(M))))
     
    # mappa gli indici agli id 
    user_inv_mapper = dict(zip(list(range(N)), np.unique(df["userId"])))
    movie_inv_mapper = dict(zip(list(range(M)), np.unique(df["movieId"])))
     
    user_index = [user_mapper[i] for i in df['userId']]
    movie_index = [movie_mapper[i] for i in df['movieId']]
 
    X = csr_matrix((df["rating"], (movie_index, user_index)), shape=(M, N))
     
    return X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper
     





"""
Trovo i film simili usando KNN: movie id, matrix, numero neighbors da considerare, metrica cosine similarity 
"""
def find_similar_movies(movie_id, X, k, movie_mapper, movie_inv_mapper, metric='cosine', show_distance=False):
     
    neighbour_ids = []
     
    movie_ind = movie_mapper[movie_id]
    movie_vec = X[movie_ind]
    k+=1
    kNN = NearestNeighbors(n_neighbors=k, algorithm="brute", metric=metric)
    kNN.fit(X)
    movie_vec = movie_vec.reshape(1,-1)
    neighbour = kNN.kneighbors(movie_vec, return_distance=show_distance)
    for i in range(0,k):
        n = neighbour.item(i)
        neighbour_ids.append(movie_inv_mapper[n])
    neighbour_ids.pop(0)
    return neighbour_ids
 

# funizione per raccomandare film sulla base delle preferenze dello user: k = numero film consigliati 
def recommend_movies_for_user(user_id, X, movie_mapper, movie_inv_mapper, k):
    df1 = ratings_df[ratings_df['userId'] == user_id]
     
    if df1.empty:
        print(f"User with ID {user_id} does not exist.")
        return
 
    movie_id = df1[df1['rating'] == max(df1['rating'])]['movieId'].iloc[0]
 
    movie_titles = dict(zip(movies_df['movieId'], movies_df['title']))
 
    similar_ids = find_similar_movies(movie_id, X, k, movie_mapper, movie_inv_mapper)

    # controllo se tra i film simili trovati ce n'è qualcuno già visto dallo user
    user_movies = ratings_df.loc[ratings_df['userId']==user_id, 'movieId'].tolist()
    for f in similar_ids:
        if f in user_movies:
            similar_ids.remove(f)

    movie_title = movie_titles.get(movie_id, "Movie not found")
 
    if movie_title == "Movie not found":
        print(f"Movie with ID {movie_id} not found.")
        return
 
    print(f"Since you watched {movie_title}, you might also like:")
    for i in similar_ids:
        print(movie_titles.get(i, "Movie not found"))


X, user_mapper, movie_mapper, user_inv_mapper, movie_inv_mapper = create_matrix(ratings_df)



user_id = 150  # user scelto per la recommendation 
recommend_movies_for_user(user_id, X, movie_mapper, movie_inv_mapper, k=10)

print("---- verifico film già visti dallo user {} -------".format(user_id))
movie_user_list = ratings_df.loc[ratings_df['userId']==user_id, 'movieId'].tolist()
title_movie_user_list = []
for m in movie_user_list:
    title_movie_user_list.append((movies_df.loc[movies_df['movieId']==m, 'title'].tolist())[0])
print(title_movie_user_list)

