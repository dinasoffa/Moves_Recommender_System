import csv
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.neighbors import NearestNeighbors


def cont_movies():

    # Open the CSV file
    with open('ml-latest-small/movies_updated.csv', 'r') as file:
        reader = csv.reader(file)
        # Skip the header row
        header = next(reader)
        # Initialize the new movie ID
        new_movie_id = 1
        # Create a new list to store the updated rows
        updated_rows = [header]
        # Iterate over the rows in the CSV file
        for row in reader:
            # Replace the movie ID with the new ID
            row[0] = str(new_movie_id)
            # Append the updated row to the new list
            updated_rows.append(row)
            # Increment the new movie ID
            new_movie_id += 1

    # Write the updated rows to the CSV file
    with open('ml-latest-small/movies_updated.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(updated_rows)

def cleandf():
    # Load the CSV dataframe
    movies= pd.read_csv('ml-latest-small/movies_updated.csv')
    movies['title'] = movies['title'].apply(lambda x: re.sub(r'[{}|\\\^~\[\]`]', '', x))
    movies.to_csv('ml-latest-small/movies_updated.csv', index=False)

    
def recommend_movies(userId, num_recommendations=5):
    # cleandf()
    users= pd.read_csv('ml-latest-small/ratings.csv')
    movies= pd.read_csv('ml-latest-small/movies_updated.csv')
    df= pd.merge(users, movies, on= 'movieId')
    df.groupby('title').agg({'rating':'mean'}).sort_values(by='rating', ascending=False)
    df.groupby('title')['rating'].count().sort_values(ascending=False)
    df['title'] = df['title'].str.replace("'", "")

    ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
    ratings['rating_counts']= pd.DataFrame(df.groupby('title')['rating'].count())
    ratings=ratings.reset_index()

    movies_df= df.pivot_table(index="title",columns='userId',values='rating').fillna(0)
    movies_df_metrix= csr_matrix(movies_df.values)

    model_knn= NearestNeighbors(metric= 'cosine', algorithm='brute')

    # Fitting the model
    model_knn.fit(movies_df_metrix)

    ratings.sort_values('rating_counts', ascending=False)
    mtrx_df = users.pivot(index = 'userId', columns ='movieId', values = 'rating').fillna(0)
    mtrx = mtrx_df.to_numpy()
    ratings_mean = np.mean(mtrx, axis = 1)
    normalized_mtrx = mtrx - ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(normalized_mtrx, k = 50)
    sigma = np.diag(sigma)
    all_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_predicted_ratings, columns = mtrx_df.columns)


    
    # Get user id, keep in mind index starts from zero
    user_row_number = userId-1
    # Sort user's predictons
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False)
    # List movies user already rated
    user_data = users[users.userId == (userId)]
    user_rated = (user_data.merge(movies, how = 'left', left_on = 'movieId', right_on = 'movieId').
                  sort_values(['rating'], ascending=False)
                 )

    # f'User {userId} has already rated {user_rated.shape[0]} films.'

    recommendations = (movies[~movies['movieId'].isin(user_rated['movieId'])].
         merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
               left_on = 'movieId',
               right_on = 'movieId').
               rename(columns = {user_row_number: 'Predictions'}).
               sort_values('Predictions', ascending = False).
               iloc[:num_recommendations, :-1]
                      )
    
    joined = pd.merge(recommendations, users, on='movieId')
    ratings = joined.groupby('movieId')['rating'].mean().reset_index()
    ratings['rating'] = ratings['rating'].round(1)

    # merge the ratings dataframe with the predictions dataframe
    recommendations = pd.merge(recommendations, ratings, on='movieId')

    return user_rated, recommendations


def recommend_alike(selected_movie_name):
    # cleandf()
    users = pd.read_csv('ml-latest-small/ratings.csv')
    movies = pd.read_csv('ml-latest-small/movies_updated.csv')
    df = pd.merge(users, movies, on='movieId')
    df.groupby('title').agg({'rating': 'mean'}).sort_values(by='rating', ascending=False)
    df.groupby('title')['rating'].count().sort_values(ascending=False)
    df['title'] = df['title'].str.replace("'", "")
    ratings = pd.DataFrame(df.groupby('title')['rating'].mean())
    ratings['rating_counts'] = pd.DataFrame(df.groupby('title')['rating'].count())
    ratings = ratings.reset_index()
    movies_df = df.pivot_table(index="title", columns='userId', values='rating').fillna(0)
    movies_df_matrix = csr_matrix(movies_df.values)

    # Fit the model
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(movies_df_matrix)

    # Find the index of the selected movie
    selected_movie_index = movies_df.index.get_loc(selected_movie_name)

    # Use the selected movie index in the code
    distances, indices = model_knn.kneighbors(movies_df.iloc[selected_movie_index, :].values.reshape(1, -1), n_neighbors=6)
    similar_movies = pd.DataFrame({'Recommended Movies': movies_df.index[indices.flatten()[1:]]})
    similar_movies = pd.merge(similar_movies, movies, left_on='Recommended Movies', right_on='title', how='inner')

    # Select the relevant columns
    similar_movies = similar_movies[['movieId', 'title', 'genres', 'poster_dest']]

    joined = pd.merge(similar_movies, users, on='movieId')
    ratings = joined.groupby('movieId')['rating'].mean().reset_index()
    ratings['rating'] = ratings['rating'].round(1)

    # merge the ratings dataframe with the predictions dataframe
    similar_movies = pd.merge(similar_movies, ratings, on='movieId')

    return selected_movie_name, similar_movies