import os

import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request, url_for

from poster_img_generator import generate_posters
from recomodel import cleandf, cont_movies, recommend_alike, recommend_movies

cleandf()

if os.path.exists("static/posters"):
    # The directory already exists, so do nothing.
    pass
else:
    # The directory doesn't exist, so create it and call a function.
    os.makedirs("static/posters")
    generate_posters()

app = Flask(__name__)

@app.route('/')
def index():
    df = pd.read_csv('ml-latest-small/movies_updated.csv')
    titles = df['title'].unique()
    return render_template('index.html', titles=titles)

@app.route('/search', methods=['POST'])
def search():
    # Get the userID entered by the user
    user_id = request.form['user_id']
    num_recommendations = int(request.form['num_recommendations'])

    # Redirect to the user page
    return redirect(f"/user/{user_id}/{num_recommendations}")


@app.route('/user/<int:user_id>/<int:num_recommendations>')
def user(user_id, num_recommendations=5):
    images = []
    history = []
    df = pd.read_csv('ml-latest-small/ratings.csv')
 
    if user_id not in df['userId'].values:
        return redirect(url_for('error'))
    already_rated, predictions = recommend_movies(user_id,num_recommendations=num_recommendations)

    for index, row in predictions.iterrows():
        image = row.to_dict()
        images.append(image)
        
    for index, row in already_rated.iterrows():
        image = row.to_dict()
        history.append(image)
    # print(num_recommendations)
    user_ids = df['userId'].unique()
    prev_user_id = user_ids[user_ids < user_id][-1] if user_ids[0] < user_id else None
    next_user_id = user_ids[user_ids > user_id][0] if user_ids[-1] > user_id else None

    return render_template('user.html', images=images, num_recommendations=num_recommendations, history=history, user_id=user_id, prev_user_id=prev_user_id, next_user_id=next_user_id)


@app.route('/search2', methods=['POST'])
def search2():
    # Get the movie title entered by the user
    movie_title = request.form['movie_title']

    # Redirect to the movie page
    return redirect(f"/movies/{movie_title}")

@app.route('/movies/<string:movie_title>')
def movies(movie_title):
    images = []

    df = pd.read_csv('ml-latest-small/movies_updated.csv')

    if movie_title not in df['title'].values:
        return redirect(url_for('error'))

    selected_title, predictions = recommend_alike(movie_title)

    for index, row in predictions.iterrows():
        image = row.to_dict()
        images.append(image)

    titles = df['title'].unique()

    # Find the index of the selected movie
    selected_movie_index = np.where(titles == movie_title)[0][0]

    # Find the previous and next movie titles
    prev_movie = titles[selected_movie_index - 1] if selected_movie_index > 0 else None
    next_movie = titles[selected_movie_index + 1] if selected_movie_index < len(titles) - 1 else None
    print(prev_movie)
    return render_template('movies.html', selected_title=selected_title, images=images,
                           movie_title=movie_title, titles=titles,
                           prev_movie=prev_movie, next_movie=next_movie)



@app.route('/user/')
def user_error():
    return redirect(url_for('nouser'))

@app.route('/nouser')
def nouser():
    return render_template('nouser.html')


@app.route('/error')
def error():
    return render_template('error.html')

if __name__ == '__main__':
    app.run(debug=True)