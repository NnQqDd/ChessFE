import json
from flask import jsonify, Flask
from flask_cors import CORS
import numpy as np
import pandas as pd
import os
import logic

app = Flask(__name__)
CORS(app)

logic.initialize('')

def get_recommend_movie(userid, base):
    res = logic.recommend(userid, base = base)
    if res is None:
        response = {
            "recommendedMovieIds": [],
            "originalTitle": ''
        }
    response = {"recommendedMovieIds": res[0], "originalTitle": res[1]}
    return response

@app.route('/user/<int:userid>', methods=['GET'])
def api_get_user_information(userid):
    user = logic.get_user_information(userid)
    if user is None:
        return jsonify({'error': 'User not found'}), 404
    return jsonify(user)

@app.route('/movie/<int:movieid>', methods=['GET'])
def api_get_movie_information(movieid):
    movie = logic.get_movie_information(movieid)
    if movie is None:
        return jsonify({'error': 'Movie not found'}), 404
    return jsonify(movie)

@app.route('/recommend/<int:userid>', methods=['GET'])
def api_recommend(userid):
    result = get_recommend_movie(userid, base='ug')
    return jsonify(result)

@app.route('/similar/<int:userid>', methods=['GET'])
def api_similar_movies(userid):
    result = get_recommend_movie(userid, base='bert')
    return jsonify(result)

@app.route('/rate', methods=['POST'])
def rate_movie():
    data = request.get_json()
    user_id = data.get('userId')
    movie_id = data.get('movieId')
    rating = data.get('rating')
    rate(userid, movieid, rating)
    return jsonify({"message": "Rating submitted successfully!", "new_average_rating": movie["average_rating"]}), 200

if __name__ == '__main__':
    app.run(port=5000)
