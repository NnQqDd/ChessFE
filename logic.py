import pandas as pd
import numpy as np
import faiss
import os

FILE_PATH = ''
MEDIUM = ['ratings_medium.csv', 'movies_medium.csv']
UG_EMBEDDING_MAP_FILE = 'movie_ug_embeddings.csv'
BERT_EMBEDDING_MAP_FILE = 'movie_bert_embeddings.csv'
BERT_EMBEDING_MAP = None
UG_EMBEDDING_MAP = None
BERT_INDEX = None
UG_INDEX = None
INDEX_TO_ID = None
df_rating = None
df_movie = None

def initialize(file_path):
  global df_movie, df_rating
  global FILE_PATH, BERT_EMBEDDING_MAP, UG_EMBEDDING_MAP
  global INDEX_TO_ID, BERT_INDEX, UG_INDEX
  FILE_PATH = file_path
  print('Reading the movie dataset...')
  df_rating = pd.read_csv(os.path.join(FILE_PATH, MEDIUM[0]))
  df_movie = pd.read_csv(os.path.join(FILE_PATH, os.path.join(FILE_PATH, MEDIUM[1])))
  print('Reading and converting the embeddings...')
  DF_UG_EMBEDDING_MAP = pd.read_csv(os.path.join(FILE_PATH, UG_EMBEDDING_MAP_FILE))
  UG_EMBEDDING_MAP = DF_UG_EMBEDDING_MAP.set_index('movieId')['embedding'].to_dict()
  UG_EMBEDDING_MAP = {k: np.array(eval(v)) for k, v in UG_EMBEDDING_MAP.items()}
  INDEX_TO_ID = {index: int(movieId) for index, movieId in enumerate(DF_UG_EMBEDDING_MAP['movieId'])}
  DF_BERT_EMBEDDING_MAP = pd.read_csv(os.path.join(FILE_PATH, BERT_EMBEDDING_MAP_FILE))
  BERT_EMBEDDING_MAP = DF_BERT_EMBEDDING_MAP.set_index('movieId')['embedding'].to_dict()
  BERT_EMBEDDING_MAP = {k: np.array(eval(v)) for k, v in BERT_EMBEDDING_MAP.items()}
  print('Creating indexes...')
  embeddings = np.array([v for _, v in BERT_EMBEDDING_MAP.items()])
  BERT_INDEX = faiss.IndexFlatL2(embeddings.shape[1])
  BERT_INDEX.add(embeddings)
  embeddings = np.array([v for _, v in UG_EMBEDDING_MAP.items()])
  UG_INDEX = faiss.IndexFlatL2(embeddings.shape[1])
  UG_INDEX.add(embeddings)
  # COURSES = [course for course in COURSES]
  print('Calculate average ratings...')
  rating_summary = df_rating.groupby('movieId').agg(
      average_rating=('rating', 'mean'),
      rating_count=('rating', 'count')
  ).reset_index()
  df_movie = df_movie.merge(rating_summary, on='movieId', how='left')
  df_movie['average_rating'] = df_movie['average_rating'].fillna(0)
  df_movie['rating_count'] = df_movie['rating_count'].fillna(0).astype(int)
  print('Done')

def get_user_information(userid):
    user_ratings = df_rating[df_rating['userId'] == userid]
    if user_ratings.empty:
      return None
    merged_data = user_ratings.merge(df_movie, on='movieId', how='inner')
    movies = []
    for _, row in merged_data.iterrows():
        movie_info = {
            "movieid": row['movieId'],
            "title": row['title'],
            "user_rating": row['rating']
        }
        movies.append(movie_info)
    return {"userid": userid, "movies": movies}

def get_movie_information(movieid):
    movie_info = df_movie[df_movie['movieId'] == movieid]
    if movie_info.empty:
        return None
    movie_info = movie_info.to_dict(orient='records')[0]
    movie_info['movieid'] = movie_info.pop('movieId')
    movie_info['genres'] = movie_info['genres'].split('|')
    return movie_info

def softmax(x):
  e_x = np.exp(x - np.max(x))
  return e_x / e_x.sum(axis=0)

def get_user_movies(userid):
    try:
      movies = get_user_information(userid)['movies']
    except:
      return None
    movies = sorted(movies, key=lambda x: x['user_rating'], reverse=True)
    all_ids = set()
    for movie in movies:
        all_ids.add(movie['movieid'])
    movie_ids = []
    movie_titles = []
    movie_prob = []
    for i in range(len(movies)):
        if movies[i]['user_rating'] > 3:
            movie_ids.append(movies[i]['movieid'])
            movie_titles.append(movies[i]['title'])
            movie_prob.append(movies[i]['user_rating'])
        else:
            break
    if len(movie_ids) == 0:
        return [], ''
    movie_prob = softmax(movie_prob)
    index = int(np.random.choice(np.arange(len(movie_ids)), p=movie_prob))
    src_title = movie_titles[index]
    src_id = movie_ids[index]
    return set(movie_ids), src_id, src_title

def recommend(userid, amt=10, base='ug'):
  movies, movieid, title = get_user_movies(userid)
  print(movieid, title)
  if base == 'ug':
    index = UG_INDEX
    embed_map = UG_EMBEDDING_MAP
  else:
    index = BERT_INDEX
    embed_map = BERT_EMBEDDING_MAP
  _, indices = index.search(embed_map[movieid].reshape(1, -1), 2*amt + 1)
  ids = [INDEX_TO_ID[i] for i in indices[0]]
  ids = [movieid for movieid in ids if movieid not in movies]
  return ids[1:amt + 1], title

def rate(userid, movieid, rating):
  index = (df_rating['userId'] == userid) & (df_rating['movieId'] == movieid)
  if index.any():
    old_rating = df_rating.loc[index, 'rating']
    df_rating.loc[index, 'rating'] = rating
    index = (df_movie['movieId'] == movieid)
    count = df_movie.loc[index, 'rating_count']
    average_rating = df_movie.loc[index, 'average_rating']
    df_movie.loc[index, 'average_rating'] = (average_rating * count - old_rating + rating) / count
  else:
    df_rating = pd.concat([df_rating, pd.DataFrame([{'userId': userid, 'movieId': movieid, 'rating': rating}])], ignore_index=True)
    index = (df_movie['movieId'] == movieid)
    count = df_movie.loc[index, 'rating_count']
    average_rating = df_movie.loc[index, 'average_rating']
    df_movie.loc[index, 'average_rating'] = (average_rating * count + rating) / (count + 1)
