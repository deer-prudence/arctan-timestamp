import pandas as pd 
import numpy as np
import random
import re
import pprint
    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(df2['overview'])
from surprise import Reader, Dataset, SVD, evaluate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt

# make list of length num_users with random, unique users
def random_users(num_users, total_users):
    random_users = list()
    while len(random_users) < num_users:
        random_user = random.randint(0, total_users)
        if random_user not in random_users:
            random_users.append(random_user)
    return random_users

def scale_score(list):
    score = list[1] * 5
    return [list[0], score]
            
def build_basic_collab_predictions():
    reader = Reader()
    ratings = pd.read_csv('/home/loudenem/MIP_real/ratings_small.csv', nrows = 20000)
    user_num = 20
    tested_user_info = list()
    all_user_list = ratings['userId'].tolist()
    random_user_list = random_users(user_num, all_user_list[-1])
    index_list = list()
    
    for user in random_user_list:
        all_users = np.array(all_user_list)
        index_vals = np.where(all_users == user)[0]
        if len(index_vals) > 10:
            take_indices = index_vals[-10:] # most recent
            user_data = ratings.iloc[take_indices]
            tested_user_info.append(user_data)
            index_list.append(take_indices)

    for item in index_list: # drop test data, leaving only training data
        for index_val in item:
            ratings = ratings.drop(index_val)

    tested_user_info = pd.DataFrame(np.concatenate(tested_user_info))
    tested_user_info.columns = ['userId', 'movieId', 'rating', 'timestamp']
    
        
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    data.split(n_folds=5)
    
    trainset = data.build_full_trainset()

    return trainset, tested_user_info, random_user_list
    
def test_collab_predictions(function):
    trainset, tested_user_info, random_user_list = (function)
    svd = SVD()
    svd.fit(trainset)
    reader = Reader()

    print("------ testing collaborative recommender system -------\n")
    for name in random_user_list:
        user_values = tested_user_info.loc[tested_user_info['userId'] == name]
        user_ratings = user_values['rating'].tolist()
        user_movies = user_values['movieId'].tolist()
        movie_predictions = list()
        for i in range(len(user_values)):
            pred = svd.predict(name, user_movies[i], 3)
            movie_predictions.append(pred[3])

        RMSE = sqrt(mean_squared_error(user_ratings, movie_predictions))
        MAE = mean_absolute_error(user_ratings, movie_predictions)
        print("for user " + str(name) + " the root mean square error was " + str(RMSE))
        print("for user " + str(name) + " the mean absolute error was " + str(MAE))
        print("\n\n")

def main():
    test_collab_predictions(build_basic_collab_predictions())

main()

# content, collab, and get_rec based on:
# https://www.kaggle.com/ibtesama/getting-started-with-a-
# movie-recommendation-system#Content-Based-Filtering
