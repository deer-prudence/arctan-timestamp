import pandas as pd 
import numpy as np
import random
import re
import pprint
import scipy
    #pp = pprint.PrettyPrinter(indent=4)
    #pp.pprint(df2['overview'])
from surprise import Reader, Dataset, SVD, evaluate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from scipy.spatial.distance import pdist,squareform
from scipy import spatial
# take first element for sort

def take_first(element):
    return element[0]

# take second element for sort
def take_second(element):
    if element[1] >= 4.0: # must have a similarity score of 4.0 or similar
        return element[1]

# this should find the three most similar users and order them from most similar to third most similar
def order_list(list_to_order, index_function, desired_elements):
        
    list_to_order.sort(key = index_function, reverse = True) # sort users from most to least similar
    list_length = len(list_to_order)
    
    if list_length >= desired_elements:
        top_values = list_to_order[:desired_elements]
    else:
        top_values = list_to_order[:(list_length)]

    final_list = list()
    for item in top_values:
        if item[0] != 0: # if the similarity score isn't 0; 0 will multiply to 0 anyway
            final_list.append([item[0], item[1]])
        
    return final_list

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

def cosine_weight(user1_data, user2_data, all_movies):
    similarity_matrix = [[None for x in range(len(all_movies))] for y in range(len(all_movies))]
    user1_list = [0 for x in range(len(all_movies))]
    user2_list = [0 for x in range(len(all_movies))]
    
    movie1 = list(user1_data['movieId'])
    rating1 = list(user1_data['rating'])
    movie2 = list(user2_data['movieId'])
    rating2 = list(user2_data['rating'])
    
    for i in range(len(all_movies)):
        if all_movies[i] in movie1:
            movie = all_movies[i]
            rating = rating1[movie1.index(movie)]
            user1_list[i] = rating 

    for i in range(len(all_movies)):
        if all_movies[i] in movie2:
            movie = all_movies[i]
            rating = rating2[movie2.index(movie)]
            user2_list[i] = rating
    
    cosine_sim = 1 - spatial.distance.cosine(user1_list, user2_list)
    # We take the mean of user ratings and subtract that mean from all individual ratings divided by the total number of ratings by user
    # add centered cosine??? 
    return cosine_sim

def build_cosine_matrix_from_file():
    ratings = pd.read_csv('/home/loudenem/MIP_real/ratings_small.csv', nrows = 2000)
    all_users = list(dict.fromkeys(ratings['userId'].tolist()))
    all_movies = list(dict.fromkeys(ratings['movieId'].tolist()))
    num_users = len(all_users)
    cosine_similarity_matrix = [[None for x in range(num_users)] for y in range(num_users)]
    for user in all_users:
        for other_user in all_users:
            user_data = ratings.loc[ratings['userId'] == user]
            other_user_data = ratings.loc[ratings['userId'] == other_user]
            cosine_sim = cosine_weight(user_data, other_user_data, all_movies)
            cosine_similarity_matrix[all_users.index(user)][all_users.index(other_user)] = cosine_sim
    return cosine_similarity_matrix

def build_cosine_matrix(ratings):
    all_users = list(dict.fromkeys(ratings['userId'].tolist()))
    all_movies = list(dict.fromkeys(ratings['movieId'].tolist()))
    num_users = len(all_users)
    cosine_similarity_matrix = [[None for x in range(num_users)] for y in range(num_users)]
    for user in all_users:
        for other_user in all_users:
            user_data = ratings.loc[ratings['userId'] == user]
            other_user_data = ratings.loc[ratings['userId'] == other_user]
            cosine_sim = cosine_weight(user_data, other_user_data, all_movies)
            cosine_similarity_matrix[all_users.index(user)][all_users.index(other_user)] = cosine_sim
    return cosine_similarity_matrix, all_users

def build_collab_predictions():
    reader = Reader()
    ratings = pd.read_csv('/home/loudenem/MIP_real/ratings_small.csv', nrows = 2000)
    user_num = 10
    tested_user_info = list()
    all_user_list = ratings['userId'].tolist()
    random_user_list = random_users(user_num, all_user_list[-1])
    index_list = list()
    for user in random_user_list:
        all_users = np.array(all_user_list)
        index_vals = np.where(all_users == user)[0]
        if len(index_vals) > 10:
            take_indices = index_vals[-10:] # most recent
            user_data = ratings[(take_indices[0]):(take_indices[-1] + 1)]
            user_data = user_data.values.tolist()
            tested_user_info.append(user_data)
            index_list.append(take_indices)

    for item in index_list: # drop test data, leaving only training data
        for index_val in item:
            ratings = ratings.drop(index_val)

    return ratings, tested_user_info, random_user_list



def predict_rating(sim_matrix, trainset, user, movie, all_users):
    matching_movies = trainset.loc[trainset['movieId'] == movie]
    if matching_movies.empty:
        return 0
    else:
        users = matching_movies['userId'].tolist()
        sim_score_list = list()
        for other_user in users:
            row_data = matching_movies.loc[matching_movies['userId'] == other_user]
            rating = float(row_data['rating'])
            index_val = all_users.index(other_user)
            sim_score = float(sim_matrix[all_users.index(user)][index_val])
            sim_score_list.append([sim_score, rating])
        top_ratings = order_list(sim_score_list, take_first, 10)

        if top_ratings == []:
            return 0
        else:
            numerator = 0
            denominator = 0
            for element in top_ratings:
                numerator += element[0] * element[1]
                denominator += element[0]
            return numerator/denominator
    
def test_collab_predictions():
    trainset, tested_user_info, random_user_list = build_collab_predictions()
    cosine_similarity_matrix, all_users = build_cosine_matrix(trainset)
    

    print("------ testing collaborative recommender system -------\n")
    for name in random_user_list:
        user_ratings = list()
        user_movies = list()
        for bigger_row in tested_user_info:
            for row in bigger_row:
                if int(row[0]) == name:
                    user_ratings.append(row[2])
                    user_movies.append(row[1])
        movie_predictions = list()
        for i in range(len(user_ratings)):
            pred = predict_rating(cosine_similarity_matrix, trainset, name, user_movies[i], all_users)
            movie_predictions.append(pred)
        print(movie_predictions)
        print(user_ratings)
        
        RMSE = sqrt(mean_squared_error(user_ratings, movie_predictions))
        MAE = mean_absolute_error(user_ratings, movie_predictions)
        print("for user " + str(name) + " the root mean square error was " + str(RMSE))
        print("for user " + str(name) + " the mean absolute error was " + str(MAE))
        print("\n")
        

def main():
    test_collab_predictions()

main()
