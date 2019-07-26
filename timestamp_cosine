import pandas as pd 
import numpy as np
import random
import re
import scipy
from surprise import Reader, Dataset, SVD, evaluate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from scipy.spatial.distance import pdist,squareform
from scipy import spatial

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
        random_user = random.randint(1, total_users)
        if random_user not in random_users:
            random_users.append(random_user)
    return random_users

def basic_cosine_timestamp_weight(user1_data, user2_data, all_movies):
    similarity_matrix = [[None for x in range(len(all_movies))] for y in range(len(all_movies))]
    user1_list = [0 for x in range(len(all_movies))]
    user2_list = [0 for x in range(len(all_movies))]
    
    movie1 = list(user1_data['movieId'])
    rating1 = list(user1_data['rating'])
    timestamp1 = list(user1_data['timestamp'])
    movie2 = list(user2_data['movieId'])
    rating2 = list(user2_data['rating'])
    timestamp2 = list(user2_data['timestamp'])
    
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
    
    timestamp1.sort()
    user1_oldest = timestamp1[0]
    user1_newest = timestamp
    timestamp2.sort()
    user2_oldest = timestamp2[0]
    user2_newest = timestamp2[-1]

    start_difference = abs(user1_oldest - user2_oldest) * 0.000001
    end_difference = abs(user1_newest - user2_newest) * 0.000001

    try:
        cosine_tan_sim = (np.arctan(1.4 / end_difference)) + cosine_sim # https://www.desmos.com/calculator/u6t1lqkqum
    except ZeroDivisionError:
        print("ZeroDivisionError")
        cosine_tan_sim = 0
    return cosine_tan_sim

def build_cosine_matrix(ratings):
    all_users = list(dict.fromkeys(ratings['userId'].tolist()))
    all_movies = list(dict.fromkeys(ratings['movieId'].tolist()))
    num_users = len(all_users)
    cosine_similarity_matrix = [[None for x in range(num_users)] for y in range(num_users)]
    for user in all_users:
        for other_user in all_users:
            user_data = ratings.loc[ratings['userId'] == user]
            other_user_data = ratings.loc[ratings['userId'] == other_user]
            cosine_sim = basic_cosine_timestamp_weight(user_data, other_user_data, all_movies)
            cosine_similarity_matrix[all_users.index(user)][all_users.index(other_user)] = cosine_sim
    return cosine_similarity_matrix, all_users

def build_collab_predictions():
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
            sim_score = float(sim_matrix[all_users.index(user)][index_val]) #https://ieeexplore.ieee.org/document/7917109/authors#authors
            sim_score_list.append([sim_score, rating])
        top_ratings = order_list(sim_score_list, lambda x: x[0], 10)

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
    totalRMSE = 0
    totalMAE = 0
    total_tested = 0 # to find average 

    max_RMSE = 0 # range of values
    min_RMSE = 0
    max_MAE = 0
    min_MAE = 0
    
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
        if len(movie_predictions) == 0:
            print("user " + str(name) + " has no predicions")
        else:
            RMSE = sqrt(mean_squared_error(user_ratings, movie_predictions))
            if RMSE > max_RMSE:
                max_RMSE = RMSE
            if RMSE > min_RMSE:
                min_RMSE = RMSE
                
            totalRMSE += RMSE
            MAE = mean_absolute_error(user_ratings, movie_predictions)
            
            if MAE > max_MAE:
                max_MAE = MAE
            if MAE > min_MAE:
                min_MAE = MAE
            totalMAE += MAE
            total_tested += 1
            print("for user " + str(name) + " the root mean square error was " + str(RMSE))
            print("for user " + str(name) + " the mean absolute error was " + str(MAE))
            print("\n")
    print("average RMSE = " + str(totalRMSE/total_tested))
    print("average MAE = " +str(totalMAE/total_tested))
        
        

def main(): 
    test_collab_predictions()

main()
