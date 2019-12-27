import numpy as np
from numpy import linalg as la
from numpy.linalg import svd
import math
import operator
import time


# Function to get top k movies

# This basically sorts movies by average rating of movies , acc to the entries of the input matrix 'temp' and return top k....

def get_top_k_movies(temp, k):
    movie_index_rating = []
    top_k_movies_for_temp = []
    avg_rating_of_movie = np.zeros(len(temp[0]))
    for j in range(len(temp[0])):
        number_of_users_rated = 0
        num = 0
        for i in range(len(temp)):
            if(temp[i][j] != 0):
                number_of_users_rated += 1
                num += temp[i][j]
        if(number_of_users_rated > 0):
            avg_rating_of_movie[j] = float(num) / number_of_users_rated
            movie_index_rating.append([j, avg_rating_of_movie[j]])

    sorted_movie_index_rating = sorted(movie_index_rating, key = operator.itemgetter(1), reverse = True)

    for i, index in zip(range(k), range(len(sorted_movie_index_rating))):
        top_k_movies_for_temp.append(sorted_movie_index_rating[i][0])

    return top_k_movies_for_temp


# Similarity function
def find_similarity(X, Y):
    numerator = 0.0
    sum_of_square_of_components_of_X = 0.0
    sum_of_square_of_components_of_Y = 0.0
    
    for i in range(len(X)):
        numerator += X[i] * Y[i]
        sum_of_square_of_components_of_X += X[i] ** 2
        sum_of_square_of_components_of_Y += Y[i] ** 2

    denomenator = math.sqrt(sum_of_square_of_components_of_X) * math.sqrt(sum_of_square_of_components_of_Y)
    if(denomenator == 0):
        return 0
    else:
        return float(numerator) / denomenator


# Collaborative filtering function
def collaborative_filtering_func(AT, BT, no_of_neighbors, movies_rated_by_user, to_be_predicted, temp, k, top_k_movies_for_B, baseline_approach):

    print("In collaborative filtering function!")
    movie_offset = np.zeros(len(AT)) #array[num of moviessssssss]
    mean_movie_rating = 0.0
    total_rating = 0.0
    number_of_ratings = 0

    # Finding mean movie rating throughout matrix
    for i in range(len(AT)):
        for j in range(len(AT[i])):
            if(AT[i][j] != 0):
                total_rating += AT[i][j]
                number_of_ratings += 1
    mean_movie_rating = float(total_rating) / number_of_ratings

    rating_deviation_of_user = np.zeros(len(AT[0]))
    rating_deviation_of_movie = np.zeros(len(AT))
    print("hi")


    # Rating deviation of each user
    # Given by: Average rating of user - mean movie rating
    for j in range(len(AT[0])):
        num = 0.0
        number_of_movies_rated = 0
        for i in range(len(AT)):
            if(AT[i][j] != 0):
                num += AT[i][j]
                number_of_movies_rated += 1
        if(number_of_movies_rated > 0):
            rating_deviation_of_user[j] = (float(num) / number_of_movies_rated) - mean_movie_rating


    # Rating deviation of each movie
    # Given by: Average rating of movie - mean movie rating
    for i in range(len(AT)):
        num = 0.0
        number_of_users_rated = 0
        for j in range(len(AT[i])):
            if(AT[i][j] != 0):
                num += AT[i][j]
                number_of_users_rated += 1
        if(number_of_users_rated > 0):
            rating_deviation_of_movie[i] = (float(num) / number_of_users_rated) - mean_movie_rating


    # Normalizing rows of AT here(MEAN centring..)
    for i in range(len(AT)):
        num = 0.0
        no_of_users_rated_current_movie = 0
        for j in range(len(AT[i])):
            if (AT[i][j] != 0):
                num += AT[i][j]
                no_of_users_rated_current_movie += 1
        if(no_of_users_rated_current_movie > 0):
            movie_offset[i] = float(num / float(no_of_users_rated_current_movie))
        for j in range(len(AT[i])):
            if(AT[i][j] != 0):
                AT[i][j] = AT[i][j] - movie_offset[i]


    number_of_predictions = 0
    sum_of_squared_error = 0.0
    absolute_error_sum = 0.0
    count = 0
    
    # Predicting the ratings here
    for data in to_be_predicted:
        # data is of the form [movie, user]
        if(count == int(0.25 * len(to_be_predicted))):
            print("25% data predicted!")
        elif(count == int(0.5 * len(to_be_predicted))):
            print("50% data predicted!")
        elif(count == int(0.75 * len(to_be_predicted))):
            print("75% data predicted!")
    
        count += 1
        sim = []

        try:
            for movie in movies_rated_by_user[data[1]]:
                sim.append([movie, find_similarity(AT[data[0]], AT[movie])])
        except KeyError:
            continue
        
        sorted_sim = sorted(sim, key = operator.itemgetter(1), reverse = True)
        numerator = 0
        denomenator = 0
        for l, i in zip(range(no_of_neighbors), range(len(sorted_sim))):
            if(baseline_approach == True):
                numerator += sorted_sim[l][1] * (BT[sorted_sim[l][0]][data[1]] - (mean_movie_rating + rating_deviation_of_user[data[1]] + rating_deviation_of_movie[sorted_sim[l][0]]))
            else:
                numerator += sorted_sim[l][1] * (BT[sorted_sim[l][0]][data[1]] - movie_offset[sorted_sim[l][0]])
            denomenator += sorted_sim[l][1]
        if(denomenator > 0):
            if(baseline_approach == True):
                rating = mean_movie_rating + rating_deviation_of_user[data[1]] + rating_deviation_of_movie[data[0]] + (numerator / float(denomenator))
            else:
                rating = (numerator / float(denomenator)) + movie_offset[data[0]]		 #MEAN noncentring ...
            sum_of_squared_error += (rating - BT[data[0]][data[1]]) ** 2
            absolute_error_sum += abs(rating - BT[data[0]][data[1]])
            temp[data[1]][data[0]] = rating
            number_of_predictions += 1

    
    # Root mean square
    rmse = math.sqrt(sum_of_squared_error) / float(math.sqrt(number_of_predictions))

    rmse = math.sqrt(sum_of_squared_error) / float((number_of_predictions))


    # Mean Absolute Error
    mae = float(absolute_error_sum/float(number_of_predictions))

    n = len(to_be_predicted)

    # Spearman Coorelation
    spearman_rank_correlation = 1 - ((6 * sum_of_squared_error) / (n * (n*n - 1)))
    
    count = 0
    
    top_k_movies_for_temp = get_top_k_movies(temp, k)
    for movie in top_k_movies_for_B:
        if(movie in top_k_movies_for_temp):
            count += 1
    print("count: " + str(count))
    print("k: " + str(k))
    precision_on_top_k = float(count) / k


    # Printing the results
    if(baseline_approach):
        print("RMSE for Collaborative filtering with baseline approach: " + str(rmse))
        print("MAE for Collaborative filtering with baseline approach: " + str(mae))
        print("Precision on top k for Collaborative filtering with baseline approach: " + str(precision_on_top_k))
    else:
        print("RMSE for Collaborative filtering without baseline approach: " + str(rmse))
        print("MAE for Collaborative filtering without baseline approach: " + str(mae))
        print("Precision on top k for Collaborative filtering without baseline approach: " + str(precision_on_top_k))	

    print("Exiting collaborative filtering function!")
    return

user_ids_index = {}
movie_ids_index = {}
user_count = 0
movie_count = 0
count = 0
max_user_no = 0
max_movie_no = 0
movies_rated_by_user = {}
to_be_predicted = []
k = 50
r = 300

# Reading file for finding max movie id and max user id
with open("ratings.dat", "r") as data_file:
    for line in data_file:
        count += 1
        line_values = line.split("::")
        a = int(line_values[0])
        b = int(line_values[1])
        if(a > max_user_no):
            max_user_no = a
        if(b > max_movie_no):
            max_movie_no = b

three_fourth_data_length = int(0.25 * count)
counter = 0
count_thousand_data_points = 0
A = np.zeros((max_user_no + 1, max_movie_no + 1))
temper = np.zeros((max_user_no + 1, max_movie_no + 1))
B = np.zeros((max_user_no + 1, max_movie_no + 1))


# Reading file
with open("ratings.dat", "r") as data_file:
    for line in data_file:
        line_values = line.split("::")
        a = int(line_values[0])
        b = int(line_values[1])
        B[a][b] = float(line_values[2])
        if(counter <= three_fourth_data_length):
            A[a][b] = float(line_values[2])
            temper[a][b] = float(line_values[2])
            counter += 1
            if a not in movies_rated_by_user:
                movies_rated_by_user[a] = [b]
            else:
                movies_rated_by_user[a].append(b)

        #making the test data set.. NEED TO CHANGE THIS.. should be randomized....

        elif(count_thousand_data_points < 1200):
            to_be_predicted.append([b, a])
            count_thousand_data_points += 1

data_file.close()


# Getting top k rated movies for B
top_k_movies_for_B = get_top_k_movies(B, k)


no_of_neighbors = 5
temp = temper.copy()
start_time = time.time()

# Calling Colloborative function without baseline approach
collaborative_filtering_func(A.T, B.T, no_of_neighbors, movies_rated_by_user, to_be_predicted, temp, k, top_k_movies_for_B, False)
print("Time taken by Collaborative filtering without baseline approach: " + str(time.time() - start_time))
print("")
start_time = time.time()

# Calling Colloborative function with baseline approach
collaborative_filtering_func(A.T, B.T, no_of_neighbors, movies_rated_by_user, to_be_predicted, temp, k, top_k_movies_for_B, True)
print("Time taken by Collaborative filtering with baseline approach: " + str(time.time() - start_time))
print("")
user_offset = np.zeros(max_user_no + 1)

# Normalizing A matrix
for i in range(max_user_no + 1):
    num = 0.0
    no_of_movies_rated_by_current_user = 0
    for j in range(max_movie_no + 1):
        if (A[i][j] != 0):
            num += A[i][j]
            no_of_movies_rated_by_current_user += 1
    if(no_of_movies_rated_by_current_user > 0):
        user_offset[i] = float(num / float(no_of_movies_rated_by_current_user))
    for j in range(max_movie_no + 1):
        if(A[i][j] != 0):
            A[i][j] = A[i][j] - user_offset[i]

temp = temper.copy()