import os
from collections import defaultdict

import pandas as pd
import numpy as np

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import GridSearchCV


def evaluate_predictions(tune_params):
   
    reader = Reader(line_format='user item rating', sep=',', skip_lines=3, rating_scale=(1, 5))

    # if tune parameters argument is true do the grid search
    if tune_params:
        trainingData = Dataset.load_from_file("project3-data/trainset.csv", reader=reader)
        # parameters to search over
        param_grid = {'reg_all': [0.2, 0.4, 0.6], 'n_factors': [1, 2, 5, 10]}
        gs = GridSearchCV(SVD, param_grid, measures=['mae'], cv=5)
        gs.fit(trainingData)

        # best RMSE score
        print(gs.best_score['mae'])

        # combination of parameters that gave the best RMSE score
        print(gs.best_params['mae'])

    data = Dataset.load_from_folds([("project3-data/trainset.csv", "project3-data/testset.csv")], reader=reader)
    pkf = PredefinedKFold()
    classifier = SVD()

    for trainset, testset in pkf.split(data):


        # train and test algorithm.
        classifier.fit(trainset)
        predictions = classifier.test(testset)

        # Compute and print Root Mean Squared Error
        accuracy.mae(predictions)
        accuracy.rmse(predictions, verbose=True)
        
        return classifier, trainset


def recommend_items(classifier, trainset):
    testset = trainset.build_anti_testset()
    predictions = classifier.test(testset)

    top_k = get_predictions(predictions, 5)

    pairs = get_user_move_pairs(top_k)
    return pairs

def get_predictions(pred, n=10):
    # map predictions to users
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in pred:
        top_n[uid].append((iid, est))

    # sort predictions
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def get_user_move_pairs(data):
    user_move_pairs = list()
    for uid, movie_n_rating in data.items():
        user = uid
        for i in range(len(movie_n_rating)):
            movie = movie_n_rating[i][0]
            user_move_pairs.append([user, movie])
    return(user_move_pairs)

def get_avg_rating(user_movie_pairs, df): 

    users = [user[0] for user in user_movie_pairs]
    movies = [movie[1] for movie in user_movie_pairs]

    df['rating'] = np.where((df['user'].isin(users)) & (df['item'].isin(movies)), df['rating'], 2)
    average_rating = df["rating"].mean()
    print(average_rating)


def main():   
    classifier, trainset = evaluate_predictions(True)
    top_k = recommend_items(classifier, trainset)
    
    df = pd.read_csv("project3-data/testset.csv")
    avg_rating = get_avg_rating(top_k, df)

main()

    