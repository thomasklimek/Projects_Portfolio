
import os
from collections import defaultdict
import numpy as np
from numpy import genfromtxt
import pandas as pd

from surprise import SVD
from surprise import Dataset
from surprise import Reader
from surprise import accuracy
from surprise.model_selection import PredefinedKFold
from surprise.model_selection import cross_validate

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV


def vector_analysis_classifier(train, test):
    reader = Reader(line_format='user item rating', sep=',', skip_lines=3, rating_scale=(1, 5))
    train_set = pd.read_csv(train)
    test_set = pd.read_csv(test)
    merged_set = pd.concat([train_set, test_set])
    data = Dataset.load_from_df(merged_set, reader=reader).build_full_trainset()
    classifier = SVD().fit(data)
    return classifier

def predict_gender(svd, tune_params):
    trainData = svd.pu
    testData = genfromtxt('project3-data/gender.csv', delimiter='\n')
    testData = np.delete(testData, (0), axis=0)
    if tune_params:
        param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}
        grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
        grid_search.fit(trainData, testData)
        grid_search.best_params_
        print(grid_search.best_params_)

    classifier = SVC(kernel='linear', C=1)
    scores = cross_val_score(classifier, trainData, testData, cv=5)
    print("SVM Classifier Lowest CV Error: ", scores.min())

def naive_method():
    testData =  genfromtxt('project3-data/release-year.csv', delimiter='\n')
    testData = np.delete(testData, (0), axis=0)
    avg_year = np.mean(testData)
    trainData = [avg_year] * len(testData)
    trainData = np.array(trainData)
    trainData = trainData.reshape(-1, 1)
    print("naive method MSE: ", mean_squared_error(testData, trainData))


def predict_release_years(svd):
    trainData = svd.qi
    testData = genfromtxt('project3-data/release-year.csv', delimiter='\n')
    testData = np.delete(testData, (0), axis=0)
    classifier = LinearRegression()
    print("linear regression MSE: ", cross_val_score(classifier, trainData, testData, scoring='neg_mean_squared_error', cv=5).max())

def main():
    svd = vector_analysis_classifier(train = "project3-data/trainset.csv", test = "project3-data/testset.csv")

    # Part 1
    predict_gender(svd, True)

    # Part 2
    naive_method()
    predict_release_years(svd)

main()
