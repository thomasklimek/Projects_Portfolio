import os
import re
import string

import pandas as pd 
import numpy as np 
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn import preprocessing

#--------------------------------------Prepare Training and Testing Data---------------------------------------#

#read the data into a pandas dataframe
def read_data(filename):
	data = pd.read_csv(filename, sep='\t', header=None, names = ['text', 'rating'])
	return data

#removes punctionation, numerics, and converts to lower case
def clean_data(data):
	# lowercase
	data.text = data.text.str.lower()
	# punctuation
	data.text = data.text.str.replace('[^\w\s]','')
	# numerics
	data.text = data.text.str.replace('\d+', '')
	#data.text = data.text.apply(lambda x: x.split())
	return data

#splits data to train and test
def split_data(data, ratio):
	x_train, x_test, y_train, y_test = train_test_split(data["text"], data["rating"], test_size=ratio, random_state=0)
	return (x_train, x_test, y_train, y_test)

#Bag of words vectorization function
def bag_of_words(train, test):
	vectorizer = CountVectorizer()
	
	train_counts = vectorizer.fit_transform(train)
	test_counts = vectorizer.transform(test)

	tfidf_transform = TfidfTransformer()
	train_tfidf = tfidf_transform.fit_transform(train_counts)
	test_tfidf = tfidf_transform.transform(test_counts)
	return train_tfidf, test_tfidf

#glove embedding vectorization
def embed(train, test):
	glove_file = datapath(os.path.abspath("glove.6B.50d.txt"))
	tmp_file = get_tmpfile("test_word2vec.txt")
	glove2word2vec(glove_file, tmp_file)

	model = KeyedVectors.load_word2vec_format(tmp_file)

	# embed
	train_data = []
	for t in train:
		t = t.split()
		vecs = np.zeros(50)
		for word in t:
			try:
				vec = model.word_vec(word)
				vecs += vec
			except KeyError:
				#vecs.append(np.zeros(50))
				pass
		#weight = sum(vecs)
		train_data.append(vecs)
	
	test_data = []
	for t in test:
		t = t.split()
		vecs = np.zeros(50)
		for word in t:
			try:
				vec = model.word_vec(word)
				vecs += vec
			except KeyError:
				pass
				#vecs.append(np.zeros(50))
		#weight = sum(vecs)
		test_data.append(vecs)
	scaler = preprocessing.StandardScaler().fit(train_data)
	scaler.transform(train_data)
	scaler.transform(test_data)
	return train_data, test_data

#--------------------------------------3 Learning Algorithms  ---------------------------------------#

# 1 svm
def svm(x_train, x_test, y_train, y_test):
	# hyper parameter tuning
	param_grid = [
                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
                   ]
	grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=0)
	grid.fit(x_train,y_train)
	
	predicted = grid.predict(x_test)
	
	print(np.mean(predicted == y_test))
	print(grid.best_params_)

#2 naive bayes
def naive_bayes(x_train, x_test, y_train, y_test):

	param_grid = {'alpha': (0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1)}

	grid = GridSearchCV(MultinomialNB(),param_grid, refit = True, verbose=1)

	#clf = MultinomialNB().fit(x_train, y_train)

	# Determine the predictions  
	grid.fit(x_train,y_train)
	print(grid.best_params_)
	predicted = grid.predict(x_test)

	print(grid.cv_results_)
	print(np.mean(predicted == y_test))
	print(grid.cv_results_)


#3 logistic regression
def logreg(x_train, x_test, y_train, y_test):

	# Create logistic regression
	logistic = LogisticRegression(random_state=0, solver='lbfgs',
	multi_class='multinomial')
	# Create regularization penalty space
	penalty = ['l2']

	# Create regularization hyperparameter space
	C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

	# Create hyperparameter options
	hyperparameters = dict(C=C, penalty=penalty)

	# Create grid search using 10-fold cross validation
	grid = GridSearchCV(logistic, hyperparameters, cv=10, verbose=0)
	grid.fit(x_train,y_train)
	print(grid.best_params_)
	predicted = grid.predict(x_test)

	print(grid.cv_results_)
	print(np.mean(predicted == y_test))


def logreg_print(x_train, x_test, y_train, y_test):

	# Create logistic regression
	logistic = LogisticRegression(random_state=0, solver='lbfgs',
	multi_class='multinomial')
	# Create regularization penalty space
	penalty = ['l2']

	# Create regularization hyperparameter space
	C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]

	# Create hyperparameter options
	hyperparameters = dict(C=C, penalty=penalty)

	# Create grid search using 10-fold cross validation
	grid = GridSearchCV(logistic, hyperparameters, cv=10, verbose=0)
	grid.fit(x_train,y_train)
	print(grid.best_params_)
	predicted = grid.predict(x_test)
	
	# ---- for making prediction on new data
	f = open( 'predicted-labels.txt', 'w' )
	for p in predicted:
		f.write(str(p))
		f.write("\n")
	f.close()
	# -----------------------------------------

def bayes_print(x_train, x_test, y_train, y_test):
	param_grid = {'alpha': (0, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1)}

	grid = GridSearchCV(MultinomialNB(),param_grid, refit = True, verbose=1)
	grid.fit(x_train,y_train)
	print(grid.best_params_)
	predicted = grid.predict(x_test)
	
	# ---- for making prediction on new data
	f = open( 'predicted-labels.txt', 'w' )
	for p in predicted:
		f.write(str(p))
		f.write("\n")
	f.close()
	# -----------------------------------------

# -------------------------------------Generate prediction for unseen data ------------------#

def make_prediction():
	x_train = read_data('trainreviews.txt')
	x_train = clean_data(x_train)

	y_train = x_train.copy()
	y_train.drop(columns = ["text"], inplace=True)
	x_train.drop(columns = ["rating"], inplace=True)

	x_test = read_data('trainreviewsunlabeled.txt')
	x_test = clean_data(x_test)
	x_test.drop(["rating"], axis = 1, inplace=True)

	#print(x_train.T.squeeze())
	#print(y_train)
	#print(x_test.T.squeeze())
	x_train, x_test = bag_of_words(x_train.T.squeeze(), x_test.T.squeeze())
	
	bayes_print(x_train, x_test, y_train, [])
	#logreg_print(x_train, x_test, y_train, [])
	



#--------------------------------------Main Function  ---------------------------------------#

def main():
	data = read_data('trainreviews.txt')
	data = clean_data(data)
	x_train, x_test, y_train, y_test = split_data(data, 0.3)
	print(x_train, x_test)
	x_train, x_test = bag_of_words(x_train, x_test)
	#x_train, x_test = embed(x_train, x_test)
	logreg(x_train, x_test, y_train, y_test)
	#naive_bayes(x_train, x_test, y_train, y_test)
	#svm(x_train, x_test, y_train, y_test)
main()
