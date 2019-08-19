"""
This problem is modified from a problem in Stanford CS 231n assignment 1. 
In this problem, we implement the neural network with tensorflow instead of numpy
"""

from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class TwoLayerNet(object):
	"""
	A two-layer fully-connected neural network. The net has an input dimension of
	N, a hidden layer dimension of H, and performs classification over C classes.
	We train the network with a softmax loss function and L2 regularization on the
	weight matrices. The network uses a ReLU nonlinearity after the first fully
	connected layer.

	In other words, the network has the following architecture:

	input - fully connected layer - ReLU - fully connected layer - softmax

	The outputs of the second fully-connected layer are the scores for each class.
	"""

	def __init__(self, input_size, hidden_size, output_size, std=1e-4):
		"""
		Initialize the model. Weights are initialized to small random values and
		biases are initialized to zero. Weights and biases are stored in the
		variable self.params, which is a dictionary with the following keys:

		W1: First layer weights; has shape (D, H)
		b1: First layer biases; has shape (H,)
		W2: Second layer weights; has shape (H, C)
		b2: Second layer biases; has shape (C,)

		Inputs:
		- input_size: The dimension D of the input data.
		- hidden_size: The number of neurons H in the hidden layer.
		- output_size: The number of classes C.
		"""

		# store parameters in numpy arrays
		self.params = {}
		self.params['W1'] = tf.Variable(std * np.random.randn(input_size, hidden_size), dtype=tf.float32)
		self.params['b1'] = tf.Variable(np.zeros(hidden_size), dtype=tf.float32)
		self.params['W2'] = tf.Variable(std * np.random.randn(hidden_size, output_size), dtype=tf.float32)
		self.params['b2'] = tf.Variable(np.zeros(output_size), dtype=tf.float32)

		self.session = None # will get a tf session at training

	def get_learned_parameters(self):
		"""
		Get parameters by running tf variables 
		"""

		learned_params = dict()

		learned_params['W1'] = self.session.run(self.params['W1'])
		learned_params['b1'] = self.session.run(self.params['b1'])
		learned_params['W2'] = self.session.run(self.params['W2'])
		learned_params['b2'] = self.session.run(self.params['b2'])

		return learned_params

	def softmax_loss(self, scores, y):
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		#print("scores shapes: ", scores.shape)
		#print("y shape:", y.shape)
		label = tf.one_hot(indices = y, depth = b2.shape[0])
		#print("formated y shape:", label.shape)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( 
						labels=label, logits=scores)) 
		return loss

	def compute_scores(self, X, C):
		"""
		Compute the loss and gradients for a two layer fully connected neural
		network. Implement this function in tensorflow

		Inputs:
		- X: Input data of shape (N, D), tf tensor. Each X[i] is a training sample.
		- C: integer, the number of classes 

		Returns:
		- scores: a tensor of shape (N, C) where scores[i, c] is the score for 
							class c on input X[i].

		"""
		# Unpack variables from the params dictionary
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		N, D = X.shape

		def makeX():
			with tf.variable_scope("x", reuse=tf.AUTO_REUSE):
				tensorX = tf.get_variable("x", initializer=X)
			return tensorX
		
		x = makeX()		
		#print("litte x shape:", x.shape)

		h1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1)) # calculate first hidden layer activations (4x10)
		h2 = tf.add(tf.matmul(h1, W2), b2) # second layed

		return h2


	def compute_objective(self, X, y, reg):
		"""
		Compute the training objective of the neural network.

		Inputs:
		- X: A numpy array of shape (N, D) giving training data.
		- y: A numpy array f shape (N,) giving training labels; y[i] = c means that
			X[i] has label c, where 0 <= c < C.
		- reg: a np.float32 scalar


		Returns: 
		- objective: a tensorflow scalar. the training objective, which is the sum of 
								 losses and the regularization term
		"""
		#print("X shape:", X.shape)
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		nclasses = b2.shape[0]
		#print("nclasses:", nclasses)
		scores = self.compute_scores(X, nclasses)
		loss = self.softmax_loss(scores, y)
		regularization = 0.5*reg*tf.reduce_sum(W1*W1) + 0.5*reg*tf.reduce_sum(W2*W2)

		return loss + regularization
		#return objective

	def train(self, X, y, X_val, y_val,
						learning_rate=1e-3, learning_rate_decay=0.95,
						reg=np.float32(5e-6), num_iters=100,
						batch_size=200, verbose=False):

		"""
		Train this neural network using stochastic gradient descent.

		Inputs:
		- X: A numpy array of shape (N, D) giving training data.
		- y: A numpy array f shape (N,) giving training labels; y[i] = c means that
			X[i] has label c, where 0 <= c < C.
		- X_val: A numpy array of shape (N_val, D) giving validation data.
		- y_val: A numpy array of shape (N_val,) giving validation labels.
		- learning_rate: Scalar giving learning rate for optimization.
		- learning_rate_decay: Scalar giving factor used to decay the learning rate
			after each epoch.
		- reg: Scalar giving regularization strength.
		- num_iters: Number of steps to take when optimizing.
		- batch_size: Number of training examples to use per step.
		- verbose: boolean; if true print progress during optimization.
		"""
		#print(y.shape)
		num_train = X.shape[0]
		x_dim = X.shape[1]

		#print("num_train", num_train)
		#print("x_dim", x_dim)
		#print("batch_size", batch_size)
		iterations_per_epoch = max(num_train / batch_size, 1)

		tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, x_dim)) 
		tf_train_labels = tf.placeholder(tf.int32, shape=batch_size) 
		#print("train shape", tf_train_labels.shape)
		tf_valid_dataset = tf.constant(X_val) 
		tf_test_dataset = tf.constant(y_val)
	
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		nclasses = b2.shape[0]

		loss = self.compute_objective(tf_train_dataset, tf_train_labels, reg)

		# calculate objective

		# get the gradient and the update operation

		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)




		# by this line, you should have constructed the tensorflow graph  
		# no more graph construction
		############################################################################
		# after this line, you should execute appropriate operations in the graph to train the mode  

		session = tf.Session()
		self.session = session
		session.run(tf.global_variables_initializer())

		# Use SGD to optimize the parameters in self.model
		loss_history = []
		train_acc_history = []
		val_acc_history = []

		for it in range(num_iters):
			# pick a randomized offset 
			#print("offsetval:", num_train - batch_size - 1)
			offset = np.random.randint(0, num_train - batch_size) 
			#print("offset: ", offset)
			# Generate a minibatch. 
			X_batch = X[offset:(offset + batch_size)]
			#print("XBATCH:", X_batch) 

			y_batch = y[offset:(offset + batch_size)] 
			#print("YBATCH:", y_batch) 
			feed_dict = {tf_train_dataset : X_batch, tf_train_labels : y_batch} 

			#########################################################################
			# TODO: Create a random minibatch of training data and labels, storing  #
			# them in X_batch and y_batch respectively.                             #
			# 
			#########################################################################
			#pass


			# Compute loss and gradients using the current minibatch
			loss = self.compute_objective(X_batch, y=y_batch, reg=reg)
			loss_history.append(session.run(loss, feed_dict=feed_dict)) # need to feed in the data batch

			session.run(optimizer, feed_dict=feed_dict)


			#########################################################################
			#                             END OF YOUR CODE                          #
			#########################################################################

			if verbose and it % 100 == 0:
				print('iteration %d / %d: ' % (it, num_iters))

			# Every epoch, check train and val accuracy and decay learning rate.
			if it % iterations_per_epoch == 0:
				# Check accuracy
				#print("predictions:", self.predict(X_batch))
				#print("y_batch:", y_batch)
				train_acc = np.mean((self.predict(X_batch) == y_batch))
				#print("predictions:", self.predict(X_val))
				val_acc = np.mean((self.predict(X_val) == y_val))


				train_acc_history.append(train_acc)
				val_acc_history.append(val_acc)

				# Decay learning rate
				learning_rate *= learning_rate_decay

		return {
			'objective_history': loss_history,
			'train_acc_history': train_acc_history,
			'val_acc_history': val_acc_history,
		}


	def predict(self, X):
		"""
		Use the trained weights of this two-layer network to predict labels for
		data points. For each data point we predict scores for each of the C
		classes, and assign each data point to the class with the highest score.

		Inputs:
		- X: A numpy array of shape (N, D) giving N D-dimensional data points to
			classify.

		Returns:
		- y_pred: A numpy array of shape (N,) giving predicted labels for each of
			the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
			to have class c, where 0 <= c < C.
		"""
		W1, b1 = self.params['W1'], self.params['b1']
		W2, b2 = self.params['W2'], self.params['b2']
		nclasses = b2.shape[0]
		#print(nclasses)
		#print(self.compute_scores(X, nclasses))
		y_pred = tf.argmax(self.compute_scores(X, nclasses), axis=1)
		#print(y_pred)
		#print(self.session.run(y_pred))
		return self.session.run(y_pred)


