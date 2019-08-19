import autograd.numpy as np
from autograd import grad
import random
import time
import matplotlib.pyplot as plt

class MyLogisticReg:
	def __init__(self, options, lam):
		self.w = []
		self.steps = 100
		self.step_size = 0.00001
		self.lam = lam
		self.options = 'sto'
	def normalize_X(self, X):
		#return np.where((self.x_max - self.x_min) != 0, (X - self.x_min) // (self.x_max - self.x_min), X)
		return (X - self.x_min) / (self.x_max - self.x_min)
	def normalize_Y(self, Y):
		#return np.where((self.y_max - self.y_min) != 0, (Y - self.y_min) // (self.y_max - self.y_min), Y)
		return (Y - self.y_min) / (self.y_max - self.y_min)

	def gradient_function(self, w):
		# separate w and w0
		w0 = w[-1]
		w = w[:-1] 

		eta = np.dot(self.x, w.T) + w0 

		grad_w = np.dot(np.where(eta > 30, self.y - 1, self.y - (np.exp(eta) / (1 + np.exp(eta)))), self.x)
		grad_w_0 = np.dot(np.where(eta > 30, self.y - 1, self.y - (np.exp(eta) / (1 + np.exp(eta)))), 1)
		print(grad_w_0)
		return np.append(grad_w, grad_w_0)


	def loss_function(self, w):
		# separate w and w0
		w0 = w[-1]
		w = w[:-1] 

		# Calculate normal
		normal = (self.lam / 2) * np.power(np.linalg.norm(w), 2)

		# Calculate Eta
		eta = np.dot(self.x, w.T) + w0 

		# Calculate a_i
		a_i = np.where(eta > 30, np.multiply(self.y, eta) - eta, np.multiply(self.y, eta) - np.log(1 + np.exp(eta)))

		# Find the loss
		loss = normal - np.sum(a_i)
		return(loss)
	def sto_loss_function(self, w):
		# separate w and w0
		w0 = w[-1]
		w = w[:-1]
		n = self.x.shape[0]

		# Generate random starting index to split for the Batch
		idx = np.random.randint(0, n-11)
		# Generate ending index for batch size of 10
		idx_a = [idx, idx+10]

		#create batch
		batch = self.x[idx_a,:]
		# Calculate normal
		normal = np.dot(w, w.T) * (self.lam/(2 * n))
		# Calculate eta using the batch instead of X
		eta = np.dot(batch, w.T) + w0 
		# Find alpha
		a = np.where(eta > 30, np.multiply(self.y, eta) - eta, np.multiply(self.y, eta) - np.log(1 + np.exp(eta)))
		# Find the loss
		loss = normal - (n/10)*np.sum(a)
		return(loss)

	def fit(self, X, y):

		start_time = time.time()
		times = []
		
		# store the x min and max for normalization
		self.x_min = np.min(X, axis=0, keepdims=True)
		self.x_max = np.max(X, axis=0, keepdims=True)

		#normalize X
		X = self.normalize_X(X)


		# store the x min and max for normalization
		self.y_min = np.min(y, axis=0, keepdims=True)
		self.y_max = np.max(y, axis=0, keepdims=True)

		#normalize X
		y = self.normalize_Y(y)


		#store x and y for use in loss calculation
		self.x = X
		self.y = y

		# initiaize variables

		#Dimension
		d = X.shape[1]
		#w = [random.uniform(0, 1)] * (d) # initialize w vector to have dimension D

		#weights
		w = np.random.uniform(0,1,d)
		w_0 = random.uniform(0, 1)
		w = np.append(w, w_0)

		#diff
		diff = 1

		#gradient
		if self.options is 'sto':
			gradient = grad(self.sto_loss_function)
		else:
			gradient = grad(self.loss_function)
		
		t = 0

		obj_val = []

		
		#Delta check for loss function
		'''
		delta = np.random.uniform(0,1,d+1) * 1e-5
		print("Gradient Check")
		print(self.loss_function(w+delta) - self.loss_function(w))
		print(np.matmul(gradient(w).transpose(),delta))
		'''

		while(diff > 1e-4):
			w_prev = w.copy()
			for i in range(self.steps):	
				t += 1
				# variant step size
				#np.append(obj_val, self.loss_function(w))
				w = w - ((0.1/(1000*(1+t))) * gradient(w))
				#w -=self.step_size * gradient(w)
			times.append(time.time() - start_time)
			obj_val.append(self.loss_function(w))
			diff = (1 / (1+d)) * np.sum(np.abs(w - w_prev))
			print(diff)
		#store weights
		self.w = w
		val = range(len(obj_val))

		# Objective function plotting
		
		plt.xlabel("iterations")
		plt.ylabel("cost")
		plt.title("Objective Curve Stochastic")
		plt.legend()
		plt.plot(val, obj_val)
		plt.autoscale()
		plt.show()
		
	def predict(self, X):
		# normalize X data
		X = self.normalize_X(X)



		# split w argument into w and w_0
		w_0 = self.w[-1]
		w = self.w[:-1]

		#ypred = (np.matmul(X, w.transpose()) + w_0).astype(int)
		ypred = (np.matmul(X, w.transpose()) + w_0) > 0

		print(ypred)
		return ypred
	def evaluate(self, y_test , y_pred):
		y_test = y_test.flatten()
		y_test = self.normalize_Y(y_test)
		np.savetxt("pred", y_pred, newline=" ")
		np.savetxt("test", y_test, newline=" ")
		
		error_rate = (np.sum(y_test == y_pred)/ y_test.size)
		print(error_rate)
		return error_rate
	

def split_data(X, Y, train_test_ratio):
	#calculate split indices
	x_split = int(X.shape[0] * train_test_ratio)
	y_split = int(Y.shape[0] * train_test_ratio)

	# preform split based on indices
	x_train, x_test = X[:x_split, :], X[x_split:, :]
	y_train, y_test = Y[:y_split, :], Y[y_split:, :]

	return (x_train, y_train, x_test, y_test)

	
def read_data(filename):
	data = np.genfromtxt(filename, dtype=float, delimiter=',', skip_header=1)
	np.random.shuffle(data) 
	return np.delete(data, [0], axis = 1), data[:,[0]]


def main():
	# generating lambda graph
	'''
	lam_vals = [0, 0.01, 0.1, 1, 10, 100, 1000]
	lam_vals_str = ['0', '0.01', '0.1', '1', '10', '100', '1000']
	graph_data = []
	for lam in lam_vals:
		X, Y = read_data('titanic_train.csv')
	
		#print(X)
		#print(Y)
		(X_train, Y_train, X_test, Y_test ) = split_data(X, Y, 0.7)
		options = None
		model = MyLogisticReg(options, lam)
		model.fit(X_train, Y_train)
		y_pred = model.predict(X_test)
		error_rate = model.evaluate(Y_test, y_pred)
		graph_data.append(error_rate)
	print(graph_data)
	plt.xlabel("lambda")
	plt.ylabel("accuracy")
	plt.title("Lambda v.s. accuracy")
	plt.xticks(np.arange(7), lam_vals_str)
	plt.scatter(range(7), graph_data)
	#plt.autoscale()
	plt.show()

	'''
	lam = 0.01
	X, Y = read_data('titanic_train.csv')
	(X_train, Y_train, X_test, Y_test ) = split_data(X, Y, 0.7)
	options = None
	model = MyLogisticReg(options, lam)
	model.fit(X_train, Y_train)
	y_pred = model.predict(X_test)
	error_rate = model.evaluate(Y_test, y_pred)

main()

def k_fold(fold):
	X, Y = read_data('titanic_train.csv')
	# get first fold index
	fold_index = len(X) / fold
	#initialize array of folds
	fold_indices = []
	# add all fold indexes to array
	for i in range(1, fold+1):
		fold_indices.append(i*fold)
	
	#initialize array to store all error rates
	error_rates = []
	for f in fold:
		# read the data
		
		# shift it over by the fold index, so partition doesnt include different fold everytime
		np.roll(X, f)
		np.roll(Y, f)
		# train the data with ratio (fold - 1) / fold to split training data into all but 1 fold
		(X_train, Y_train, X_test, Y_test ) = split_data(X, Y, (fold-1)/fold)
		options = None
		model = MyLogisticReg(options, lam)
		model.fit(X_train, Y_train)
		y_pred = model.predict(X_test)
		error_rate = model.evaluate(Y_test, y_pred)
		# append the error rate from the current fold
		error_rates.append(error_rate)
	# get the average error rate by summing and diving for each fold
	return np.sum(error_rates) / folds


'''
	def loss(self, w):
		w_0 = w[-1]
		w_1 = w[:-1]

		norm = (self.lam / 2) * np.power(np.linalg.norm(w), 2)
		eta = np.matmul(self.x, np.transpose(w_1)) + w_0
		#print(eta.shape)
		#print(self.y.shape)
		#alpha = np.sum(np.multiply(self.y, eta) - np.greater(eta, 30) * eta - 
		#np.less_equal(eta, 30) * np.log(1 + np.exp(eta)))
		#alpha = np.sum(np.multiply(self.y, eta) - eta)
		#alpha = np.multiply(self.y, eta) - np.greater(eta, 30) * eta + np.less_equal(eta, 30) *  np.log(1 + np.exp(eta))
		#a = np.sum(alpha)
		#alpha = np.sum(np.multiply(self.y, eta) - np.log(1 + np.exp(eta)))
		
		#for ind, e in enumerate(eta):
			#if e > 30:
			#	pass
			#else:
			#	e = np.log(1 + np.exp(e))
		
		a = np.where(eta > 30, np.multiply(self.y, eta) - eta, np.multiply(self.y, eta) - np.log(1 + np.exp(eta)))
		a_i = np.sum(a)
		loss_val = norm - a_i
		return loss_val
'''