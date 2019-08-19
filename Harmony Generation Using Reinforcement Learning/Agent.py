from Environment import Environment
import MusicBank as mb
import numpy as np

class Agent:
	def __init__(self, env):
		self.env = env
		self.alpha = 0.01
		self.epsilon = 0.2
		self.gamma = 0.999
		self.update = 0
		self.lastAction = 0
		self.reward = 0
		self.explore = True

		self.Q = [0] * len(env.get_actions())
		self.w = [0] * len(env.features)

	# run one episode (once through the melody)
	def episode(self, tamer):
		self.env.time = 0
		self.evaluate()
		while not self.env.terminal_state():
			self.process_step(False, tamer)
		self.process_step(True, tamer)
		
	# update weights, take selected action, set new Q values
	def process_step(self, last, tamer):
		# print("Prcoess step:", str(self.env.time) + ", state =", self.env.states[self.env.time])
		# if self.env.time > 0:
			# print('update', self.alpha*(self.update), self.env.actions[self.lastAction], self.env.states[self.env.time-1])
		self.update_weights(self.alpha*(self.reward))
		self.update_Q(self.env.time)
	
		if tamer:
			self.lastAction = self.selectedActions[self.env.time]
		else:
			self.lastAction = self.select_action()

		self.reward, time = self.env.take_action(self.lastAction)

		# print("Took action:", self.lastAction, self.env.actions[self.lastAction])
		# print("Got reward:", self.reward)

		if not last:
			self.update = (self.gamma * self.Q[self.lastAction]) + self.reward - self.Q[self.lastAction]


	# select actions for the entire melody
	# explore boolean to enable epsilon exploration
	# if tie, randomly selects from max Q values
	def evaluate(self):
		states = self.env.states
		actions = self.env.actions
		selectedActions = []

		for i,state in enumerate(states):
			self.update_Q(i)
			selectedActions.append(self.select_action())
		# print(self.w)
		# print(selectedActions)
		self.selectedActions = selectedActions

	
	def update_weights(self, update):
		features = self.env.calc_features(self.env.time-1, self.lastAction)
		for i,w in enumerate(self.w):
			# print('\tweight', i, '+=', update, '*', features[i], '=', update*features[i])
			self.w[i] += update * features[i]

	def update_Q(self, time):
		for a in self.env.get_actions():
			self.Q[a] = self.evaluate_action(time, a)

	# use linear approximation to calculate Q values for a given state+action
	def evaluate_action(self, time, action):
		val = 0
		features = self.env.calc_features(time, action)
		for i,w in enumerate(self.w):
			val += features[i] * self.w[i]
		return val

	def select_action(self):
		if self.explore and np.random.rand() <= self.epsilon: # remove -1 to activate exploration
			return np.random.choice(self.env.get_actions())
		else:
			maxIndeces = [i for i,x in enumerate(self.Q) if x == max(self.Q)]
			return np.random.choice(maxIndeces)



