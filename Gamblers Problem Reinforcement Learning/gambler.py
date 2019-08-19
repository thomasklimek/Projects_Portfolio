import numpy as np
import sys
import matplotlib.pyplot as plt

class gambler():
	def __init__(self, p_h, theta, discount_factor, states):
		self.states = states
		self.p_h = p_h
		self.theta = theta
		self.discount_factor = discount_factor
		self.V = np.zeros(self.states + 1) #add extra state for termination
		self.rewards = np.zeros(self.states + 1)
		self.rewards[self.states] = 1
		self.policy = np.zeros(self.states)

	def value_update(self, s):

		A = np.zeros(self.states + 1)
		stakes = range(1, min(s, self.states-s) +1)
		for a in stakes:
			A[a] = self.p_h * (self.rewards[s + a] + self.V[s + a] * self.discount_factor) 
			+ (1 - self.p_h) * (self.rewards[s - a] + self.V[s - a] * self.discount_factor)

		return A

	def value_iteration(self):
		
		while True:
			delta = 0
			for s in range(1,self.states):
				best_action_value = np.max( self.value_update(s))
				delta = max(delta, np.abs(best_action_value - self.V[s]))
				self.V[s] = best_action_value
			if delta < self.theta:
				break

		for p in range(1, self.states):
			best_action = self.states - np.argmax(np.flip(self.value_update(p)))
			#best_action = np.argmax(self.value_update(p))
			self.policy[p] = best_action

	def plot_policy(self):
		
		x = range(self.states)
		y = self.policy

		plt.bar(x, y, align='center', alpha=0.5)

		plt.xlabel('State (current winnings)')
		plt.ylabel('Final Policy (Stake)')

		plt.title('Final Policy')
		plt.show()

	def plot_value(self):

		x = range(self.states)
		y = self.V[:100]

		plt.plot(x, y)

		plt.xlabel('State (current winnings)')
		plt.ylabel('Value')
		plt.title('Value Function with p_h = 0.55')

		plt.show()

		

g = gambler(0.55, 0.0001, 1.0, 100)
g.value_iteration()
print(g.policy)
print(g.V)
g.plot_value()