'''
t = 0,1,2,3...
S = [['g',r','d','d']...]
A = [['g','b','d']...]
R = [0,1,0,0,1...]
'''
import Features

class Environment:
	def __init__(self, actions, states=[], rewards=[]):
		self.states = states
		self.actions = actions
		self.actionHistory = []
		self.time = 0
		# # make sure reward for each state
		# assert(len(rewards) == len(states))
		# # make sure reward for each action in each state
		# for r in rewards:
		# 	assert(len(r) == len(actions))
		self.rewards = rewards
		self.features = Features.all_features()
		# self.features.append(interval_feature)

	def get_states(self):
		return self.states
	def get_actions(self):
		return list(range(len(self.actions)))
	def get_current_state(self):
		return self.states[self.time]

	def take_action(self, action):
		# make sure you are taking a valid action
		assert(action < len(self.actions))
		self.actionHistory.append(action)

		# get reward
		reward = self.rewards[self.time][action]

		# move to the next state
		self.time = (self.time + 1) # % len(self.states)

		return reward, self.time

	def set_rewards(self, r):
		self.rewards = r.copy()

	def terminal_state(self):
		if self.time == len(self.states) - 1:
			return True
		else:
			return False

	def is_terminal_state(self, time):
		if time == len(self.states) - 1:
			return True
		else:
			return False

	def calc_features(self, time, action):
		return Features.calc_features(self, time, action)


def unit_test():

	test_states = [['b', 'a', 'g', 'a'], ['b','b', 'b', 'r'], ['a', 'a', 'a', 'r'], ['b', 'd', 'd', 'r']]
	test_actions = [['g', 'b', 'd'], ['d', 'f#', 'a'], ['c','e','g']]
	test_rewards = [[1, 0, 0],[1, 0, 0],[0, 1, 0],[1, 0, 0]]


	baba_blacksheep = environment(test_states, test_actions, test_rewards)

	print(baba_blacksheep.get_states())
	print(baba_blacksheep.get_actions())

	print(baba_blacksheep.get_current_state())

	reward = baba_blacksheep.take_action(0)

	print(reward)
	print(baba_blacksheep.get_current_state())

	reward = baba_blacksheep.take_action(0)

	print(reward)
	print(baba_blacksheep.get_current_state())
	reward = baba_blacksheep.take_action(0)

	print(reward)
	print(baba_blacksheep.get_current_state())
	reward = baba_blacksheep.take_action(0)

	print(reward)
	print(baba_blacksheep.get_current_state())
	reward = baba_blacksheep.take_action(0)

	print(reward)
	print(baba_blacksheep.get_current_state())

if __name__ == "__main__":
	unit_test()