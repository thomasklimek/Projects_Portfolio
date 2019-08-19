import numpy as numpy
import random
import matplotlib.pyplot as plt

class Bandit:
	
	def __init__(self, initial_value):
		self.initial_value = initial_value
		self.arms = [self.initial_value for i in range(10)]
	
	def random_walk(self):
		self.arms += numpy.random.normal(loc=0.0, scale=0.01, size=10)
	
	def reward(self, a):
		self.random_walk()
		return self.arms[a]
	
	def get_optimal(self):
		return self.arms.argmax()

	def reset(self):
		self.arms = [self.initial_value for i in range(10)]


class Agent:

	def __init__(self, steps, epsilon, alpha):
		self.steps = steps
		self.epsilon = epsilon
		self.alpha = alpha


class Sample_Average(Agent):

	def solve(self, bandit, stats):
		Q = {i: 0 for i in range(10)} #  Value function for 10 arms, with initial value 0
		N = {i: 0 for i in range(10)} #  Number of actions, for update rule

		for i in range(self.steps): 
			if random.uniform(0, 1) < self.epsilon: # exploration
				action = random.randint(0, 10 - 1) 
			else:
				action = max(Q, key=Q.get) 

			reward = bandit.reward(action)
			# Update Rule 
			N[action] += 1 
			Q[action] += (1 / N[action]) * (reward - Q[action])
			#Average Values
			stats.update(i, reward, 1 if action == bandit.get_optimal() else 0)			


class Weighted_Average(Agent):
	
	def solve(self, bandit, stats):
		Q = {i: 0 for i in range(10)} #  Value function for 10 arms, with initial value 0
		N = {i: 0 for i in range(10)} #  Number of actions, for update rule

		for i in range(self.steps): 
			if random.random() < self.epsilon: # exploration
				action = random.randint(0, 10 - 1) 
			else:
				action = max(Q, key=Q.get) 

			reward = bandit.reward(action)
			# Update Rule 
			N[action] += 1 
			Q[action] += self.alpha * (reward - Q[action])
			#Average Values
			stats.update(i, reward, 1 if action == bandit.get_optimal() else 0)


class Statistics():

	def __init__(self, trials, agent):
		self.agent = agent
		self.steps = agent.steps
		self.trials = trials
		# stores average data for 1 trail
		self.reset_trials()
		# stores average data over all trials
		self.avg_data = {'avg': [0 for i in range(self.steps)], 'opt': [0 for i in range(self.steps)]}

	def update(self, step, reward, optimal):
		self.average_reward += (1 / (step + 1)) * (reward - self.average_reward)
		self.optimal_percentage += (1 / (step + 1)) * (optimal - self.optimal_percentage)
		self.average_rewards = numpy.append(self.average_rewards, self.average_reward)
		self.optimal_precentages = numpy.append(self.optimal_precentages, self.optimal_percentage)
		# update average data over all trials
		self.update_averages(step)

	def update_averages(self, step):
		self.avg_data['avg'][step] += self.average_rewards[step] * (1/self.trials) 
		self.avg_data['opt'][step] += self.optimal_precentages[step] * (1/self.trials)

	def reset_trials(self):
		self.average_reward = 0
		self.average_rewards = numpy.array([])
		self.optimal_percentage = 0
		self.optimal_precentages = numpy.array([])

	def run_trials(self, bandit):
		for i in range(self.trials):
			self.agent.solve(bandit, self)
			self.reset_trials()
			bandit.reset()


def plot2(stat1, stat2):
		fig, ax = plt.subplots(1,2, figsize=(20, 5), dpi=70)

		ax[0].plot(numpy.array(list(range(stat1.steps))), stat1.avg_data['opt'], label='Sample_Average')
		ax[0].plot(numpy.array(list(range(stat2.steps))), stat2.avg_data['opt'], label='Weighted_Average')
		ax[1].plot(numpy.array(list(range(stat1.steps))), stat1.avg_data['avg'], label='Sample_Average')
		ax[1].plot(numpy.array(list(range(stat2.steps))), stat2.avg_data['avg'], label='Weighted_Average')

		ax[0].set_ylabel('Optimal', fontsize=16)
		ax[0].set_xlabel('Step number', fontsize=16)
		ax[1].set_ylabel('Average reward', fontsize=16)
		ax[1].set_xlabel('Step number', fontsize=16)

		ax[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

		plt.suptitle('Increment v.s. Static Step Size', fontsize=32)
		plt.show()

STEPS = 10000
EPSILON = 0.1
ALPHA = 0.1
TRIALS = 100

INITIAL_ARM_VALUE = numpy.random.normal(loc=0.0, scale=.1)

bandit = Bandit(INITIAL_ARM_VALUE)
stats_sample = Statistics(TRIALS, Sample_Average(STEPS, EPSILON, ALPHA))
stats_sample.run_trials(bandit)

bandit = Bandit(INITIAL_ARM_VALUE)
stats_weighted = Statistics(TRIALS, Weighted_Average(STEPS, EPSILON, ALPHA))
stats_weighted.run_trials(bandit)

plot2(stats_sample, stats_weighted)
