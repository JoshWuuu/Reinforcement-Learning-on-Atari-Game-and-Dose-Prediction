from abc import ABC, abstractmethod

import numpy as np
import csv
import os

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from data import load_data, LABEL_KEY

import pdb

def dose_class(weekly_dose):
	if weekly_dose < 21:
		return 'low'
	elif 21 <= weekly_dose and weekly_dose <= 49:
		return 'medium'
	else:
		return 'high'

def dose_class_action(action):
	if action == 0:
		return 'low'
	elif action == 1:
		return 'medium'
	else:
		return 'high'


# Base classes
class BanditPolicy(ABC):
	@abstractmethod
	def choose(self, x): pass

	@abstractmethod
	def update(self, x, a, r): pass

class StaticPolicy(BanditPolicy):
	def update(self, x, a, r): pass

class RandomPolicy(StaticPolicy):
	def __init__(self, probs=None):
		self.probs = probs if probs is not None else [1./3., 1./3., 1./3.]

	def choose(self, x):
		return np.random.choice(('low', 'medium', 'high'), p=self.probs)

# Baselines
class FixedDosePolicy(StaticPolicy):
	def choose(self, x):
		"""
		Args:
			x: Dictionary containing the possible patient features.
		Returns:
			output: string containing one of ('low', 'medium', 'high')

		TODO:
		Please implement the fixed dose algorithm.
		"""
		#######################################################
		#########   YOUR CODE HERE - ~1 lines.   #############
		return 'medium'
		#######################################################
		#########

class ClinicalDosingPolicy(StaticPolicy):
	def choose(self, x):
		"""
		Args:
			x: Dictionary containing the possible patient features.
		Returns:
			output: string containing one of ('low', 'medium', 'high')

		TODO:
		Please implement the Clinical Dosing algorithm.

		Hint:
			- You may need to do a little data processing here.
			- Look at the "main" function to see the key values of the features you can use. The
				age in decades is implemented for you as an example.
			- You can treat Unknown race as missing or mixed race.
			- Use dose_class() implemented for you.
		"""
		age_in_decades = x['Age in decades']
		height = x['Height (cm)']
		weight = x['Weight (kg)']
		asian = x['Asian']
		black = x['Black']
		mixed = 1 if any([x['White'], x['Unknown race']]) else 0
		enzyme = 1 if any([x['Carbamazepine (Tegretol)'], x['Phenytoin (Dilantin)'], x['Rifampin or Rifampicin']]) else 0
		ami = x['Amiodarone (Cordarone)']
		#######################################################
		#########   YOUR CODE HERE - ~2-10 lines.   #############
		result = 4.0376 - 0.2546 * age_in_decades + 0.0118 * height + 0.0134 * weight - 0.6752 * asian + 0.406 * black + 0.0443 * mixed + 1.2799 * enzyme - 0.5695 * ami
		return dose_class(result**2)
		#######################################################
		#########

# Upper Confidence Bound Linear Bandit
class LinUCB(BanditPolicy):
	def __init__(self, n_arms, features, alpha=1.):
		"""
		See Algorithm 1 from paper:
			"A Contextual-Bandit Approach to Personalized News Article Recommendation"

		Args:
			n_arms: int, the number of different arms/ actions the algorithm can take
			features: list of strings, contains the patient features to use
			alpha: float, hyperparameter for step size.

		TODO:
		Please initialize the following internal variables for the Disjoint Linear Upper Confidence Bound Bandit algorithm.
		Please refer to the paper to understadard what they are.
		Please feel free to add additional internal variables if you need them, but they are not necessary.

		Hints:
		Keep track of a seperate A, b for each action (this is what the Disjoint in the algorithm name means)
		"""
		#######################################################
		#########   YOUR CODE HERE - ~5 lines.   #############
		self.n_arms = n_arms
		self.features = features
		self.alpha = alpha
		self.A = [np.identity(len(self.features)) for i in range(self.n_arms)]
		self.b = [np.zeros((len(self.features), 1)) for i in range(self.n_arms)]
		#######################################################
		#########          END YOUR CODE.          ############

	def choose(self, x):
		"""
		See Algorithm 1 from paper:
			"A Contextual-Bandit Approach to Personalized News Article Recommendation"

		Args:
			x: Dictionary containing the possible patient features.
		Returns:
			output: string containing one of ('low', 'medium', 'high')

		TODO:
		Please implement the "forward pass" for Disjoint Linear Upper Confidence Bound Bandit algorithm.
		"""
		#######################################################
		#########   YOUR CODE HERE - ~7 lines.   #############
		self.x_array = np.array([x[f] for f in self.features]).reshape(-1, 1)
		self.theta = [np.zeros((len(self.features), 1)) for i in range(self.n_arms)]
		p = np.zeros((self.n_arms))
		for i in range(self.n_arms):
			A_inv = np.linalg.inv(self.A[i])
			self.theta[i] = np.dot(A_inv, self.b[i])
			p[i] = np.dot(self.theta[i].T, self.x_array) + self.alpha * np.sqrt(np.dot(np.dot(self.x_array.T, A_inv), self.x_array))
		# print(dose_class(action=np.argmax(p)))
		return dose_class_action(action=np.argmax(p))
		#######################################################
		#########

	def update(self, x, a, r):
		"""
		See Algorithm 1 from paper:
			"A Contextual-Bandit Approach to Personalized News Article Recommendation"

		Args:
			x: Dictionary containing the possible patient features.
			a: string, indicating the action your algorithem chose ('low', 'medium', 'high')
			r: the reward you recieved for that action
		Returns:
			Nothing

		TODO:
		Please implement the update step for Disjoint Linear Upper Confidence Bound Bandit algorithm.

		Hint: Which parameters should you update?
		"""
		#######################################################
		#########   YOUR CODE HERE - ~4 lines.   #############
		x_array = np.array([x[f] for f in self.features]).reshape(-1, 1)
		if a == 'low':
			selected_action = 0
		elif a == 'medium':
			selected_action = 1
		else:
			selected_action = 2
		self.A[selected_action] += np.dot(x_array, np.transpose(x_array))
		self.b[selected_action] += r * x_array
		#######################################################
		#########          END YOUR CODE.          ############

# eGreedy Linear bandit
class eGreedyLinB(LinUCB):
	def __init__(self, n_arms, features, alpha=1.):
		super(eGreedyLinB, self).__init__(n_arms, features, alpha=1.)
		self.time = 0
	def choose(self, x):
		"""
		Args:
			x: Dictionary containing the possible patient features.
		Returns:
			output: string containing one of ('low', 'medium', 'high')

		TODO:
		Instead of using the Upper Confidence Bound to find which action to take,
		compute the probability of each action using a simple dot product between Theta & the input features.
		Then use an epsilion greedy algorithm to choose the action.
		Use the value of epsilon provided
		"""

		self.time += 1
		epsilon = float(1./self.time)* self.alpha
		#######################################################
		#########   YOUR CODE HERE - ~7 lines.   #############
		p = np.zeros((self.n_arms))
		x_array = np.array([x[f] for f in self.features]).reshape(-1, 1)
		for i in range(self.n_arms):
			p[i] = np.dot(np.dot(np.linalg.inv(self.A[i]), self.b[i]).T, x_array)
		best_action = np.argmax(p)
		if np.random.uniform() < epsilon:
			best_action = np.random.randint(0, self.n_arms-1)
		return dose_class_action(action=best_action)
		#######################################################
		#########


# Thompson Sampling
class ThomSampB(BanditPolicy):
	def __init__(self, n_arms, features, alpha=1.):
		"""
		See Algorithm 1 and section 2.2 from paper:
			"Thompson Sampling for Contextual Bandits with Linear Payoffs"

		Args:
			n_arms: int, the number of different arms/ actions the algorithm can take
			features: list of strings, contains the patient features to use
			alpha: float, hyperparameter for step size.

		TODO:
		Please initialize the following internal variables for the Disjoint Thompson Sampling Bandit algorithm.
		Please refer to the paper to understadard what they are.
		Please feel free to add additional internal variables if you need them, but they are not necessary.

		Hints:
			- Keep track of a seperate B, mu, f for each action (this is what the Disjoint in the algorithm name means)
			- Unlike in section 2.2 in the paper where they sample a single mu_tilde, we'll sample a mu_tilde for each arm
				based on our saved B, f, and mu values for each arm. Also, when we update, we only update the B, f, and mu
				values for the arm that we selected
			- What the paper refers to as b in our case is the medical features vector
			- The paper uses a summation (from time =0, .., t-1) to compute the model paramters at time step (t),
				however if you can't access prior data how might one store the result from the prior time steps.

		"""

		#######################################################
		#########   YOUR CODE HERE - ~6 lines.   #############
		self.n_arms = n_arms
		self.features = features
		#Simply use aplha for the v mentioned in the paper
		self.v2 = alpha
		self.B = [np.identity(len(self.features)) for i in range(self.n_arms)]

		#Variable used to keep track of data needed to compute mu
		self.f = [np.zeros((len(self.features),1)) for i in range(self.n_arms)]

		#You can actually compute mu from B and f at each time step. So you don't have to use this.
		self.mu = [np.zeros((len(self.features),1)) for i in range(self.n_arms)]
		#######################################################
		#########          END YOUR CODE.          ############



	def choose(self, x):
		"""
		See Algorithm 1 and section 2.2 from paper:
			"Thompson Sampling for Contextual Bandits with Linear Payoffs"

		Args:
			x: Dictionary containing the possible patient features.
		Returns:
			output: string containing one of ('low', 'medium', 'high')

		TODO:
		Please implement the "forward pass" for Disjoint Thompson Sampling Bandit algorithm.
		Please use the gaussian distribution like they do in the paper
		"""

		#######################################################
		#########   YOUR CODE HERE - ~8 lines.   #############
		p = np.zeros(self.n_arms)
		x_array = np.array([x[f] for f in self.features]).reshape(-1, 1)
		for i in range(self.n_arms):
			mu = np.random.multivariate_normal(np.squeeze(self.mu[i]), self.v2 * np.linalg.inv(self.B[i]))
			p[i] = np.dot(x_array.T, mu)
		return dose_class_action(action=np.argmax(p))
		#######################################################
		#########          END YOUR CODE.          ############

	def update(self, x, a, r):
		"""
		See Algorithm 1 and section 2.2 from paper:
			"Thompson Sampling for Contextual Bandits with Linear Payoffs"

		Args:
			x: Dictionary containing the possible patient features.
			a: string, indicating the action your algorithem chose ('low', 'medium', 'high')
			r: the reward you recieved for that action
		Returns:
			Nothing

		TODO:
		Please implement the update step for Disjoint Thompson Sampling Bandit algorithm.
		Please use the gaussian distribution like they do in the paper

		Hint: Which parameters should you update?
		"""

		#######################################################
		#########   YOUR CODE HERE - ~6 lines.   #############
		if a == 'low':
			selected_action = 0
		elif a == 'medium':
			selected_action = 1
		else:
			selected_action = 2
		x_array = np.array([x[f] for f in self.features]).reshape(-1, 1)
		self.B[selected_action] += np.outer(x_array, x_array)
		self.f[selected_action] += x_array * r
		self.mu[selected_action] = np.dot(np.linalg.inv(self.B[selected_action]), self.f[selected_action])
		#######################################################
		#########          END YOUR CODE.          ############



def run(data, learner, large_error_penalty=False):
	# Shuffle
	data = data.sample(frac=1)
	T = len(data)
	n_egregious = 0
	correct = np.zeros(T, dtype=bool)
	for t in range(T):
		x = dict(data.iloc[t])
		label = x.pop(LABEL_KEY)
		action = learner.choose(x)
		correct[t] = (action == dose_class(label))
		reward = int(correct[t]) - 1
		if (action == 'low' and dose_class(label) == 'high') or (action == 'high' and dose_class(label) == 'low'):
			n_egregious += 1
			reward = large_error_penalty
		learner.update(x, action, reward)

	return {
		'total_fraction_correct': np.mean(correct),
		'average_fraction_incorrect': np.mean([
			np.mean(~correct[:t]) for t in range(1,T) ]),
		'fraction_incorrect_per_time': [
			np.mean(~correct[:t]) for t in range(1,T)],
		'fraction_egregious': float(n_egregious) / T
	}

def main(args):
	data = load_data()

	frac_incorrect = []
	features = [
			'Age in decades',
			'Height (cm)', 'Weight (kg)',
			'Male', 'Female',
			'Asian', 'Black', 'White', 'Unknown race',
			'Carbamazepine (Tegretol)',
			'Phenytoin (Dilantin)',
			'Rifampin or Rifampicin',
			'Amiodarone (Cordarone)'
		]

	extra_features = [
			'VKORC1AG', 'VKORC1AA', 'VKORC1UN',
			'CYP2C912', 'CYP2C913', 'CYP2C922',
			'CYP2C923', 'CYP2C933', 'CYP2C9UN'
		]

	features = features + extra_features

	if args.run_fixed:
		avg = []
		for i in range(args.runs):
			print('Running fixed')
			results = run(data, FixedDosePolicy())
			avg.append(results["fraction_incorrect_per_time"])
			print([(x,results[x]) for x in results if x != "fraction_incorrect_per_time"])
		frac_incorrect.append(("Fixed", np.mean(np.asarray(avg),0)))

	if args.run_clinical:
		avg = []
		for i in range(args.runs):
			print('Runnining clinical')
			results = run(data, ClinicalDosingPolicy())
			avg.append(results["fraction_incorrect_per_time"])
			print([(x,results[x]) for x in results if x != "fraction_incorrect_per_time"])
		frac_incorrect.append(("Clinical", np.mean(np.asarray(avg),0)))

	if args.run_linucb:
		avg = []
		for i in range(args.runs):
			print('Running LinUCB bandit')
			results = run(data, LinUCB(3, features, alpha=args.alpha), large_error_penalty=args.large_error_penalty)
			avg.append(results["fraction_incorrect_per_time"])
			print([(x,results[x]) for x in results if x != "fraction_incorrect_per_time"])
		frac_incorrect.append(("LinUCB", np.mean(np.asarray(avg),0)))

	if args.run_egreedy:
		avg = []
		for i in range(args.runs):
			print('Running eGreedy bandit')
			results = run(data, eGreedyLinB(3, features, alpha=args.ep), large_error_penalty=args.large_error_penalty)
			avg.append(results["fraction_incorrect_per_time"])
			print([(x,results[x]) for x in results if x != "fraction_incorrect_per_time"])
		frac_incorrect.append(("eGreedy", np.mean(np.asarray(avg),0)))

	if args.run_thompson:
		avg = []
		for i in range(args.runs):
			print('Running Thompson Sampling bandit')
			results = run(data, ThomSampB(3, features, alpha=args.v2), large_error_penalty=args.large_error_penalty)
			avg.append(results["fraction_incorrect_per_time"])
			print([(x,results[x]) for x in results if x != "fraction_incorrect_per_time"])
		frac_incorrect.append(("Thompson", np.mean(np.asarray(avg),0)))

	os.makedirs('results', exist_ok=True)
	if frac_incorrect != []:
		for algorithm, results in frac_incorrect:
			with open(f'results/{algorithm}.csv', 'w') as f:
				csv.writer(f).writerows(results.reshape(-1, 1).tolist())
	frac_incorrect = []
	for filename in os.listdir('results'):
		if filename.endswith('.csv'):
			algorithm = filename.split('.')[0]
			with open(os.path.join('results', filename), 'r') as f:
				frac_incorrect.append((algorithm, np.array(list(csv.reader(f))).astype('float64').squeeze()))
	plt.xlabel("examples seen")
	plt.ylabel("fraction_incorrect")
	legend = []
	for name, values in frac_incorrect:
		legend.append(name)
		plt.plot(values[10:])
	plt.ylim(0.0, 1.0)
	plt.legend(legend)
	plt.savefig(os.path.join('results', 'fraction_incorrect.png'))

if __name__ == '__main__':
	from argparse import ArgumentParser
	parser = ArgumentParser()
	parser.add_argument('--run-fixed', action='store_true')
	parser.add_argument('--run-clinical', action='store_true')
	parser.add_argument('--run-linucb', action='store_true')
	parser.add_argument('--run-egreedy', action='store_true')
	parser.add_argument('--run-thompson', action='store_true')
	parser.add_argument('--alpha', type=float, default=1.)
	parser.add_argument('--ep', type=float, default=1)
	parser.add_argument('--v2', type=float, default=0.001)
	parser.add_argument('--runs', type=int, default=5)
	parser.add_argument('--large-error-penalty', type=float, default=-1)
	args = parser.parse_args()
	main(args)