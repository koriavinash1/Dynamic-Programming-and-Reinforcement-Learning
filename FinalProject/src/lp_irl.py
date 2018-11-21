'''
Implementation of linear programming inverse reinforcement learning in
	Ng & Russell 2000 paper: Algorithms for Inverse Reinforcement Learning
	http://ai.stanford.edu/~ang/papers/icml00-irl.pdf
'''
from __future__ import print_function
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import pickle
import math
import os
from collections import namedtuple

Step = namedtuple('Step','cur_state action next_state reward done')


def normalize(vals):
	min_val = np.min(vals)
	max_val = np.max(vals)
	return (vals - min_val) / (max_val - min_val)


def sigmoid(xs):
	return [1 / (1 + math.exp(-x)) for x in xs]


def lp_irl(trans_probs, policy, gamma=0.5, l1=10, R_max=10):
	"""
	inputs:
		trans_probs       NxNxN_ACTIONS transition matrix
		policy            policy vector / map
		R_max             maximum possible value of recoverred rewards
		gamma             RL discount factor
		l1                l1 regularization lambda
	returns:
		rewards           Nx1 reward vector
	"""
	l1 = 7
	N_STATES, _, N_ACTIONS = np.shape(trans_probs)
	N_STATES = int(N_STATES)
	N_ACTIONS = int(N_ACTIONS)

	# Formulate a linear IRL problem
	A = np.zeros([2 * N_STATES * (N_ACTIONS + 1), 3 * N_STATES])
	print("Shape of A : ",np.shape(A))
	b = np.zeros([2 * N_STATES * (N_ACTIONS + 1)])
	print("Shape of b : ",np.shape(b))
	c = np.zeros([3 * N_STATES])
	print("Shape of c : ",np.shape(c))

	for i in range(N_STATES):
		a_opt = int(policy[i])
		tmp_inv = np.linalg.inv(np.identity(N_STATES) - gamma * trans_probs[:, :, a_opt])

		cnt = 0
		for a in range(N_ACTIONS):
			if a != a_opt:
				A[i * (N_ACTIONS - 1) + cnt, :N_STATES] = - \
						np.dot(trans_probs[i, :, a_opt] - trans_probs[i, :, a], tmp_inv)
				A[N_STATES * (N_ACTIONS - 1) + i * (N_ACTIONS - 1) + cnt, :N_STATES] = - \
						np.dot(trans_probs[i, :, a_opt] - trans_probs[i, :, a], tmp_inv)
				A[N_STATES * (N_ACTIONS - 1) + i * (N_ACTIONS - 1) + cnt, N_STATES + i] = 1
				cnt += 1

	for i in range(N_STATES):
		A[2 * N_STATES * (N_ACTIONS - 1) + i, i] = 1
		b[2 * N_STATES * (N_ACTIONS - 1) + i] = R_max

	for i in range(N_STATES):
		A[2 * N_STATES * (N_ACTIONS - 1) + N_STATES + i, i] = -1
		b[2 * N_STATES * (N_ACTIONS - 1) + N_STATES + i] = 0

	for i in range(N_STATES):
		A[2 * N_STATES * (N_ACTIONS - 1) + 2 * N_STATES + i, i] = 1
		A[2 * N_STATES * (N_ACTIONS - 1) + 2 * N_STATES + i, 2 * N_STATES + i] = -1

	for i in range(N_STATES):
		A[2 * N_STATES * (N_ACTIONS - 1) + 3 * N_STATES + i, i] = 1
		A[2 * N_STATES * (N_ACTIONS - 1) + 3 * N_STATES + i, 2 * N_STATES + i] = -1

	for i in range(N_STATES):
		c[N_STATES:2 * N_STATES] = -1
		c[2 * N_STATES:] = l1

	sol = solvers.lp(matrix(c), matrix(A), matrix(b))
	rewards = sol['x'][:N_STATES]
	rewards = normalize(rewards) * R_max
	return rewards

if __name__ == "__main__":

	###################################################################################################
	#                                        Args Selection                                           #
	###################################################################################################
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default='CartPole-v0')
	parser.add_argument('--exp_name', type=str, default='IRL')
	parser.add_argument('--policy_dir', type=str)
	parser.add_argument('--verbose', type=int, default=1)
	args = parser.parse_args()

	print ("[INFO]: Policy Directory: {}".format(args.policy_dir))
	file = open(os.path.join(args.policy_dir, 'ARGS1.txt'),"r")
	irl_args = pickle.load(file)
	print("No: of states",np.shape(irl_args[0]))

	P_s = irl_args[0]
	policy = irl_args[1]

	gt_rewards = irl_args[2]
	rewards = lp_irl(P_s,policy)
	
	plt.plot(gt_rewards)
	plt.plot(rewards)
	plt.title('Comparision IRL and GT rewards, total correlation: {}'.format(np.round(np.mean((gt_rewards - rewards)**2), 3)))
	plt.legend(['GT_rewards', 'IRL_reward'])
	plt.ylabel('Reward')
	plt.xlabel('States')
	plt.savefig(os.path.join(args.policy_dir, 'lp_irl_results.png'))
	np.save(os.path.join(args.policy_dir, 'IRL_rewards'),rewards)