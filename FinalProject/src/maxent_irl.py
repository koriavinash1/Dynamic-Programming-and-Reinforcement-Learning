'''
Implementation of maximum entropy inverse reinforcement learning in
	Ziebart et al. 2008 paper: Maximum Entropy Inverse Reinforcement Learning
	https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf

Acknowledgement:
	This implementation is largely influenced by Matthew Alger's maxent implementation here:
	https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/maxent.py
'''

#from __future__ import print_function
import numpy as np
#import mdp.gridworld as gridworld
import ValueIteration as value_iteration
import dill
import pickle
import gym
import matplotlib.pyplot as plt

import math
from collections import namedtuple


Step = namedtuple('Step','cur_state action next_state reward done')


def show_img(img):
	print img.shape, img.dtype
	plt.imshow(img[:,:,0])
	plt.ion()
	plt.show()
	raw_input()


def heatmap2d(hm_mat, title='', block=True, fig_num=1, text=True):
	"""
	Display heatmap
	input:
		hm_mat:   mxn 2d np array
	"""
	print 'map shape: {}, data type: {}'.format(hm_mat.shape, hm_mat.dtype)

	if block:
		plt.figure(fig_num)
		plt.clf()
	
	# plt.imshow(hm_mat, cmap='hot', interpolation='nearest')
	plt.imshow(hm_mat, interpolation='nearest')
	plt.title(title)
	plt.colorbar()
	
	if text:
		for y in range(hm_mat.shape[0]):
			for x in range(hm_mat.shape[1]):
				plt.text(x, y, '%.1f' % hm_mat[y, x],
								 horizontalalignment='center',
								 verticalalignment='center',
								 )

	if block:
		plt.ion()
		print 'press enter to continue'
		plt.show()
		raw_input()


def heatmap3d(hm_mat, title=''):
	from mpl_toolkits.mplot3d import Axes3D
	import matplotlib.pyplot as plt
	import numpy as np
	#
	# Assuming you have "2D" dataset like the following that you need
	# to plot.
	#
	data_2d = hm_mat
	#
	# Convert it into an numpy array.
	#
	data_array = np.array(data_2d)
	#
	# Create a figure for plotting the data as a 3D histogram.
	#
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	plt.title(title)
	# _, ax = fig.add_subplot(111)
	#
	# Create an X-Y mesh of the same dimension as the 2D data. You can
	# think of this as the floor of the plot.
	#
	x_data, y_data = np.meshgrid( np.arange(data_array.shape[1]),
																np.arange(data_array.shape[0]) )
	#
	# Flatten out the arrays so that they may be passed to "ax.bar3d".
	# Basically, ax.bar3d expects three one-dimensional arrays:
	# x_data, y_data, z_data. The following call boils down to picking
	# one entry from each array and plotting a bar to from
	# (x_data[i], y_data[i], 0) to (x_data[i], y_data[i], z_data[i]).
	#
	x_data = x_data.flatten()
	y_data = y_data.flatten()
	z_data = data_array.flatten()
	ax.bar3d( x_data,
						y_data,
						np.zeros(len(z_data)),
						1, 1, z_data )
	#
	# Finally, display the plot.
	#
	plt.show()
	raw_input()


def normalize(vals):
	"""
	normalize to (0, max_val)
	input:
		vals: 1d array
	"""
	min_val = np.min(vals)
	max_val = np.max(vals)
	return (vals - min_val) / (max_val - min_val)


def sigmoid(xs):
	"""
	sigmoid function
	inputs:
		xs      1d array
	"""
	return [1 / (1 + math.exp(-x)) for x in xs]

def compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=True):
	"""compute the expected states visition frequency p(s| theta, T) 
	using dynamic programming

	inputs:
		P_a     NxNxN_ACTIONS matrix - transition dynamics
		gamma   float - discount factor
		trajs   list of list of Steps - collected from expert
		policy  Nx1 vector (or NxN_ACTIONS if deterministic=False) - policy

	
	returns:
		p       Nx1 vector - state visitation frequencies
	"""
	N_STATES, _, N_ACTIONS = np.shape(P_a)

	T = len(trajs[0])
	# mu[s, t] is the prob of visiting state s at time t
	mu = np.zeros([N_STATES, T]) 

	for traj in trajs:    
		# print("Loop Entered .....")     
		mu[int(traj[0].cur_state), 0] += 1
	mu[:,0] = mu[:,0]/len(trajs)

	for s in range(N_STATES):
		# print("N_states .....",s)
		for t in range(T-1):
			if deterministic:
				mu[s, t+1] = sum([mu[pre_s, t]*P_a[pre_s, s, int(policy[pre_s])] for pre_s in range(N_STATES)])
			else:
				mu[s, t+1] = sum([sum([mu[pre_s, t]*P_a[pre_s, s, a1]*policy[pre_s, a1] for a1 in range(N_ACTIONS)]) for pre_s in range(N_STATES)])
	p = np.sum(mu, 1)
	return p



def maxent_irl(feat_map, P_a, gamma, trajs, lr=0.1, n_iters=50):
	"""
	Maximum Entropy Inverse Reinforcement Learning (Maxent IRL)

	inputs:
		feat_map    NxD matrix - the features for each state
		P_a         NxNxN_ACTIONS matrix - P_a[s0, s1, a] is the transition prob of 
																			 landing at state s1 when taking action 
																			 a at state s0
		gamma       float - RL discount factor
		trajs       a list of demonstrations
		lr          float - learning rate
		n_iters     int - number of optimization steps

	returns
		rewards     Nx1 vector - recoverred state rewards
	"""
	N_STATES, _, N_ACTIONS = np.shape(P_a)

	# init parameters
	theta = np.random.uniform(size=(feat_map.shape[1],))

	# calc feature expectations
	feat_exp = np.zeros([feat_map.shape[1]])
	for episode in trajs:
		for step in episode:
			feat_exp += feat_map[int(step.cur_state),:]
	feat_exp = feat_exp/len(trajs)

	# training
	for iteration in range(n_iters):
	
		if iteration % (n_iters/20) == 0:
			print 'iteration: {}/{}'.format(iteration, n_iters)
		
		# compute reward function
		rewards = np.dot(feat_map, theta)

		# compute policy
		_, policy = value_iteration.value_iteration(P_a, rewards, gamma, error=0.01, deterministic=False)
		
		# compute state visition frequences
		svf = compute_state_visition_freq(P_a, gamma, trajs, policy, deterministic=False)
		
		# compute gradients
		grad = feat_exp - feat_map.T.dot(svf)

		# update params
		theta += lr * grad

	rewards = np.dot(feat_map, theta)
	# return sigmoid(normalize(rewards))
	return normalize(rewards)

def conv_trajs_to_reqd_format(trajs):
	#temp_episodes = []
	temp_trajs = []
	Step = namedtuple('Step','cur_state action next_state reward done')
	for episode in trajs:
		temp_episodes = []
		for each_tuple in episode:
			temp_episodes.append(Step(cur_state=each_tuple[0], action=each_tuple[1], next_state=each_tuple[2], reward=each_tuple[3], done=each_tuple[4]))
		temp_trajs.append(temp_episodes)
	return(temp_trajs)




if __name__ == "__main__":
	seed = 1
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default='CartPole-v0')
	parser.add_argument('--exp_name', type=str, default='IRL')
	parser.add_argument('--policy_dir', type=str)
	parser.add_argument('--verbose', type=int, default=1)
	args = parser.parse_args()
	
	path = args.policy_dir
	P_s=np.load(os.path.join(path, 'Trans_prob.npy'))
	trajs = np.load(os.path.join(path,'Trajs.npy'))
	rew = np.load(os.path.join(path,'Rewards_Gt.npy'))
	trajs = conv_trajs_to_reqd_format(trajs)
	shape_Ps = np.shape(P_s)
	feat_map = np.eye(shape_Ps[0])
	try:
		gamma = args[2];lr = args[3]; n_iters = args[4]
		norm_rewards = maxent_irl(feat_map, P_s, gamma, trajs, lr, n_iters)
	except Exception as e:
		norm_rewards = maxent_irl(feat_map, P_s,0.9,trajs)

	plt.plot(rew)
	plt.plot(norm_rewards)
	plt.title('Comparision MaxEntropy and GT rewards, total correlation: {}'.format(np.round(np.mean((rew - norm_rewards)**2), 3)))
	plt.legend(['GT_rewards', 'MaxEntropy_reward'])
	plt.ylabel('Reward')
	plt.xlabel('States')
	plt.savefig(os.path.join(args.policy_dir, 'maxentropy_irl_results.png'))
	np.save(os.path.join(args.policy_dir, 'IRL_rewards'),rew)