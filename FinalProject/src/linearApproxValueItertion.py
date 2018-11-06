####################################################################
#  Reinforcement learning agent with linear approimate Q-function  #
####################################################################
import gym
import numpy as np
import random
import math
import os
import matplotlib.pyplot as plt 
from gym import wrappers
import tensorflow as tf

NUM_EPISODES = 5000
MAX_T = 10000
GAMMA = 0.99
LEARNING_RATE = 0.001
EXPLORATION_RATE = 1.0
EXPLORATION_RATE_DECAY = 0.010
MIN_EXPLORATION_RATE = 0
LAMBDA = 0.01


OUTPUT_RESULTS_DIR = "../logs"
# ENVIRONMENT = 'InvertedPendulum-v2'
ENVIRONMENT = 'CartPole-v1'

# TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
TIMESTAMP = 'RESULTS'

SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "ApproxValueIteration", ENVIRONMENT, TIMESTAMP)
env = gym.make(ENVIRONMENT)
env._max_episode_steps = 100000
env = wrappers.Monitor(env, os.path.join(SUMMARY_DIR, ENVIRONMENT), force=True, video_callable=None)

NUM_ACTIONS = env.action_space.n #shape[0]
NUM_OBS = env.observation_space.shape[0]

######################################################################
#						Tensorflow Graph Define                      #
######################################################################

session = tf.Session()
state_ = tf.placeholder(tf.float32, [None, NUM_OBS])
targets = tf.placeholder(tf.float32, [None, NUM_ACTIONS])

# with tf.name_scope("fc1"):
# 		fc1 = tf.contrib.layers.fully_connected(inputs = state_,
# 												num_outputs = 400,
# 												activation_fn=tf.nn.relu,
# 												weights_initializer=tf.contrib.layers.xavier_initializer())
# with tf.name_scope("output"):
# 		output = tf.contrib.layers.fully_connected(inputs = fc1,
# 												num_outputs = NUM_ACTIONS,
# 												activation_fn=tf.nn.softmax,
# 												weights_initializer=tf.contrib.layers.xavier_initializer())


hidden_weights = tf.Variable(tf.random_uniform(shape=[NUM_OBS, NUM_ACTIONS], dtype=tf.float32))
output = tf.matmul(state_, hidden_weights)

loss = tf.reduce_mean(tf.square(output - targets)) + LAMBDA*tf.reduce_mean(hidden_weights)
train_operation = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)


saver = tf.train.Saver()
session.run(tf.initialize_all_variables())

######################################################################
#                         Control Parameters                         # 
######################################################################

STOP_TRAIN = False
DEBUG_MODE = False
Train = False
avg = 0

# to store samples for training
X = []
y = []

score_100 = [0 for i in range(100)]
score_ptr = 0
accumulated_samples = 0

######################################################################
#                         Required Functions                         # 
######################################################################
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def get_reward(state, on = True):
	p = np.random.uniform(0,1)
	# epsilon greedy action
	if p < EXPLORATION_RATE and on == True:
		return np.random.uniform(0,1,(1,NUM_ACTIONS))
	
	# action according to policy
	reward = session.run(output, feed_dict={state_: [state] })
	return reward

# prepare for update given s, s' and r
def update(state, state_prime, r):
	reward = get_reward(state, on = False)[0]
	reward_prime = get_reward(state_prime, on = False)[0]

	q_prime = max(reward_prime)
	action_prime = np.argmax(reward_prime)

	retval = []
	for i in range(NUM_ACTIONS):
		if i==action_prime:
			retval.append( r + GAMMA*q_prime )
		else:
			retval.append( reward[i] )

	X.append(state)
	y.append(retval)




######################################################################
#                             Optimization                           # 
######################################################################

episode_rewards = []
mean_rewards = []

# begin RL
if Train:
	##########################################################################
	#                           Model Training                               #
	##########################################################################
	for episode in range(NUM_EPISODES):
		state = env.reset()
		# env.render()

		episode_reward = 0

		for t in range(MAX_T):
			# get reward accoding to policy
			rewards_from_action = get_reward(state)[0]

			# argmax{a} among rewards_from_action
			action = np.argmax(rewards_from_action)

			# take step with action
			state_prime, reward, done, info = env.step(action)
			episode_reward = episode_reward + reward
			
			if not STOP_TRAIN:
				update(state, state_prime, reward)

			accumulated_samples += 1

			if done or t == MAX_T - 1:
				episode_rewards.append(episode_reward)
				mean_rewards.append(np.mean(episode_rewards))
				if DEBUG_MODE:
					print("[INFO Data {}]============================".format(t))
					print("Episode: ", episode)
					print("Reward: ", episode_reward)
					print("Mean Reward: ", np.mean(episode_rewards))
					print("Max reward so far: ", max(episode_rewards))

				avg += t

				score_100[score_ptr] = t
				score_ptr = (score_ptr+1) % 100
				env.close()
				break

			state = state_prime


		if not STOP_TRAIN:
			session.run(train_operation, feed_dict={ state_: X, targets: y } )
			X, y = [], []
		
		if episode%50 == 0:
			EXPLORATION_RATE = max(MIN_EXPLORATION_RATE, EXPLORATION_RATE*EXPLORATION_RATE_DECAY)
			
			saver.save(session, os.path.join(SUMMARY_DIR, "model.ckpt"), global_step=None)
			print("At %d episodes" % (episode))
			print("EXPLORATION_RATE:", EXPLORATION_RATE)
			print("Average of 100:", episode_reward)
			print("Max reward so far: ", max(episode_rewards))
			print ("\n")
			avg = 0



	episode_rewards = moving_average(episode_rewards, n = 50)
	episode_rewards[0] = mean_rewards[30]
	plt.plot(episode_rewards)
	plt.plot(mean_rewards[30:])
	plt.title('Linear Approx Value Iteration Reward Convergence')
	plt.legend(['Episode reward with smoothening widow of n = 25', 'Mean episode reward'])
	plt.ylabel('Reward')
	plt.xlabel('Episodes')
	plt.show()



else:

	##########################################################################
	#                         Model Evaluation                               #
	##########################################################################

	saver.restore(session, os.path.join(SUMMARY_DIR, "model.ckpt"))
	state = env.reset()

	env.render()

	episode_reward = 0
	
	while True:
		rewards_from_action = get_reward(state, on=False)[0]
		action = np.argmax(rewards_from_action)
		state, reward, done, info = env.step(action)
		episode_reward = episode_reward + reward
		if done: break

	print ("[INFO] Final evaluation reward: {}".format(episode_reward))
	env.close()