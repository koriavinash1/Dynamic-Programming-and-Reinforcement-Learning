from __future__ import print_function
import numpy as np
import os
import gym
from gym import wrappers

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='CartPole-v0')
parser.add_argument('--max_episode_len', type=int, default=10000)
parser.add_argument('--max_episode_steps', type=int, default=10000)
parser.add_argument('--policy_dir', type=str)

args = parser.parse_args()
ENVIRONMENT            = args.env_name
MAX_T                  = args.max_episode_len
MAXENVSTEPS            = args.max_episode_steps

env = gym.make(ENVIRONMENT)
env._max_episode_steps = MAXENVSTEPS

env = wrappers.Monitor(env, args.policy_dir, force=True, video_callable=None)
env.seed(1)

gt_reward = np.load(os.path.join(args.policy_dir, 'gt_state_rewards.npy'))
irl_reward = np.load(os.path.join(args.policy_dir, 'irl_state_rewards.npy'))

gt_state_values = np.load(os.path.join(args.policy_dir, 'gt_state_values.npy'))
irl_state_values = np.load(os.path.join(args.policy_dir, 'irl_state_values.npy'))

gt_state_transition_probabilities = np.load(os.path.join(args.policy_dir, 'gt_state_transition_probabilities.npy'))
irl_state_transition_probabilities = np.load(os.path.join(args.policy_dir, 'irl_state_transition_probabilities.npy'))

select_observations = lambda O: np.array([O[1],O[2],O[3]])

##########################################################################
#                         Model Evaluation                               #
##########################################################################

def observation_to_state(observation):
	state = 0
	for observation_dimension in range(observation_dimensions):
		state = state + np.digitize(observation[observation_dimension],\
					observation_dimension_bins[observation_dimension])*\
					NUMBER_OF_BINS**observation_dimension 
	return state


def pick_best_action(current_state, state_values, state_transition_probabilities, eval_ = False):
	best_action = -1
	best_action_value = -np.Inf
	for a_i in range(num_actions):
		action_value = state_transition_probabilities[current_state,:,a_i].dot(state_values)
		if eval_ or (action_value > best_action_value):
			best_action_value = action_value
			best_action = a_i
		elif (action_value == best_action_value):
			if np.random.randint(0,2) == 0:
					best_action = a_i

	return best_action
	
env.render()
current_observation = env.reset()
current_observation = select_observations(current_observation)
current_state = observation_to_state(current_observation)
gt_episode_reward = 0


while True:
	action = pick_best_action(current_state, gt_state_values, gt_state_transition_probabilities)
	old_state = current_state
	observation, reward, done, info = env.step(action)
	gt_episode_reward = gt_episode_reward + reward
	current_state = observation_to_state(select_observations(observation))
	if done: break


current_observation = env.reset()
current_observation = select_observations(current_observation)
current_state = observation_to_state(current_observation)
irl_episode_reward = 0

while True:
	action = pick_best_action(current_state, irl_state_values, irl_state_transition_probabilities)
	old_state = current_state
	observation, reward, done, info = env.step(action)
	irl_episode_reward = irl_episode_reward + reward
	current_state = observation_to_state(select_observations(observation))
	if done: break

env.close()

print ("[INFO] Final evaluation gt_reward: {}, irl_reward: {}".format(gt_episode_reward, irl_episode_reward))
file = open('cumilativeRewardlog.txt',"a")
file.write("[INFO] Final evaluation Model: {}, gt_reward: {}, irl_reward: {}".format(args.policy_dir.split("/")[-3], gt_episode_reward, irl_episode_reward))
file.write("\n")
file.close()
