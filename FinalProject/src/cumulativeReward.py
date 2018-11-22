from __future__ import print_function
import numpy as np
import os
import gym
# from gym import wrappers

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='CartPole-v0')
parser.add_argument('--max_episode_len', type=int, default=1000)
parser.add_argument('--max_episode_steps', type=int, default=1000)
parser.add_argument('--policy_dir', type=str)

args = parser.parse_args()
ENVIRONMENT            = args.env_name
MAX_T                  = args.max_episode_len
MAXENVSTEPS            = args.max_episode_steps

env = gym.make(ENVIRONMENT)
env._max_episode_steps = MAXENVSTEPS

# env = wrappers.Monitor(env, args.policy_dir, force=True, video_callable=None)
env.seed(1)

gt_reward = np.load(os.path.join(args.policy_dir, 'gt_state_rewards.npy'))
irl_lp_reward = np.load(os.path.join(args.policy_dir, 'irl_lp_state_rewards.npy'))
# irl_maxentropy_reward = np.load(os.path.join(args.policy_dir, 'irl_maxentropy_state_rewards.npy'))

gt_state_values = np.load(os.path.join(args.policy_dir, 'gt_state_values.npy'))
irl_lp_state_values = np.load(os.path.join(args.policy_dir, 'irl_lp_state_values.npy'))
# irl_maxentropy_state_values = np.load(os.path.join(args.policy_dir, 'irl_maxentropy_state_values.npy'))

gt_state_transition_probabilities = np.load(os.path.join(args.policy_dir, 'gt_state_transition_probabilities.npy'))
irl_lp_state_transition_probabilities = np.load(os.path.join(args.policy_dir, 'irl_lp_state_transition_probabilities.npy'))
# irl_maxentropy_state_transition_probabilities = np.load(os.path.join(args.policy_dir, 'irl_maxentropy_state_transition_probabilities.npy'))

select_observations = lambda O: np.array([O[1],O[2],O[3]])

##########################################################################
#                         Model Evaluation                               #
##########################################################################
def make_observation_bins(minV, maxV, num_bins):
	if(minV == -np.Inf) or (minV < -10e4):
		minV = -5 # Should really learn this const instead
	if(maxV == np.Inf) or (maxV > 10e4):
		maxV = 5
	bins = np.arange(minV, maxV, (float(maxV)-float(minV))/((num_bins - 2)))
	bins = np.sort(np.append(bins, [0])) # Ensure we split at 0
	return bins

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
	
#env.render()
current_observation = env.reset()
current_observation = select_observations(current_observation)
observation_dimensions = np.size(current_observation)
try: num_actions = env.action_space.n
except: num_actions = env.action_space.shape[0]
observation_space_high = env.observation_space.high
observation_space_low = env.observation_space.low

NUMBER_OF_BINS = int(args.policy_dir.split('/')[2].split("_").pop())
num_states = NUMBER_OF_BINS**observation_dimensions
observation_dimension_bins = []
for observation_dimension in range(observation_dimensions):
	observation_dimension_bins.append(make_observation_bins(observation_space_low[observation_dimension], 
							observation_space_high[observation_dimension], 
								NUMBER_OF_BINS))
current_state = observation_to_state(current_observation)
gt_episode_reward = 0


while True:
	action = pick_best_action(current_state, gt_state_values, gt_state_transition_probabilities)
	old_state = current_state
	observation, reward, done, info = env.step(action)
	gt_episode_reward = gt_episode_reward + reward
	current_state = observation_to_state(select_observations(observation))
	if done: 
		#env.close()
		break

#env.render()
current_observation = env.reset()
current_observation = select_observations(current_observation)
current_state = observation_to_state(current_observation)
irl_lp_episode_reward = 0

while True:
	action = pick_best_action(current_state, irl_lp_state_values, irl_lp_state_transition_probabilities)
	old_state = current_state
	observation, reward, done, info = env.step(action)
	irl_lp_episode_reward = irl_lp_episode_reward + reward
	current_state = observation_to_state(select_observations(observation))
	if done: 
		#env.close()
		break
		
#env.render()
current_observation = env.reset()
current_observation = select_observations(current_observation)
current_state = observation_to_state(current_observation)
irl_maxentropy_episode_reward = 0

"""
while True:
	action = pick_best_action(current_state, irl_maxentropy_state_values, irl_maxentropy_state_transition_probabilities)
	old_state = current_state
	observation, reward, done, info = env.step(action)
	irl_maxentropy_episode_reward = irl_maxentropy_episode_reward + reward
	current_state = observation_to_state(select_observations(observation))
	if done: break
        env.close()
"""


print ("[INFO] Final evaluation gt_reward: {}, irl_lp_reward: {}, irl_,maxentropy_reward: {}".format(gt_episode_reward, irl_lp_episode_reward, irl_maxentropy_episode_reward))
file = open('cumilativeRewardlog.txt',"a")
file.write("[INFO] Final evaluation Model: {}, gt_reward: {}, irl_lp_reward: {}, irl_,maxentropy_reward: {}".format(args.policy_dir.split("/")[-3], gt_episode_reward, irl_lp_episode_reward, irl_maxentropy_episode_reward))
file.write("\n")
file.close()
