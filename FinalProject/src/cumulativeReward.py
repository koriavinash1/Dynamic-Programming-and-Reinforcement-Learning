from __future__ import print_function
import numpy as np
import os

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

gt_reward = np.load(os.path.join(args.policy_dir, 'Rewards_GT.npy'))
irl_reward = np.load(os.path.join(args.policy_dir, 'IRL_rewards.npy'))

optimal_policy = np.load(os.path.join(args.policy_dir, 'Optimal_policy.npy'))
state_trans_probab = np.load(os.path.join(args.policy_dir, 'Trans_prob.npy'))



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
	
state_values = np.load(os.path.join(SUMMARY_DIR, 'state_values.npy'))
state_transition_probabilities = np.load(os.path.join(SUMMARY_DIR, 'state_transition_probabilities.npy'))
# env.seed(1)
env.render()
current_observation = env.reset()
current_observation = select_observations(current_observation)
current_state = observation_to_state(current_observation)
episode_reward = 0

while True:
	action = pick_best_action(current_state, state_values, state_transition_probabilities)
	old_state = current_state
	observation, reward, done, info = env.step(action)
	episode_reward = episode_reward + reward
	current_state = observation_to_state(select_observations(observation))
	if done: break

print ("[INFO] Final evaluation reward: {}".format(episode_reward))
env.close()