# Finite-state MDP solved using Value Iteration
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import os
from gym import wrappers


OUTPUT_RESULTS_DIR = "../logs"
ENVIRONMENT = 'CartPole-v0'

# TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
TIMESTAMP = 'RESULTS'
SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "CartPoleValueIteration-Observations2", ENVIRONMENT, TIMESTAMP)
env = gym.make(ENVIRONMENT)
env = wrappers.Monitor(env, os.path.join(SUMMARY_DIR, ENVIRONMENT), force=True, video_callable=None)
env.seed(1)


observation = env.reset()#[2:]


observation_dimensions = np.size(observation)
num_actions = env.action_space.n

observation_space_high = env.observation_space.high
observation_space_low = env.observation_space.low


# Hyperparameter
num_bins_per_observation = 4																																																																														 # Could try different number of bins for the different dimensions
num_states = num_bins_per_observation**observation_dimensions


###################################################################################################
#                                     descretizing funcitons                                      #
###################################################################################################

def make_observation_bins(minV, maxV, num_bins):
	if(minV == -np.Inf) or (minV < -10e4):
		minV = -5 # Should really learn this const instead
	if(maxV == np.Inf) or (maxV > 10e4):
		maxV = 5
	bins = np.arange(minV, maxV, (float(maxV)-float(minV))/((num_bins - 2)))
	bins = np.sort(np.append(bins, [0])) # Ensure we split at 0
	return bins

observation_dimension_bins = []
for observation_dimension in range(observation_dimensions):
	observation_dimension_bins.append(make_observation_bins(observation_space_low[observation_dimension], 
							    observation_space_high[observation_dimension], 
							    num_bins_per_observation))
    
print("[INFO]: observation_dimension {} \n high : {} \n low : {}, \n \
		num_bins_per_observation: {}, \n observation_dimension_bins : {}".format(observation_dimensions,\
																				observation_space_high,\
																				observation_space_low,\
																				num_bins_per_observation,\
																				observation_dimension_bins))



###################################################################################################
#                                 state observation functions                                     #
###################################################################################################

def observation_to_state(observation):
	state = 0
	for observation_dimension in range(observation_dimensions):
		state = state + np.digitize(observation[observation_dimension],\
				observation_dimension_bins[observation_dimension])*\
				num_bins_per_observation**observation_dimension
	
	return state
  
print("[INFO]: Min State: {} Max State: {} Num States: {}".format(observation_to_state([-5,-5,-5,-5]), \
							    observation_to_state([5,5,5,5]),
							    num_states))

state_values = np.random.rand(num_states) * 0.1
state_rewards = np.zeros((num_states))
state_transition_probabilities = np.ones((num_states, num_states, num_actions)) / num_states
state_transition_counters = np.zeros((num_states, num_states, num_actions))




###################################################################################################
#                                 best action pickup functions                                    #
###################################################################################################


def pick_best_action(current_state, state_values, state_transition_probabilities):
	best_action = -1
	best_action_value = -np.Inf
	for a_i in range(num_actions):
		action_value = state_transition_probabilities[current_state,:,a_i].dot(state_values)
		if (action_value > best_action_value):
			best_action_value = action_value
			best_action = a_i
		elif (action_value == best_action_value):
			if np.random.randint(0,2) == 0:
				best_action = a_i
	    
	return best_action


def update_state_transition_probabilities_from_counters(probabilities, counters):
	for a_i in range(num_actions):
		for s_i in range(num_states):
			total_transitions_out_of_state = np.sum(counters[s_i,:,a_i])
			if(total_transitions_out_of_state > 0):
				probabilities[s_i,:,a_i] = counters[s_i,:,a_i] / total_transitions_out_of_state
	    
	return probabilities


def run_value_iteration(state_values, state_transition_probabilities, state_rewards):
	gamma = 0.9
	convergence_tolerance = 0.0001
	iteration = 0
	max_dif = np.Inf
	while max_dif > convergence_tolerance:  
		iteration = iteration + 1
		old_state_values = np.copy(state_values)

		best_action_values = np.zeros((num_states)) - np.Inf

		for a_i in range(num_actions):
			best_action_values = np.maximum(best_action_values,\
						state_transition_probabilities[:,:,a_i].dot(state_values))

		state_values = state_rewards + gamma * best_action_values
		max_dif = np.max(np.abs(state_values - old_state_values))       
		
		# print("[INFO ValueIteration]============================")
		# print("Max Value Difference: ", max_dif)
	return state_values
    
	  


###################################################################################################
#                                        iterate loops                                            #
###################################################################################################
maxtval = 10000
eps     = 1.0
mineps  = 0.001
	
episode_rewards = []
for i_episode in range(10000):
	current_observation = env.reset() # [2:]
	current_state = observation_to_state(current_observation)

	episode_reward = 0
	# env.render()

	eps = max(mineps, eps * 0.1)
	if np.random.uniform() < eps: current_state = np.random.randint(0, num_states, 1)

	for t in range(maxtval):
		action = pick_best_action(current_state, state_values, state_transition_probabilities)
		# action = np.random.randint(2)

		old_state = current_state
		observation, reward, done, info = env.step(action)
		# observation = observation[2:]
		current_state = observation_to_state(observation)
		

		state_transition_counters[old_state, current_state, action] += 1

		episode_reward = episode_reward + reward        

		if done or t == maxtval:
			episode_rewards.append(episode_reward)
			print("[INFO Data]============================")
			print("Episode: ", i_episode)
			print("Reward: ", episode_reward)
			print("Mean Reward", np.mean(episode_rewards))
			print("Max reward so far: ", max(episode_rewards))

			
			# Average length of episode is > 195, anything less than than 195 has -ve reward
			# state_rewards[current_state] = (-1 if(t < 195) else 0)
			# state_rewards[current_state] = (-1 if(t < maxtval) else +1)
			state_rewards[current_state] = -1*maxtval/(t+1) if (t < maxtval) else +1 

			state_transition_probabilities = \
				update_state_transition_probabilities_from_counters(state_transition_probabilities,\
											state_transition_counters)
			state_values = run_value_iteration(state_values, state_transition_probabilities, state_rewards)
			break



np.save(os.path.join(SUMMARY_DIR, 'state_values.npy') , state_values)
np.save(os.path.join(SUMMARY_DIR, 'state_transition_probabilities.npy') , state_transition_probabilities)
plt.plot(episode_rewards)
plt.show()