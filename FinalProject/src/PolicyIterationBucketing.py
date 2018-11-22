import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import os
# from gym import wrappers
import pickle
from collections import namedtuple
import dill
from time import time 
from datetime import datetime
from tqdm import tqdm

select_observations = lambda O: np.array([O[1],O[2],O[3]])
###################################################################################################
#                                     descretizing funcitons                                      #
###################################################################################################
def moving_average(a, n=3) :
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n


def make_observation_bins(minV, maxV, num_bins):
	if(minV == -np.Inf) or (minV < -10e4):
		minV = -5 # Should really learn this const instead
	if(maxV == np.Inf) or (maxV > 10e4):
		maxV = 5
	bins = np.arange(minV, maxV, (float(maxV)-float(minV))/((num_bins - 2)))
	bins = np.sort(np.append(bins, [0])) # Ensure we split at 0
	return bins


###################################################################################################
#                                 state observation functions                                     #
###################################################################################################

def observation_to_state(observation):
	state = 0
	for observation_dimension in range(observation_dimensions):
		state = state + np.digitize(observation[observation_dimension],\
					observation_dimension_bins[observation_dimension])*\
					NUMBER_OF_BINS**observation_dimension 
	return state
	

###################################################################################################
#                                 best action pickup functions                                    #
###################################################################################################


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


def update_state_transition_probabilities_from_counters(probabilities, counters):
	for a_i in range(num_actions):
		for s_i in range(num_states):
			total_transitions_out_of_state = np.sum(counters[s_i,:,a_i])
			if(total_transitions_out_of_state > 0):
				probabilities[s_i,:,a_i] = counters[s_i,:,a_i] / total_transitions_out_of_state
			
	return probabilities



def run_policy_iteration(state_values, state_transition_probabilities, state_rewards, M = 100):
	P =  np.transpose(np.copy(state_transition_probabilities),(0,2,1))
	reward = np.copy(state_rewards)
	n_states = P.shape[0]
	n_actions = P.shape[1]
	policy = np.zeros(n_states,dtype=int)
	
	iter = 0 
	while(True):
		iter += 1

		#Policy Evaluation
		Ppi = np.zeros([n_states, n_states])
		for state in range(n_states):
			Ppi[state, :] = P[state, policy[state], :]
		
		if M:
			Jpi = np.zeros(n_states)
			for _ in range(M):
				Jpi = reward + np.max(GAMMA*np.einsum('ijk, k -> ij', P, Jpi))
		else:
			Jpi = np.linalg.inv(np.eye(n_states) - GAMMA * Ppi).dot(reward.reshape(n_states, 1))

		#Policy Improvement
		opt_vals   = np.zeros(n_states)
		new_policy = np.zeros(n_states,dtype=int)

		for state in range(n_states):
			val_mat = reward[state]+ np.array([GAMMA*np.sum(P[state][action]*Jpi) for action in range(n_actions)])
			opt_vals[state]   = np.max(val_mat)
			new_policy[state] = np.argmax(val_mat)
		
		#Convergence condition
		if(np.sum(policy - new_policy) == 0 or iter == 150):
			# print("Optimal Policy using Policy Iteration : new_policy :{}, old_policy: {} ".format(new_policy, policy))
			break
		#Update policy
		policy = np.copy(new_policy)

	return opt_vals
		

def generate_demonstrations(n_trajs=10, len_traj=500):

	"""gatheres expert demonstrations

	inputs:
	gw          Gridworld - the environment
	policy      Nx1 matrix
	n_trajs     int - number of trajectories to generate
	rand_start  bool - randomly picking start position or not
	start_pos   2x1 list - set start position, default [0,0]
	returns:
	trajs       a list of trajectories - each element in the list is a list of Steps representing an episode
	"""
	Step = namedtuple('Step','cur_state action next_state reward done')
	Step.__module__ = '__main__'
	trajs = []
	for i in range(n_trajs):
		episode = []
		current_observation = env.reset() 
		current_observation = select_observations(current_observation)
		current_state = observation_to_state(current_observation)
		action = pick_best_action(current_state,state_values,state_transition_probabilities)
		next_state, reward, is_done,info = env.step(action)
		#episode.append(namedtuple(cur_state=current_state, action=action,\ 
		# next_state=observation_to_state(select_observations(next_state)), reward=reward, done=is_done))
		episode.append(tuple([current_state,action,observation_to_state(select_observations(next_state)),reward,is_done]))
		for _ in range(len_traj):
			current_observation = select_observations(next_state)
			current_state = observation_to_state(current_observation)
			action = pick_best_action(current_state,state_values,state_transition_probabilities)
			next_state, reward, is_done,info = env.step(action)
			episode.append(tuple([current_state,action,observation_to_state(select_observations(next_state)),reward,is_done]))
			#episode.append(namedtuple(cur_state=current_state, action=action,\ 
			#	next_state=observation_to_state(select_observations(next_state)), reward=reward, done=is_done))
			if is_done:
				break
		trajs.append(episode)
	return trajs			

if __name__ == "__main__":
	
	###################################################################################################
	#                                        Args Selection                                           #
	###################################################################################################
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--env_name', type=str, default='CartPole-v0')
	parser.add_argument('--exp_name', type=str, default='PolicyIteration')
	parser.add_argument('--num_episodes', type=int, default=50)
	parser.add_argument('--max_episode_len', type=int, default=1000)
	parser.add_argument('--max_episode_steps', type=int, default=1000)
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--expl_rate', type=float, default=1.0)
	parser.add_argument('--min_expl_rate', type=float, default=0.2)
	parser.add_argument('--expl_rate_decay', type=float, default=0.010)
	parser.add_argument('--num_bins', type=int, default=4)
	parser.add_argument('--modified_policy_iteration', type=int, default=500) 
	parser.add_argument('--policy_iteration_type', type=str, default="Modified") 	 	
	parser.add_argument('--log_dir', type=str, default="../logs/")
	parser.add_argument('--reward_type', type=str, default="gt")
	parser.add_argument('--irl_reward_path', type=str, default="NA")
	parser.add_argument('--train', type=int, default=1)
	parser.add_argument('--verbose', type=int, default=1)
	args = parser.parse_args()

	print ("[INFO] ", args)
	ENVIRONMENT            = args.env_name
	EXPNAME                = args.exp_name
	NUM_EPISODES           = args.num_episodes
	MAX_T                  = args.max_episode_len
	MAXENVSTEPS            = args.max_episode_steps
	GAMMA                  = args.gamma
	EXPLORATION_RATE       = args.expl_rate
	MIN_EXPLORATION_RATE   = args.min_expl_rate
	EXPLORATION_RATE_DECAY = args.expl_rate_decay
	Train                  = args.train
	NUMBER_OF_BINS         = args.num_bins
	OUTPUT_RESULTS_DIR     = args.log_dir
	# TIMESTAMP              = datetime.now().strftime("%Y%m%d-%H%M%S")
	TIMESTAMP = 'RESULTS'
	if args.policy_iteration_type == "Modified":
		M = args.modified_policy_iteration
	else: 
		M = None
		if NUMBER_OF_BINS > 7:
			raise ValueError("Inverse cann't be calculated, set method to modified policy iteration.")
	
	if args.reward_type != 'gt':
		if args.irl_reward_path == 'NA':
			raise ValueError("IRL Reward path not given.")
		#TIMESTAMP = args.irl_reward_path.split("/")[-3]
		if args.reward_type == 'irl_lp':
			irl_lp_reward = np.load(os.path.join(args.irl_reward_path, 'IRL_rewards.npy'))
		elif args.reward_type == 'irl_maxentropy':
			irl_max_entropy_reward = np.load(os.path.join(args.irl_reward_path, 'IRL_rewards.npy'))


	episode_rewards = []
	mean_reward     = []
	SUMMARY_DIR     = os.path.join(OUTPUT_RESULTS_DIR,  args.policy_iteration_type + EXPNAME + "_No_Bins_" + str(NUMBER_OF_BINS),\
										 ENVIRONMENT, TIMESTAMP)


	if not os.path.exists(SUMMARY_DIR):
		os.makedirs(SUMMARY_DIR)

	env = gym.make(ENVIRONMENT)
	env._max_episode_steps = MAXENVSTEPS

	# env = wrappers.Monitor(env, os.path.join(SUMMARY_DIR, ENVIRONMENT), force=True, video_callable=None)
	env.seed(1)

	observation = env.reset()
	observation = select_observations(observation)


	observation_dimensions = np.size(observation)
	try: num_actions = env.action_space.n
	except: num_actions = env.action_space.shape[0]

	observation_space_high = env.observation_space.high
	observation_space_low = env.observation_space.low


	num_states = NUMBER_OF_BINS**observation_dimensions


	observation_dimension_bins = []
	for observation_dimension in range(observation_dimensions):
		observation_dimension_bins.append(make_observation_bins(observation_space_low[observation_dimension], 
								observation_space_high[observation_dimension], 
									NUMBER_OF_BINS))
			
	print("[INFO]: observation_dimension {} \n high : {} \n low : {}, \n \
		NUMBER_OF_BINS: {}, \n observation_dimension_bins : {}".format(observation_dimensions,\
										observation_space_high,\
										observation_space_low,\
										NUMBER_OF_BINS,\
										observation_dimension_bins))


	print("[INFO]: Min State: {} Max State: {} Num States: {}".format(observation_to_state([-10,-10,10,-10.5]), \
									observation_to_state([10,10,10,10.5]),
									num_states))



	state_values = np.random.rand(num_states) * 0.1
	state_rewards = np.zeros((num_states))
	state_transition_probabilities = np.ones((num_states, num_states, num_actions)) / num_states
	state_transition_counters = np.zeros((num_states, num_states, num_actions))


	###################################################################################################
	#                                        iterate loops                                            #
	###################################################################################################


	if Train:
		start_time = time()

		for i_episode in tqdm(range(NUM_EPISODES)):
			current_observation = env.reset()
			current_observation = select_observations(current_observation)
			current_state = observation_to_state(current_observation)

			episode_reward = 0
			# env.render()
			if i_episode % 50 == 49: EXPLORATION_RATE = max(MIN_EXPLORATION_RATE, EXPLORATION_RATE * 0.1)

			if np.random.uniform() <= EXPLORATION_RATE: current_state = np.random.randint(0, num_states, 1)

			for t in range(MAX_T):
				action = pick_best_action(current_state, state_values, state_transition_probabilities)

				old_state = current_state
				observation, reward, done, info = env.step(action)
				observation = select_observations(observation)
				current_state = observation_to_state(observation)
				

				state_transition_counters[old_state, current_state, action] += 1

				episode_reward = episode_reward + reward        
				st_time = time()
				     
				if done or t == MAX_T-1:
					episode_rewards.append(episode_reward)
					mean_reward.append(np.mean(episode_rewards))

					if args.verbose:
						print("[INFO Data {}]============================".format(t))
						print("Episode: ", i_episode)
						print("Reward: ", episode_reward)
						print("Mean Reward: ", np.mean(episode_rewards))
						print("Max reward so far: ", max(episode_rewards))

					if args.reward_type == 'gt':
						if t < 195:
							state_rewards[current_state] = -1
						elif t < 300:
							state_rewards[current_state] = 1 
						else:
							state_rewards[current_state] = 2
					elif args.reward_type == 'irl_lp':
						state_rewards[current_state] == irl_lp_reward[current_state]
						
					elif args.reward_type == 'irl_maxentropy':
						state_rewards[current_state] == irl_max_entropy_reward[current_state]

					else:
						raise ValueError("Invalid reward_type")

					state_transition_probabilities = \
						update_state_transition_probabilities_from_counters(state_transition_probabilities,\
													state_transition_counters)
					state_values = run_policy_iteration(state_values, state_transition_probabilities, state_rewards, M)
					# env.close()
					break


			if i_episode % 20 == 19:
				np.save(os.path.join(SUMMARY_DIR, args.reward_type + '_state_values.npy') , state_values)
				np.save(os.path.join(SUMMARY_DIR, args.reward_type + '_state_rewards.npy') , state_rewards)
				np.save(os.path.join(SUMMARY_DIR, args.reward_type + '_state_transition_probabilities.npy') , state_transition_probabilities)
				if args.verbose: print("[INFO] Model Saved Successfully ... ")

			# terminaltion condition
			# print (np.mean(np.array(episode_rewards)[-10:]))
			if np.mean(np.array(episode_rewards)[-15:]) == MAXENVSTEPS and i_episode > 32: break


		end_time = time()
		episode_rewards = moving_average(episode_rewards, n = 20)
		episode_rewards[0] = mean_reward[5]
		plt.plot(episode_rewards)
		plt.plot(mean_reward[5:])
		plt.title('Value Iteration Reward Convergence for '+ str(NUMBER_OF_BINS) +' Bins')
		plt.legend(['Episode reward with smoothening widow of n = 10', 'Mean episode reward'])
		plt.ylabel('Reward')
		plt.xlabel('Episodes')
		plt.savefig(os.path.join(SUMMARY_DIR, args.reward_type + 'mean_epi_plot_training_time_'+\
							str(end_time - start_time)+'.png'))
		
		plt.clf()
		plt.plot(episode_rewards)
		plt.title('Value Iteration Reward Convergence for '+ str(NUMBER_OF_BINS) +' Bins')
		plt.legend(['Episode reward with smoothening widow of n = 10'])
		plt.ylabel('Reward')
		plt.xlabel('Episodes')
		plt.savefig(os.path.join(SUMMARY_DIR, args.reward_type + 'epi_plot_training_time_'+\
							str(end_time - start_time)+'.png'))
		
		plt.clf()
		plt.plot(mean_reward[5:])
		plt.title('Value Iteration Reward Convergence for '+ str(NUMBER_OF_BINS) +' Bins')
		plt.legend(['Mean episode reward'])
		plt.ylabel('Reward')
		plt.xlabel('Episodes')
		plt.savefig(os.path.join(SUMMARY_DIR, args.reward_type + 'mean_plot_training_time_'+\
							str(end_time - start_time)+'.png'))
		

	else:

		##########################################################################
		#                         Model Evaluation                               #
		##########################################################################
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



	if args.reward_type == 'gt':
		###################################################################################################
		#                                 Code to write details for LP IRL                                #
		###################################################################################################

		print("[INFO] Writing IRL LP File ....")
		policy = np.array([pick_best_action(i,state_values,state_transition_probabilities) for i in range(num_states)])
		LPPATH = os.path.join(SUMMARY_DIR, "IRL/LP/")
		if not os.path.exists(LPPATH): os.makedirs(LPPATH)
		filename = "ARGS1.txt"
		file = open(LPPATH+filename,"wb")
		args = [state_transition_probabilities, policy, state_rewards]
		pickle.dump(args, file, protocol=2)
		file.close()
		np.save(LPPATH+'Trans_prob',state_transition_probabilities)
		np.save(LPPATH+'Optimal_policy', policy)
		np.save(LPPATH+'Rewards_Gt',state_rewards)


		###################################################################################################
		#                            Code to write details for MAXENTROPY IRL                             #
		###################################################################################################

		print("[INFO] Writing MAXENTROPY LP File ....")



		MAXENTROPYPATH = os.path.join(SUMMARY_DIR, "IRL/Max_entropy/")
		if not os.path.exists(MAXENTROPYPATH): os.makedirs(MAXENTROPYPATH)
		np.save(MAXENTROPYPATH+'Trans_prob',state_transition_probabilities)
		np.save(MAXENTROPYPATH+'Trajs',generate_demonstrations(),allow_pickle=True)
		np.save(MAXENTROPYPATH+'Rewards_Gt',state_rewards)
