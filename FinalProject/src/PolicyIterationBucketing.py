import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import os
from gym import wrappers
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




def run_policy_iteration(state_values, state_transition_probabilities, state_rewards):
		P =  np.transpose(np.copy(state_transition_probabilities),(0,2,1))
		n_states = len(state_values)
		reward = np.copy(state_rewards)
		n_actions = P.shape[1]
		policy = np.zeros(n_states,dtype=int)
		iter = 0 
		while(True):
			iter+=1
			#print("Iteration : ",iter)
			
			#Policy Evaluation
			A_mat = np.zeros([n_states,n_states])
			C_mat = np.zeros([n_states,1])

			if(iter%100==0):
				print("Iterations:",iter)

			for state in range(n_states):
				# C_mat[state] = np.sum(P[state][policy[state]]*reward[state])
				A_mat[state] = np.eye(num_states)[state] - GAMMA * P[state][policy[state]]

			C_mat = reward.reshape(n_states, 1)
			J = np.matmul(np.linalg.inv(A_mat),C_mat)
			
			#Policy Improvement
			new_policy = np.zeros(n_states,dtype=int)
			val_mat = np.zeros(n_actions)
			opt_vals = np.zeros(n_states)

			for state in range(n_states):
				for action in range(n_actions):
					val_mat[action] = np.sum(reward[state]+ GAMMA*P[state][action]*(np.reshape(J,[1,n_states])))
				
				opt_vals[state] = np.max(val_mat)
				new_policy[state] = np.argmax(val_mat)
			
			# print (new_policy, policy)
			#Convergence condition
			if(np.sum(policy - new_policy) == 0):
				print("Optimal Policy using Policy Iteration : ", new_policy)
				break

			#Update policy
			policy = np.copy(new_policy)
		# return np.round(opt_vals,2),new_policy
		return opt_vals
	# def policy_evaluation():
	# 	pass

	# def policy_update():
	# 	pass
		# return state_values
		

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
	parser.add_argument('--num_episodes', type=int, default=150)
	parser.add_argument('--max_episode_len', type=int, default=10000)
	parser.add_argument('--max_episode_steps', type=int, default=10000)
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--expl_rate', type=float, default=1.0)
	parser.add_argument('--min_expl_rate', type=float, default=0.2)
	parser.add_argument('--expl_rate_decay', type=float, default=0.010)
	parser.add_argument('--num_bins', type=int, default=4)
	parser.add_argument('--log_dir', type=str, default="../logs/")
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
	TIMESTAMP              = datetime.now().strftime("%Y%m%d-%H%M%S")
	# TIMESTAMP = 'RESULTS'
	
	episode_rewards = []
	mean_reward     = []
	SUMMARY_DIR     = os.path.join(OUTPUT_RESULTS_DIR,  EXPNAME + "_No_Bins_" + str(NUMBER_OF_BINS),\
										 ENVIRONMENT, TIMESTAMP)


	if not os.path.exists(SUMMARY_DIR):
			os.makedirs(SUMMARY_DIR)

	env = gym.make(ENVIRONMENT)
	env._max_episode_steps = MAXENVSTEPS

	env = wrappers.Monitor(env, os.path.join(SUMMARY_DIR, ENVIRONMENT), force=True, video_callable=None)
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
			env.render()
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
				if not done:
					state_rewards[current_state] = 0.1            
				elif done or t == MAX_T-1:
					episode_rewards.append(episode_reward)
					mean_reward.append(np.mean(episode_rewards))

					if args.verbose:
						print("[INFO Data {}]============================".format(t))
						print("Episode: ", i_episode)
						print("Reward: ", episode_reward)
						print("Mean Reward: ", np.mean(episode_rewards))
						print("Max reward so far: ", max(episode_rewards))

					if t < 195:
						state_rewards[current_state] = -1
					elif t < 300:
						state_rewards[current_state] = 1 
					else:
						state_rewards[current_state] = 2

					state_transition_probabilities = \
						update_state_transition_probabilities_from_counters(state_transition_probabilities,\
													state_transition_counters)
					state_values = run_policy_iteration(state_values, state_transition_probabilities, state_rewards)
					env.close()
					break


			if i_episode % 20 == 19:
				np.save(os.path.join(SUMMARY_DIR, 'state_values.npy') , state_values)
				np.save(os.path.join(SUMMARY_DIR, 'state_rewards.npy') , state_rewards)
				np.save(os.path.join(SUMMARY_DIR, 'state_transition_probabilities.npy') , state_transition_probabilities)
				print("[INFO] Model Saved Successfully ... ")


		end_time = time()
		episode_rewards = moving_average(episode_rewards, n = 25)
		episode_rewards[0] = mean_reward[30]
		plt.plot(episode_rewards)
		plt.plot(mean_reward[30:])
		plt.title('Value Iteration Reward Convergence for '+ str(NUMBER_OF_BINS) +' Bins')
		plt.legend(['Episode reward with smoothening widow of n = 25', 'Mean episode reward'])
		plt.ylabel('Reward')
		plt.xlabel('Episodes')
		plt.savefig(os.path.join(SUMMARY_DIR, 'mean_epi_plot_training_time_'+\
							str(end_time - start_time)+'.png'))
		
		plt.clf()
		plt.plot(episode_rewards)
		plt.title('Value Iteration Reward Convergence for '+ str(NUMBER_OF_BINS) +' Bins')
		plt.legend(['Episode reward with smoothening widow of n = 25'])
		plt.ylabel('Reward')
		plt.xlabel('Episodes')
		plt.savefig(os.path.join(SUMMARY_DIR, 'epi_plot_training_time_'+\
							str(end_time - start_time)+'.png'))
		
		plt.clf()
		plt.plot(mean_reward[30:])
		plt.title('Value Iteration Reward Convergence for '+ str(NUMBER_OF_BINS) +' Bins')
		plt.legend(['Mean episode reward'])
		plt.ylabel('Reward')
		plt.xlabel('Episodes')
		plt.savefig(os.path.join(SUMMARY_DIR, 'mean_plot_training_time_'+\
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