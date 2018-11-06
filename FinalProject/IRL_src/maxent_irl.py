'''
Implementation of maximum entropy inverse reinforcement learning in
  Ziebart et al. 2008 paper: Maximum Entropy Inverse Reinforcement Learning
  https://www.aaai.org/Papers/AAAI/2008/AAAI08-227.pdf

Acknowledgement:
  This implementation is largely influenced by Matthew Alger's maxent implementation here:
  https://github.com/MatthewJA/Inverse-Reinforcement-Learning/blob/master/irl/maxent.py

By Yiren Lu (luyirenmax@gmail.com), May 2017
'''
#from __future__ import print_function
import numpy as np
#import mdp.gridworld as gridworld
import ValueIteration as value_iteration
import img_utils
from utils import *
import dill
import pickle
import gym
import matplotlib.pyplot as plt
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
  path = "../logs/IRL/Max_entropy/"
  P_s=np.load(path+'Trans_prob.npy')
  trajs = np.load(path+'Trajs.npy')
  rew = np.load(path+'Rewards_Gt.npy')
  trajs = conv_trajs_to_reqd_format(trajs)
  #P_s = args[0]; trajs = args[1] ;
  shape_Ps = np.shape(P_s)
  print(shape_Ps)
  feat_map = np.eye(shape_Ps[0])
  try:
    gamma = args[2];lr = args[3]; n_iters = args[4]
    norm_rewards = maxent_irl(feat_map, P_s, gamma, trajs, lr, n_iters)
  except Exception as e:
    norm_rewards = maxent_irl(feat_map, P_s,0.9,trajs)

  plt.plot(norm_rewards, 'r')
  plt.plot(rew, 'b')
  plt.legend(['true', 'gt'])
  plt.show()