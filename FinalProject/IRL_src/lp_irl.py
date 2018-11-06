'''
Implementation of linear programming inverse reinforcement learning in
  Ng & Russell 2000 paper: Algorithms for Inverse Reinforcement Learning
  http://ai.stanford.edu/~ang/papers/icml00-irl.pdf

By Yiren Lu (luyirenmax@gmail.com), May 2017
'''
from __future__ import print_function
import numpy as np
from cvxopt import matrix, solvers
from utils import *

import pickle




#def lp_irl(trans_probs, policy, gamma=0.5, l1=10, R_max=10):
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
  print(np.shape(trans_probs))
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
  path = "/home/hari/Github_repo/Dynamic-Programming-and-Reinforcement-Learning/FinalProject/logs/IRL/"
  file_name = "ARGS1.txt" #"ARGS.txt"
  file = open(path+file_name,"r")
  args = pickle.load(file)
  # print(args)
  print("No: of states",np.shape(args[0]))
  P_s = args[0];  policy = args[1]; 
  try:
     gamma = args[2];  l = args[3];  R_max = args[4]
     rewards = lp_irl(P_s,policy,gamma,l,R_max)
  except Exception as e:
     #Actual rewards saved as args[2]
     rewards = lp_irl(P_s,policy)
  

