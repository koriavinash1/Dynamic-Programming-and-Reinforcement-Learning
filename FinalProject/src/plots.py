import matplotlib.pyplot as plt 
import numpy as np
import os

def moving_average(a, n=3) :
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n

pre_path = '../logs/'
post_path = 'CartPole-v0/RESULTS/gt_mean_episode_rewards.npy'

exp_legend = ['ValueIteration_No_Bins_5',
			  'ValueIteration_No_Bins_7',
			  'ValueIteration_No_Bins_9',
			  'ValueIteration_No_Bins_11']

# exp_legend = ['InversePolicyIteration_No_Bins_5',
# 			  'InversePolicyIteration_No_Bins_7',
#          		# 'ModifiedPolicyIteration_No_Bins_5',
#            #     'ModifiedPolicyIteration_No_Bins_7',
#                'ModifiedPolicyIteration_No_Bins_9',
#                'ModifiedPolicyIteration_No_Bins_11']
for i_exp in exp_legend:
	path = os.path.join(pre_path, i_exp, post_path)
	print ("[INFO]: ", path)
	mean_rewards = np.load(path)
	# mean_rewards = moving_average(np.load(path),25)
	if i_exp.__contains__('Inverse'):
		plt.plot(mean_rewards, linestyle='--', linewidth=3.0)
	else:
		plt.plot(mean_rewards, linewidth=3.0)

plt.legend(exp_legend)
plt.grid(True)
plt.xlabel('episodes')
plt.ylabel('mean_rewards')
# plt.title('PolicyIteration Discretization effect')
plt.title('ValueIteration Discretization effect')
plt.show()
# plt.savefig('ValueIteration_Exp.png')