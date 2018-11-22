# ValueIteration 
# python3 ValueIterationBucketing.py --num_episodes 500 --num_bins 5 --verbose 0
# python3 ValueIterationBucketing.py --num_episodes 500 --num_bins 7 --verbose 0
python3 ValueIterationBucketing.py --num_episodes 500 --num_bins 9 --verbose 0
# python3 ValueIterationBucketing.py --num_episodes 500 --num_bins 11 --verbose 0
# python3 ValueIterationBucketing.py --num_episodes 500 --num_bins 13 --verbose 0

# Policy Iteration
# python3 PolicyIterationBucketing.py --policy_iteration_type 'Inverse' --num_episodes 500 --num_bins 5 --verbose 0
# python3 PolicyIterationBucketing.py --policy_iteration_type 'Inverse' --num_episodes 500 --num_bins 7 --verbose 0
python3 PolicyIterationBucketing.py --policy_iteration_type 'Inverse' --num_episodes 500 --num_bins 9 --verbose 0
# python3 PolicyIterationBucketing.py --num_episodes 500 --num_bins 5 --verbose 0
# python3 PolicyIterationBucketing.py --num_episodes 500 --num_bins 7 --verbose 0
python3 PolicyIterationBucketing.py --num_episodes 500 --num_bins 9 --verbose 0
# python3 PolicyIterationBucketing.py --num_episodes 500 --num_bins 11 --verbose 0
# python3 PolicyIterationBucketing.py --num_episodes 500 --num_bins 13 --verbose 0

# Policy Gradient
# python3 montecarloPG.py --exp_name MCPG50 --num_episodes 1500 --verbose 0

# LP IRL Value Iteration run
# python lp_irl.py --policy_dir '../logs/ValueIteration_No_Bins_5/CartPole-v0/RESULTS/IRL/LP/'
# python lp_irl.py --policy_dir '../logs/ValueIteration_No_Bins_7/CartPole-v0/RESULTS/IRL/LP/'
python lp_irl.py --policy_dir '../logs/ValueIteration_No_Bins_7/CartPole-v0/RESULTS/IRL/LP/'
# python lp_irl.py --policy_dir '../logs/ValueIteration_No_Bins_11/CartPole-v0/RESULTS/IRL/LP/'
# python lp_irl.py --policy_dir '../logs/ValueIteration_No_Bins_13/CartPole-v0/RESULTS/IRL/LP/'


# LP IRL Policy Iteration run
# python lp_irl.py --policy_dir '../logs/InversePolicyIteration_No_Bins_5/CartPole-v0/RESULTS/IRL/LP/' 
# python lp_irl.py --policy_dir '../logs/InversePolicyIteration_No_Bins_7/CartPole-v0/RESULTS/IRL/LP' 
python lp_irl.py --policy_dir '../logs/InversePolicyIteration_No_Bins_9/CartPole-v0/RESULTS/IRL/LP' 
# python lp_irl.py --policy_dir '../logs/ModifiedPolicyIteration_No_Bins_5/CartPole-v0/RESULTS/IRL/LP'
# python lp_irl.py --policy_dir '../logs/ModifiedPolicyIteration_No_Bins_7/CartPole-v0/RESULTS/IRL/LP'
python lp_irl.py --policy_dir '../logs/ModifiedPolicyIteration_No_Bins_9/CartPole-v0/RESULTS/IRL/LP'
# python lp_irl.py --policy_dir '../logs/ModifiedPolicyIteration_No_Bins_11/CartPole-v0/RESULTS/IRL/LP'
# python lp_irl.py --policy_dir '../logs/ModifiedPolicyIteration_No_Bins_13/CartPole-v0/RESULTS/IRL/LP'


# run maxentropy

# ValueIteration with IRL agent
# python3 ValueIterationBucketing.py --num_episodes 500 --num_bins 5 --verbose 0 --reward_type 'irl_lp' --irl_reward_path '../logs/ValueIteration_No_Bins_5/CartPole-v0/RESULTS/IRL/LP'
# python3 ValueIterationBucketing.py --num_episodes 500 --num_bins 7 --verbose 0 --reward_type 'irl_lp' --irl_reward_path '../logs/ValueIteration_No_Bins_7/CartPole-v0/RESULTS/IRL/LP'
python3 ValueIterationBucketing.py --num_episodes 500 --num_bins 9 --verbose 0 --reward_type 'irl_lp' --irl_reward_path '../logs/ValueIteration_No_Bins_9/CartPole-v0/RESULTS/IRL/LP'
# python3 ValueIterationBucketing.py --num_episodes 500 --num_bins 11 --verbose 0 --reward_type 'irl_lp' --irl_reward_path '../logs/ValueIteration_No_Bins_11/CartPole-v0/RESULTS/IRL/LP'
# python3 ValueIterationBucketing.py --num_episodes 500 --num_bins 13 --verbose 0 --reward_type 'irl_lp' --irl_reward_path '../logs/ValueIteration_No_Bins_13/CartPole-v0/RESULTS/IRL/LP'

# Policy Iteration with IRL agent
# python3 PolicyIterationBucketing.py --policy_iteration_type 'Inverse' --num_episodes 500 --num_bins 5 --verbose 0 --reward_type 'irl_lp' --irl_reward_path '../logs/InversePolicyIteration_No_Bins_5/CartPole-v0/RESULTS/IRL/LP/'
# python3 PolicyIterationBucketing.py --policy_iteration_type 'Inverse' --num_episodes 500 --num_bins 7 --verbose 0 --reward_type 'irl_lp' --irl_reward_path '../logs/InversePolicyIteration_No_Bins_7/CartPole-v0/RESULTS/IRL/LP/'
python3 PolicyIterationBucketing.py --policy_iteration_type 'Inverse' --num_episodes 500 --num_bins 9 --verbose 0 --reward_type 'irl_lp' --irl_reward_path '../logs/InversePolicyIteration_No_Bins_9/CartPole-v0/RESULTS/IRL/LP/'
# python3 PolicyIterationBucketing.py --num_episodes 500 --num_bins 5 --verbose 0 --reward_type 'irl_lp' --irl_reward_path '../logs/ModifiedPolicyIteration_No_Bins_5/CartPole-v0/RESULTS/IRL/LP'
# python3 PolicyIterationBucketing.py --num_episodes 500 --num_bins 7 --verbose 0 --reward_type 'irl_lp' --irl_reward_path '../logs/ModifiedPolicyIteration_No_Bins_7/CartPole-v0/RESULTS/IRL/LP'
python3 PolicyIterationBucketing.py --num_episodes 500 --num_bins 9 --verbose 0 --reward_type 'irl_lp' --irl_reward_path '../logs/ModifiedPolicyIteration_No_Bins_9/CartPole-v0/RESULTS/IRL/LP'
# python3 PolicyIterationBucketing.py --num_episodes 500 --num_bins 11 --verbose 0 --reward_type 'irl_lp' --irl_reward_path '../logs/ModifiedPolicyIteration_No_Bins_11/CartPole-v0/RESULTS/IRL/LP'
# python3 PolicyIterationBucketing.py --num_episodes 500 --num_bins 13 --verbose 0 --reward_type 'irl_lp' --irl_reward_path '../logs/ModifiedPolicyIteration_No_Bins_13/CartPole-v0/RESULTS/IRL/LP'

# run value and policy iteration with maxentropy

# IRL Cumilative Cost Comparision
python3 cumulativeReward.py --policy_dir '../logs/ValueIteration_No_Bins_5/CartPole-v0/RESULTS'
python3 cumulativeReward.py --policy_dir '../logs/ValueIteration_No_Bins_7/CartPole-v0/RESULTS'
python3 cumulativeReward.py --policy_dir '../logs/ValueIteration_No_Bins_9/CartPole-v0/RESULTS'
python3 cumulativeReward.py --policy_dir '../logs/ValueIteration_No_Bins_11/CartPole-v0/RESULTS'
python3 cumulativeReward.py --policy_dir '../logs/ValueIteration_No_Bins_13/CartPole-v0/RESULTS'

python3 cumulativeReward.py --policy_dir '../logs/InversePolicyIteration_No_Bins_5/CartPole-v0/RESULTS'
python3 cumulativeReward.py --policy_dir '../logs/InversePolicyIteration_No_Bins_7/CartPole-v0/RESULTS'
python3 cumulativeReward.py --policy_dir '../logs/InversePolicyIteration_No_Bins_9/CartPole-v0/RESULTS'
python3 cumulativeReward.py --policy_dir '../logs/ModifiedPolicyIteration_No_Bins_5/CartPole-v0/RESULTS'
python3 cumulativeReward.py --policy_dir '../logs/ModifiedPolicyIteration_No_Bins_7/CartPole-v0/RESULTS'
python3 cumulativeReward.py --policy_dir '../logs/ModifiedPolicyIteration_No_Bins_9/CartPole-v0/RESULTS'
python3 cumulativeReward.py --policy_dir '../logs/ModifiedPolicyIteration_No_Bins_11/CartPole-v0/RESULTS'
python3 cumulativeReward.py --policy_dir '../logs/ModifiedPolicyIteration_No_Bins_13/CartPole-v0/RESULTS'