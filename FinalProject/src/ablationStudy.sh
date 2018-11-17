# ValueIteration 
# python3 ValueIterationBucketing.py --num_episodes 150 --num_bins 4 --verbose 0
# python3 ValueIterationBucketing.py --num_episodes 150 --num_bins 7 --verbose 0
# python3 ValueIterationBucketing.py --num_episodes 150 --num_bins 11 --verbose 0
# python3 ValueIterationBucketing.py --num_episodes 150 --num_bins 13 --verbose 0

# PolicyIteration
python3 montecarloPG.py --exp_name MCPG150 --num_episodes 150 --verbose 0
python3 montecarloPG.py --exp_name MCPG500 --num_episodes 500 --verbose 0
python3 montecarloPG.py --exp_name MCPG1000 --num_episodes 1000 --verbose 0
python3 montecarloPG.py --exp_name MCPG1500 --num_episodes 1500 --verbose 0
