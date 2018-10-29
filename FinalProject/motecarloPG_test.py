
# coding: utf-8

# # Cartpole: REINFORCE Monte Carlo Policy Gradients

# In this notebook we'll implement an agent <b>that plays Cartpole </b>
# 
# <img src="http://neuro-educator.com/wp-content/uploads/2017/09/DQN.gif" alt="Cartpole gif"/>
# 

# ## This notebook is part of the Free Deep Reinforcement Course 📝
# <img src="https://simoninithomas.github.io/Deep_reinforcement_learning_Course/assets/img/preview.jpg" alt="Deep Reinforcement Course" style="width: 500px;"/>
# 
# <p> Deep Reinforcement Learning Course is a free series of blog posts about Deep Reinforcement Learning, where we'll learn the main algorithms, <b>and how to implement them in Tensorflow.</b></p>
# 
# <p>The goal of these articles is to <b>explain step by step from the big picture</b> and the mathematical details behind it, to the implementation with Tensorflow </p>
# 
# 
# <a href="https://simoninithomas.github.io/Deep_reinforcement_learning_Course/">Syllabus</a><br>
# <a href="https://medium.freecodecamp.org/an-introduction-to-reinforcement-learning-4339519de419">Part 0: Introduction to Reinforcement Learning </a><br>
# <a href="https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe"> Part 1: Q-learning with FrozenLake</a><br>
# <a href="https://medium.freecodecamp.org/an-introduction-to-deep-q-learning-lets-play-doom-54d02d8017d8"> Part 2: Deep Q-learning with Doom</a><br>
# <a href=""> Part 3: Policy Gradients with Doom </a><br>
# 
# ## Checklist 📝
# - To launch tensorboard : `tensorboard --logdir=/tensorboard/pg/1`
# - ⚠️⚠️⚠️ You need to download vizdoom and place the folder in the repos.
# - If don't want to train, you must change **training to False** (in hyperparameters step). 
# 
# 
# ## Any questions 👨‍💻
# <p> If you have any questions, feel free to ask me: </p>
# <p> 📧: <a href="mailto:hello@simoninithomas.com">hello@simoninithomas.com</a>  </p>
# <p> Github: https://github.com/simoninithomas/Deep_reinforcement_learning_Course </p>
# <p> 🌐 : https://simoninithomas.github.io/Deep_reinforcement_learning_Course/ </p>
# <p> Twitter: <a href="https://twitter.com/ThomasSimonini">@ThomasSimonini</a> </p>
# <p> Don't forget to <b> follow me on <a href="https://twitter.com/ThomasSimonini">twitter</a>, <a href="https://github.com/simoninithomas/Deep_reinforcement_learning_Course">github</a> and <a href="https://medium.com/@thomassimonini">Medium</a> to be alerted of the new articles that I publish </b></p>
#     
# 
# ## How to help  🙌
# 3 ways:
# - **Clap our articles a lot**:Clapping in Medium means that you really like our articles. And the more claps we have, the more our article is shared
# - **Share and speak about our articles**: By sharing our articles you help us to spread the word.
# - **Improve our notebooks**: if you found a bug or **a better implementation** you can send a pull request.
# <br>

# ## Step 1: Import the libraries 📚

# In[1]:


import tensorflow as tf
import numpy as np
import gym


# ## Step 2: Create our environment 🎮
# This time we use <a href="https://gym.openai.com/">OpenAI Gym</a> which has a lot of great environments.

# In[2]:


env = gym.make('InvertedDoublePendulum-v2')
env = env.unwrapped
# Policy gradient has high variance, seed for reproducability
env.seed(1)


# ## Step 3: Set up our hyperparameters ⚗️

# In[3]:


## ENVIRONMENT Hyperparameters
state_size = 11
action_size = env.action_space.shape[0]

print ("[INFO] action_size: {}, state_size: {}".format(env.action_space.shape, state_size))
## TRAINING Hyperparameters
max_episodes = 10000
learning_rate = 0.01
gamma = 0.95 # Discount rate


# ## Step 4 : Define the preprocessing functions ⚙️
# This function takes <b>the rewards and perform discounting.</b>

# In[4]:


def discount_and_normalize_rewards(episode_rewards):
    discounted_episode_rewards = np.zeros_like(episode_rewards)
    cumulative = 0.0
    for i in reversed(range(len(episode_rewards))):
        cumulative = cumulative * gamma + episode_rewards[i]
        discounted_episode_rewards[i] = cumulative
    
    mean = np.mean(discounted_episode_rewards)
    std = np.std(discounted_episode_rewards)
    discounted_episode_rewards = (discounted_episode_rewards - mean) / (std)
    
    return discounted_episode_rewards


# ## Step 5: Create our Policy Gradient Neural Network model 🧠

# <img src="https://raw.githubusercontent.com/simoninithomas/Deep_reinforcement_learning_Course/master/Policy%20Gradients/Cartpole/assets/catpole.png">
# 
# The idea is simple:
# - Our state which is an array of 4 values will be used as an input.
# - Our NN is 3 fully connected layers.
# - Our output activation function is softmax that squashes the outputs to a probability distribution (for instance if we have 4, 2, 6 --> softmax --> (0.4, 0.2, 0.6)

# In[5]:


with tf.name_scope("inputs"):
    input_ = tf.placeholder(tf.float32, [None, state_size], name="input_")
    actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
    discounted_episode_rewards_ = tf.placeholder(tf.float32, [None,], name="discounted_episode_rewards")
    
    # Add this placeholder for having this variable in tensorboard
    mean_reward_ = tf.placeholder(tf.float32 , name="mean_reward")

    with tf.name_scope("fc1"):
        fc1 = tf.contrib.layers.fully_connected(inputs = input_,
                                                num_outputs = 10,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("fc2"):
        fc2 = tf.contrib.layers.fully_connected(inputs = fc1,
                                                num_outputs = action_size,
                                                activation_fn= tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())
    
    with tf.name_scope("fc3"):
        fc3 = tf.contrib.layers.fully_connected(inputs = fc2,
                                                num_outputs = action_size,
                                                activation_fn= None,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("softmax"):
        action_distribution = tf.nn.softmax(fc3)

    with tf.name_scope("loss"):
        # tf.nn.softmax_cross_entropy_with_logits computes the cross entropy of the result after applying the softmax function
        # If you have single-class labels, where an object can only belong to one class, you might now consider using 
        # tf.nn.sparse_softmax_cross_entropy_with_logits so that you don't have to convert your labels to a dense one-hot array. 
        neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits = fc3, labels = actions)
        loss = tf.reduce_mean(neg_log_prob * discounted_episode_rewards_) 
        
    
    with tf.name_scope("train"):
        train_opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)


# ## Step 6: Set up Tensorboard 📊
# For more information about tensorboard, please watch this <a href="https://www.youtube.com/embed/eBbEDRsCmv4">excellent 30min tutorial</a> <br><br>
# To launch tensorboard : `tensorboard --logdir=/tensorboard/pg/1`

# In[6]:


# Setup TensorBoard Writer
writer = tf.summary.FileWriter("/tensorboard/pg/1")

## Losses
tf.summary.scalar("Loss", loss)

## Reward mean
tf.summary.scalar("Reward_mean", mean_reward_)

write_op = tf.summary.merge_all()


# ## Step 7: Train our Agent 🏃‍♂️
# Create the NN
"""
maxReward = 0 # Keep track of maximum reward
for episode in range(max_episodes):
    episode + 1
    reset environment
    reset stores (states, actions, rewards)
    
    For each step:
        Choose action a
        Perform action a
        Store s, a, r
        If done:
            Calculate sum reward
            Calculate gamma Gt
            Optimize
"""
# In[7]:


allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
episode = 0
episode_states, episode_actions, episode_rewards = [],[],[]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for episode in range(max_episodes):
        
        episode_rewards_sum = 0

        # Launch the game
        state = env.reset()
        
        env.render()
           
        while True:
            
            # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
            action_probability_distribution = sess.run(action_distribution, feed_dict={input_: state.reshape([1,state_size])})
            
            action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel())  # select action w.r.t the actions prob

            # Perform a
            new_state, reward, done, info = env.step(action)

            # Store s, a, r
            episode_states.append(state)
                        
            # For actions because we output only one (the index) we need 2 (1 is for the action taken)
            # We need [0., 1.] (if we take right) not just the index
            action_ = np.zeros(action_size)
            action_[action] = 1
            
            episode_actions.append(action_)
            
            episode_rewards.append(reward)
            if done:
                # Calculate sum reward
                episode_rewards_sum = np.sum(episode_rewards)
                
                allRewards.append(episode_rewards_sum)
                
                total_rewards = np.sum(allRewards)
                
                # Mean reward
                mean_reward = np.divide(total_rewards, episode+1)
                
                
                maximumRewardRecorded = np.amax(allRewards)
                
                print("==========================================")
                print("Episode: ", episode)
                print("Reward: ", episode_rewards_sum)
                print("Mean Reward", mean_reward)
                print("Max reward so far: ", maximumRewardRecorded)
                
                # Calculate discounted reward
                discounted_episode_rewards = discount_and_normalize_rewards(episode_rewards)
                                
                # Feedforward, gradient and backpropagation
                loss_, _ = sess.run([loss, train_opt], feed_dict={input_: np.vstack(np.array(episode_states)),
                                                                 actions: np.vstack(np.array(episode_actions)),
                                                                 discounted_episode_rewards_: discounted_episode_rewards 
                                                                })
                
 
                                                                 
                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={input_: np.vstack(np.array(episode_states)),
                                                                 actions: np.vstack(np.array(episode_actions)),
                                                                 discounted_episode_rewards_: discounted_episode_rewards,
                                                                    mean_reward_: mean_reward
                                                                })
                
               
                writer.add_summary(summary, episode)
                writer.flush()
                
                # Reset the transition stores
                episode_states, episode_actions, episode_rewards = [],[],[]
                
                break
            
            state = new_state

