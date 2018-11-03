import tensorflow as tf
import numpy as np
import gym
import os
from gym import wrappers



OUTPUT_RESULTS_DIR = "../logs"
ENVIRONMENT = 'InvertedDoublePendulum-v2'

# TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
TIMESTAMP = 'RESULTS'
SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "MonteCarloPG", ENVIRONMENT, TIMESTAMP)
env = gym.make(ENVIRONMENT)
env = wrappers.Monitor(env, os.path.join(SUMMARY_DIR, ENVIRONMENT), force=True, video_callable=None)



# Policy gradient has high variance, seed for reproducability
env.seed(1)



## ENVIRONMENT Hyperparameters
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

print ("[INFO] action_size: {}, state_size: {}".format(env.action_space.shape, state_size))




## TRAINING Hyperparameters
max_episodes = 10000
learning_rate = 0.01
gamma = 0.95 # Discount rate



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



def save_model(model_path, saver, sess, step=None):
    save_path = saver.save(sess, os.path.join(model_path, "model.ckpt"), global_step=step)
    return save_path

def restore_model(model_path, saver, sess):
    saver.restore(sess, os.path.join(model_path, "model.ckpt"))
    print("Model restored from", model_path)


with tf.name_scope("inputs"):
    input_ = tf.placeholder(tf.float32, [None, state_size], name="input_")
    actions = tf.placeholder(tf.int32, [None, action_size], name="actions")
    discounted_episode_rewards_ = tf.placeholder(tf.float32, [None,], name="discounted_episode_rewards")
    
    # Add this placeholder for having this variable in tensorboard
    mean_reward_ = tf.placeholder(tf.float32 , name="mean_reward")

    with tf.name_scope("fc1"):
        fc1 = tf.contrib.layers.fully_connected(inputs = input_,
                                                num_outputs = 400,
                                                activation_fn=tf.nn.relu,
                                                weights_initializer=tf.contrib.layers.xavier_initializer())

    with tf.name_scope("fc2"):
        fc2 = tf.contrib.layers.fully_connected(inputs = fc1,
                                                num_outputs = 400,
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






# Setup TensorBoard Writer
writer = tf.summary.FileWriter(SUMMARY_DIR)

## Losses
tf.summary.scalar("Loss", loss)

## Reward mean
tf.summary.scalar("Reward_mean", mean_reward_)

write_op = tf.summary.merge_all()



allRewards = []
total_rewards = 0
maximumRewardRecorded = 0
episode = 0
episode_states, episode_actions, episode_rewards = [],[],[]





saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for episode in range(max_episodes):
        
        episode_rewards_sum = 0

        # Launch the game
        state = env.reset()
        
        # env.render()
           
        while True:
            
            # Choose action a, remember WE'RE NOT IN A DETERMINISTIC ENVIRONMENT, WE'RE OUTPUT PROBABILITIES.
            action_probability_distribution = sess.run(action_distribution, feed_dict={input_: state.reshape([1,state_size])})
            action = np.random.choice(range(action_probability_distribution.shape[1]), p=action_probability_distribution.ravel())

            # Perform a
            new_state, reward, done, info = env.step(action)

            # Store s, a, r
            episode_states.append(state)
            action_ = np.zeros(action_size)
            action_[action] = 1
            
            episode_actions.append(action_)
            
            episode_rewards.append(reward)
            if done:
                episode_rewards_sum = np.sum(episode_rewards)
                allRewards.append(episode_rewards_sum)
                total_rewards = np.sum(allRewards)
                mean_reward = np.divide(total_rewards, episode+1)
                maximumRewardRecorded = np.amax(allRewards)
                


                print("==========================================")
                print("Episode: ", episode)
                print("Reward: ", episode_rewards_sum)
                print("Mean Reward", mean_reward)
                print("Max reward so far: ", maximumRewardRecorded)
                
                # Calculate discounted reward
                discounted_episode_rewards = discount_and_normalize_rewards(episode_rewards)
                # discounted_episode_rewards = episode_rewards
                                




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
                
                if episode % 100 == 0 and episode > 0:
                    path = save_model(SUMMARY_DIR, saver, sess)
                    print('Saved model at episode', episode, 'in', path)
                break
            
            state = new_state