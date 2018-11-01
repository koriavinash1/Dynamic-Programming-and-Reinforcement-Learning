import tensorflow as tf
import numpy as np
import gym
import os
import scipy.signal
from gym import wrappers
from datetime import datetime
from time import time
from utils import RunningStats, discount, add_histogram
OUTPUT_RESULTS_DIR = "../logs"

EP_MAX = 10000
GAMMA = 0.99
LAMBDA = 0.95
ENTROPY_BETA = 0.01  # 0.01 for discrete, 0.0 for continuous
LR = 0.0001
BATCH = 8192  # 128 for discrete, 8192 for continuous
MINIBATCH = 32
EPOCHS = 10
EPSILON = 0.1
VF_COEFF = 1.0
L2_REG = 0.001
SIGMA_FLOOR = 0.0

# MODEL_RESTORE_PATH = "/path/to/saved/model"
MODEL_RESTORE_PATH = None


class PPO(object):
    def __init__(self, environment, summary_dir="./"):
        if len(environment.action_space.shape) > 0:
            self.discrete = False
            self.s_dim, self.a_dim = environment.observation_space.shape, environment.action_space.shape[0]
            self.a_bound = (environment.action_space.high - environment.action_space.low) / 2
            self.actions = tf.placeholder(tf.float32, [None, self.a_dim], 'action')
        else:
            self.discrete = True
            self.s_dim, self.a_dim = environment.observation_space.shape, environment.action_space.n
            self.actions = tf.placeholder(tf.int32, [None, 1], 'action')

        self.sess = tf.Session()
        self.state = tf.placeholder(tf.float32, [None] + list(self.s_dim), 'state')
        self.advantage = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.rewards = tf.placeholder(tf.float32, [None, 1], 'discounted_r')




        self.dataset = tf.data.Dataset.from_tensor_slices({"state": self.state, "actions": self.actions,
                                                           "rewards": self.rewards, "advantage": self.advantage})
        self.dataset = self.dataset.shuffle(buffer_size=10000)
        self.dataset = self.dataset.batch(MINIBATCH)
        self.dataset = self.dataset.cache()
        self.dataset = self.dataset.repeat(EPOCHS)
        self.iterator = self.dataset.make_initializable_iterator()
        batch = self.iterator.get_next()




        pi_old, pi_old_params = self._build_anet(batch["state"], 'oldpi')
        pi, pi_params = self._build_anet(batch["state"], 'pi')
        pi_eval, _ = self._build_anet(self.state, 'pi', reuse=True)

        
        vf_old, vf_old_params = self._build_cnet(batch["state"], "oldvf")
        self.v, vf_params = self._build_cnet(batch["state"], "vf")
        self.vf_eval, _ = self._build_cnet(self.state, 'vf', reuse=True)

        

        self.sample_op = tf.squeeze(pi_eval.sample(1), axis=0, name="sample_action")
        self.eval_action = pi_eval.mode()  # Used mode for discrete case. Mode should equal mean in continuous
        self.global_step = tf.train.get_or_create_global_step()
        self.saver = tf.train.Saver()



        with tf.variable_scope('loss'):
            epsilon_decay = tf.train.polynomial_decay(EPSILON, self.global_step, 1e5, 0.01, power=0.0)

            with tf.variable_scope('policy'):
                # Use floor functions for the probabilities to prevent NaNs when prob = 0
                ratio = tf.maximum(pi.prob(batch["actions"]), 1e-6) / tf.maximum(pi_old.prob(batch["actions"]), 1e-6)
                ratio = tf.clip_by_value(ratio, 0, 10)
                surr1 = batch["advantage"] * ratio
                surr2 = batch["advantage"] * tf.clip_by_value(ratio, 1 - epsilon_decay, 1 + epsilon_decay)
                loss_pi = -tf.reduce_mean(tf.minimum(surr1, surr2))
                tf.summary.scalar("loss", loss_pi)

            with tf.variable_scope('value_function'):
                # Sometimes values clipping helps, sometimes just using raw residuals is better 
                clipped_value_estimate = vf_old + tf.clip_by_value(self.v - vf_old, -epsilon_decay, epsilon_decay)
                loss_vf1 = tf.squared_difference(clipped_value_estimate, batch["rewards"])
                loss_vf2 = tf.squared_difference(self.v, batch["rewards"])
                loss_vf = tf.reduce_mean(tf.maximum(loss_vf1, loss_vf2)) * 0.5
                # loss_vf = tf.reduce_mean(tf.square(self.v - batch["rewards"])) * 0.5
                tf.summary.scalar("loss", loss_vf)


            with tf.variable_scope('entropy'):
                entropy = pi.entropy()
                pol_entpen = -ENTROPY_BETA * tf.reduce_mean(entropy)

            loss = loss_pi + loss_vf * VF_COEFF + pol_entpen
            tf.summary.scalar("total", loss)



        with tf.variable_scope('train'):
            opt = tf.train.AdamOptimizer(LR)
            self.train_op = opt.minimize(loss, global_step=self.global_step, var_list=pi_params + vf_params)


        with tf.variable_scope('update_old'):
            self.update_pi_old_op = [oldp.assign(p) for p, oldp in zip(pi_params, pi_old_params)]
            self.update_vf_old_op = [oldp.assign(p) for p, oldp in zip(vf_params, vf_old_params)]



        self.writer = tf.summary.FileWriter(summary_dir, self.sess.graph)
        self.sess.run(tf.global_variables_initializer())


        tf.summary.scalar("value", tf.reduce_mean(self.v))
        tf.summary.scalar("policy_entropy", tf.reduce_mean(entropy))

        if not self.discrete:
            tf.summary.scalar("sigma", tf.reduce_mean(pi.stddev()))

        self.summarise = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))



    def save_model(self, model_path, step=None):
        save_path = self.saver.save(self.sess, os.path.join(model_path, "model.ckpt"), global_step=step)
        return save_path

    def restore_model(self, model_path):
        self.saver.restore(self.sess, os.path.join(model_path, "model.ckpt"))
        print("Model restored from", model_path)



    def update(self, s, a, r, adv):
        start = time()
        e_time = []

        self.sess.run([self.update_pi_old_op, self.update_vf_old_op, self.iterator.initializer],
                      feed_dict={self.state: s, self.actions: a, self.rewards: r, self.advantage: adv})

        while True:
            try:
                e_start = time()
                summary, step, _ = self.sess.run([self.summarise, self.global_step, self.train_op])
                e_time.append(time() - e_start)
            except tf.errors.OutOfRangeError:
                break
        print("Trained in %.3fs. Average %.3fs/batch. Global step %i" % (time() - start, np.mean(e_time), step))
        return summary



    def _build_anet(self, state_in, name, reuse=False):
        w_reg = tf.contrib.layers.l2_regularizer(L2_REG)

        with tf.variable_scope(name, reuse=reuse):
            layer_1 = tf.layers.dense(state_in, 400, tf.nn.relu, kernel_regularizer=w_reg, name="pi_l1")
            layer_2 = tf.layers.dense(layer_1, 400, tf.nn.relu, kernel_regularizer=w_reg, name="pi_l2")

            if self.discrete:
                a_logits = tf.layers.dense(layer_2, self.a_dim, kernel_regularizer=w_reg, name="pi_logits")
                dist = tf.distributions.Categorical(logits=a_logits)
            else:
                mu = tf.layers.dense(layer_2, self.a_dim, tf.nn.tanh, kernel_regularizer=w_reg, name="pi_mu")
                log_sigma = tf.get_variable(name="pi_sigma", shape=self.a_dim, initializer=tf.zeros_initializer())
                dist = tf.distributions.Normal(loc=mu * self.a_bound, scale=tf.maximum(tf.exp(log_sigma), SIGMA_FLOOR))
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return dist, params



    def _build_cnet(self, state_in, name, reuse=False):
        w_reg = tf.contrib.layers.l2_regularizer(L2_REG)

        with tf.variable_scope(name, reuse=reuse):
            l1 = tf.layers.dense(state_in, 400, tf.nn.relu, kernel_regularizer=w_reg, name="vf_l1")
            l2 = tf.layers.dense(l1, 400, tf.nn.relu, kernel_regularizer=w_reg, name="vf_l2")
            vf = tf.layers.dense(l2, 1, kernel_regularizer=w_reg, name="vf_output")

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return vf, params


    def evaluate_state(self, state, stochastic=True):
        if stochastic:
            action, value = self.sess.run([self.sample_op, self.vf_eval], {self.state: state[np.newaxis, :]})
        else:
            action, value = self.sess.run([self.eval_action, self.vf_eval], {self.state: state[np.newaxis, :]})
        return action[0], np.squeeze(value)





if __name__ == '__main__':
    # Discrete environments
    # ENVIRONMENT = 'CartPole-v1'

    # Continuous environments
    ENVIRONMENT = 'InvertedPendulum-v2'

    # TIMESTAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
    TIMESTAMP = 'RESULTS'
    SUMMARY_DIR = os.path.join(OUTPUT_RESULTS_DIR, "PPO", ENVIRONMENT, TIMESTAMP)

    env = gym.make(ENVIRONMENT)
    env = wrappers.Monitor(env, os.path.join(SUMMARY_DIR, ENVIRONMENT), video_callable=None)
    ppo = PPO(env, SUMMARY_DIR)

    if MODEL_RESTORE_PATH is not None:
        ppo.restore_model(MODEL_RESTORE_PATH)

    t, terminal = 0, False
    buffer_s, buffer_a, buffer_r, buffer_v, buffer_terminal = [], [], [], [], []
    rolling_r = RunningStats()

    for episode in range(EP_MAX + 1):

        s = env.reset()
        ep_r, ep_t, ep_a = 0, 0, []

        while True:
            a, v = ppo.evaluate_state(s)

            # Update ppo
            if t == BATCH:  # or (terminal and t < BATCH):
                # Normalise rewards
                rewards = np.array(buffer_r)
                rolling_r.update(rewards)
                rewards = np.clip(rewards / rolling_r.std, -10, 10)

                v_final = [v * (1 - terminal)]  # v = 0 if terminal, otherwise use the predicted v
                values = np.array(buffer_v + v_final)
                terminals = np.array(buffer_terminal + [terminal])

                # Generalized Advantage Estimation - https://arxiv.org/abs/1506.02438
                delta = rewards + GAMMA * values[1:] * (1 - terminals[1:]) - values[:-1]
                advantage = discount(delta, GAMMA * LAMBDA, terminals)
                returns = advantage + np.array(buffer_v)
                advantage = (advantage - advantage.mean()) / np.maximum(advantage.std(), 1e-6)

                bs, ba, br, badv = np.reshape(buffer_s, (t,) + ppo.s_dim), np.vstack(buffer_a), \
                                   np.vstack(returns), np.vstack(advantage)

                graph_summary = ppo.update(bs, ba, br, badv)
                buffer_s, buffer_a, buffer_r, buffer_v, buffer_terminal = [], [], [], [], []
                t = 0

            buffer_s.append(s)
            buffer_a.append(a)
            buffer_v.append(v)
            buffer_terminal.append(terminal)
            ep_a.append(a)

            if not ppo.discrete:
                a = np.clip(a, env.action_space.low, env.action_space.high)


            s, r, terminal, _ = env.step(a)
            buffer_r.append(r)

            ep_r += r
            ep_t += 1
            t += 1

            if terminal:
                # print('Episode: %i' % episode, "| Reward: %.2f" % ep_r, '| Steps: %i' % ep_t)

                # End of episode summary
                worker_summary = tf.Summary()
                worker_summary.value.add(tag="Reward", simple_value=ep_r)

                # Create Action histograms for each dimension
                actions = np.array(ep_a)
                if ppo.discrete:
                    add_histogram(ppo.writer, "Action", actions, episode, bins=ppo.a_dim)
                else:
                    for a in range(ppo.a_dim):
                        add_histogram(ppo.writer, "Action/Dim" + str(a), actions[:, a], episode)

                try:
                    ppo.writer.add_summary(graph_summary, episode)
                except NameError:
                    pass

                ppo.writer.add_summary(worker_summary, episode)
                ppo.writer.flush()


                # Save the model
                if episode % 100 == 0 and episode > 0:
                    path = ppo.save_model(SUMMARY_DIR)
                    print('Saved model at episode', episode, 'in', path)

                break

    env.close()

    # Run trained policy
    env = gym.make(ENVIRONMENT)
    env = wrappers.Monitor(env, os.path.join(SUMMARY_DIR, ENVIRONMENT + "_trained"), video_callable=None)
    while True:
        s = env.reset()
        ep_r, ep_t = 0, 0
        while True:
            #env.render()
            a, v = ppo.evaluate_state(s, stochastic=False)
            if not ppo.discrete:
                a = np.clip(a, env.action_space.low, env.action_space.high)
            s, r, terminal, _ = env.step(a)
            ep_r += r
            ep_t += 1
            if terminal:
                print("Reward: %.2f" % ep_r, '| Steps: %i' % ep_t)
                break