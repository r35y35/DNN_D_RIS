import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from replay_ppo import SetMemory
from network_ppo import ActorNetwork, CriticNetwork

class Agent:  

    def __init__(self, env, gamma, gae_lambda, batch_size, alpha, clip, epoch,
                 nn_actor, nn_critic):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = clip
        self.n_epoch = epoch
        self.memory = SetMemory(batch_size, env.observation_space, env.action_space)
        self.batch_size = batch_size
        self.n_actions = env.action_space
        self.max_action = 1
        self.min_action = 0

        self.actor = ActorNetwork(env.N, nn_actor, self.n_actions, name='actor')
        self.critic = CriticNetwork(nn_critic,name='critic')
        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=alpha))


    def remember(self, state, action, prob, reward):
        self.memory.store_transition(state, action, prob, reward)

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        
        mu, sigma_old = self.actor(state) #mu & sigma
        # mu = tf.concat((loc_mu,phi_mu),axis=1)
        distrib = tfp.distributions.Normal(mu, scale=sigma_old)
        actions = distrib.sample()
        log_prob = distrib.log_prob(actions)
        actions = actions.numpy()[0]
        actions = np.clip(actions, self.min_action, self.max_action)
        log_prob = log_prob.numpy()[0]

        return actions, log_prob
    
    def learn(self):
        if self.memory.mem_cntr == self.batch_size:
            state_arr, action_arr, old_prob_arr, reward_arr = \
                self.memory.take_memories()
            values_arr = self.critic(state_arr).numpy()
            
            for _ in range(self.n_epoch):   
                #prepare the advantage values
                advantage = np.zeros(self.batch_size, dtype=np.float32)
                for t in range(self.batch_size-1):
                    discount = 1
                    a_t = 0
                    for k in range(t, self.batch_size-1): #advantage is from time t to T-1
                        a_t += discount*(reward_arr[k] + self.gamma*values_arr[k] - values_arr[k])
                        discount *= self.gamma*self.gae_lambda
                    advantage[t] = a_t
                #training
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr,dtype=tf.float32)
                    old_probs = tf.convert_to_tensor(old_prob_arr,dtype=tf.float32) #old mu
                    actions = tf.convert_to_tensor(action_arr,dtype=tf.float32)

                    mu, sigma= self.actor(states) #mu & sigma
                    # mu = tf.concat((loc_mu,phi_mu),axis=1)
                    dist = tfp.distributions.Normal(mu, scale=sigma)
                    new_probs = dist.log_prob(actions)

                    critic_value = self.critic(states)
                    critic_value = tf.squeeze(critic_value, 1)

                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    
                    repmat_adv = tf.tile(tf.reshape(advantage, (-1,1)) , [1,self.n_actions]) #to repeat advantage, so dot operation is possible 
                    weighted_probs = repmat_adv * prob_ratio #elementwise
                    clipped_probs = tf.clip_by_value(prob_ratio,
                                                     1-self.policy_clip,
                                                     1+self.policy_clip)
                    weighted_clipped_probs = clipped_probs * repmat_adv #elementwise
                    actor_loss = -tf.math.minimum(weighted_probs,
                                                  weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage + values_arr
                    critic_loss = tf.keras.losses.MSE(critic_value, returns)

                actor_params = self.actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_params = self.critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                self.actor.optimizer.apply_gradients(
                        zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(
                        zip(critic_grads, critic_params))

'''class Agent:
        
    def __init__(self, env, gamma, gae_lambda, batch_size, alpha, clip, epoch, nn_actor, nn_critic):
        
        self.env   = env
        self.gamma = gamma
        self.gae_lambda  = gae_lambda
        self.policy_clip = clip
        self.n_epoch     = epoch
        self.memory      = SetMemory(batch_size, env.observation_space, env.action_space)
        self.batch_size  = batch_size
        self.n_actions   = env.action_space
        
        self.max_action = 1
        self.min_action = 0

        self.actor  = ActorNetwork(nn_actor, self.n_actions, name='actor')
        self.critic = CriticNetwork(nn_critic,name='critic')
        self.actor.compile(optimizer=Adam(learning_rate=alpha, amsgrad= True))
        self.critic.compile(optimizer=Adam(learning_rate=alpha, amsgrad= True))

    def remember(self, state, action, prob, reward):
        self.memory.store_transition(state, action, prob, reward)

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        
        mu, sigma_old = self.actor(state) #mu & sigma
        # mu = tf.concat((loc_mu,phi_mu),axis=1)
        distrib = tfp.distributions.Normal(mu, scale=sigma_old)
        actions = distrib.sample()
        log_prob = distrib.log_prob(actions)
        actions = actions.numpy()[0]
        actions = np.clip(actions, self.min_action, self.max_action)
        log_prob = log_prob.numpy()[0]

        return actions, log_prob
    
    def learn(self):
        if self.memory.mem_cntr == self.batch_size:
            state_arr, action_arr, old_prob_arr, reward_arr = \
                self.memory.take_memories()
            values_arr = self.critic(state_arr).numpy()
            
            for _ in range(self.n_epoch):   
                #prepare the advantage values
                advantage = np.zeros(self.batch_size, dtype=np.float32)
                for t in range(self.batch_size-1):
                    discount = 1
                    a_t = 0
                    for k in range(t, self.batch_size-1): #advantage is from time t to T-1
                        a_t += discount*(reward_arr[k] + self.gamma*values_arr[k] - values_arr[k])
                        discount *= self.gamma*self.gae_lambda
                    advantage[t] = a_t
                #training
                with tf.GradientTape(persistent=True) as tape:
                    states = tf.convert_to_tensor(state_arr,dtype=tf.float32)
                    old_probs = tf.convert_to_tensor(old_prob_arr,dtype=tf.float32) #old mu
                    actions = tf.convert_to_tensor(action_arr,dtype=tf.float32)

                    mu, sigma= self.actor(states) #mu & sigma
                    # mu = tf.concat((loc_mu,phi_mu),axis=1)
                    dist = tfp.distributions.Normal(mu, scale=sigma)
                    new_probs = dist.log_prob(actions)

                    critic_value = self.critic(states)
                    critic_value = tf.squeeze(critic_value, 1)

                    prob_ratio = tf.math.exp(new_probs - old_probs)
                    
                    repmat_adv = tf.tile(tf.reshape(advantage, (-1,1)) , [1,self.n_actions]) #to repeat advantage, so dot operation is possible 
                    weighted_probs = repmat_adv * prob_ratio #elementwise
                    clipped_probs = tf.clip_by_value(prob_ratio,
                                                     1-self.policy_clip,
                                                     1+self.policy_clip)
                    weighted_clipped_probs = clipped_probs * repmat_adv #elementwise
                    actor_loss = -tf.math.minimum(weighted_probs,
                                                  weighted_clipped_probs)
                    actor_loss = tf.math.reduce_mean(actor_loss)

                    returns = advantage + values_arr
                    critic_loss = tf.keras.losses.MSE(critic_value, returns)

                actor_params = self.actor.trainable_variables
                actor_grads = tape.gradient(actor_loss, actor_params)
                critic_params = self.critic.trainable_variables
                critic_grads = tape.gradient(critic_loss, critic_params)
                self.actor.optimizer.apply_gradients(
                        zip(actor_grads, actor_params))
                self.critic.optimizer.apply_gradients(
                        zip(critic_grads, critic_params))'''



