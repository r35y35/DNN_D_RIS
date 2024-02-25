# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 17:37:33 2023

@author: WITS
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from network_td3 import ActorNetwork, CriticNetwork
from replay_td3 import ReplayBuffer

class TD3_agent:
    def __init__(self, nn_actor, nn_critic, env,alpha, beta, gamma, buffer_size,tau,batch_size, 
                 update_actor_interval, noise=0.01):
        self.env   = env
        self.gamma = gamma
        self.tau   = tau
                
        self.n_actions = env.action_space
        self.memory = ReplayBuffer(buffer_size, env.observation_space, env.action_space)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        
        self.max_action = 1
        self.min_action = 0
        
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(nn_actor, n=env.N, n_actions=self.n_actions, name='actor')

        self.critic_1 = CriticNetwork(nn_critic,name='critic_1')
        self.critic_2 = CriticNetwork(nn_critic,name='critic_2')

        self.target_actor = ActorNetwork(nn_actor,n=env.N,n_actions=self.n_actions,
                                         name='target_actor')
        self.target_critic_1 = CriticNetwork(nn_critic,name='target_critic_1')
        self.target_critic_2 = CriticNetwork(nn_critic,name='target_critic_2')

        self.actor.compile(optimizer=Adam(learning_rate=alpha), loss='mean')
        self.critic_1.compile(optimizer=Adam(learning_rate=beta),
                              loss='mean_squared_error')
        self.critic_2.compile(optimizer=Adam(learning_rate=beta),
                              loss='mean_squared_error')

        self.target_actor.compile(optimizer=Adam(learning_rate=alpha),
                                  loss='mean')
        self.target_critic_1.compile(optimizer=Adam(learning_rate=beta),
                                     loss='mean_squared_error')
        self.target_critic_2.compile(optimizer=Adam(learning_rate=beta),
                                     loss='mean_squared_error')

        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        # returns a batch size of 1, want a scalar array
        phi_mu = self.actor(state)
        mu = tf.concat((phi_mu),axis=1)
        mu_prime = mu + np.random.normal(scale=self.noise)
        mu_prime = tf.clip_by_value(mu_prime, self.min_action, self.max_action)

        return mu_prime.numpy()[0]

    def remember(self, state, action, reward, new_state):
        self.memory.store_transition(state, action, reward, new_state)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, new_states= \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_states, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            phi_mu = self.target_actor(states_)
            target_actions = tf.concat((phi_mu),axis=1)
            # target_actions = target_actions + \
            #     tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5) scale nya kelebaran
            
            target_actions = target_actions + \
                tf.clip_by_value(np.random.normal(scale=self.noise), -0.5, 0.5)

            target_actions = tf.clip_by_value(target_actions, self.min_action,
                                              self.max_action)

            q1_ = self.target_critic_1(states_, target_actions)
            q2_ = self.target_critic_2(states_, target_actions)

            q1 = tf.squeeze(self.critic_1(states, actions), 1)
            q2 = tf.squeeze(self.critic_2(states, actions), 1)

            # shape is [batch_size, 1], want to collapse to [batch_size]
            q1_ = tf.squeeze(q1_, 1)
            q2_ = tf.squeeze(q2_, 1)

            critic_value_ = tf.math.minimum(q1_, q2_)
            # in tf2 only integer scalar arrays can be used as indices
            # and eager exection doesn't support assignment, so we can't do
            # q1_[dones] = 0.0
            target = rewards + self.gamma*critic_value_
            critic_1_loss = keras.losses.MSE(target, q1)
            critic_2_loss = keras.losses.MSE(target, q2)

        critic_1_gradient = tape.gradient(critic_1_loss,
                                          self.critic_1.trainable_variables)
        critic_2_gradient = tape.gradient(critic_2_loss,
                                          self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(
                   zip(critic_1_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(
                   zip(critic_2_gradient, self.critic_2.trainable_variables))

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        with tf.GradientTape() as tape:
            phi_mu = self.actor(states)
            new_actions = tf.concat((phi_mu),axis=1)
            critic_1_value = self.critic_1(states, new_actions)
            actor_loss = -tf.math.reduce_mean(critic_1_value)

        actor_gradient = tape.gradient(actor_loss,
                                       self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
                        zip(actor_gradient, self.actor.trainable_variables))

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_critic_1.set_weights(weights)

        weights = []
        targets = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_critic_2.set_weights(weights)