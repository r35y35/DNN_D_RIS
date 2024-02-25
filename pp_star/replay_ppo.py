# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 13:56:56 2023

@author: WITS
"""

import numpy as np

class SetMemory:
    def __init__(self, batch_size, input_shape, n_actions): #mem_size%batch_size must == 0
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((batch_size, input_shape))
        self.action_memory = np.zeros((batch_size, n_actions))
        self.probs_memory = np.zeros((batch_size, n_actions))
        self.reward_memory = np.zeros(batch_size)
        self.terminal_memory = np.zeros(batch_size, dtype=np.bool)

    def store_transition(self, state, action, prob, reward):
        index = self.mem_cntr % self.batch_size
        if index==0:
            self.mem_cntr = 0

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.probs_memory[index] = prob
        self.reward_memory[index] = reward


        self.mem_cntr += 1

    def take_memories(self):
        states = self.state_memory
        actions = self.action_memory
        probs = self.probs_memory
        rewards = self.reward_memory

        return states, actions, probs, rewards