
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 13:13:44 2022

@author: WITS
"""

import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size   ##  memory cant be bounded
        self.mem_cntr =0
        self.state_memory = np.zeros(( self.mem_size, input_shape ))
        self.new_state_memory =np. zeros((self.mem_size, input_shape))
        self.action_memory = np.zeros ((self.mem_size, n_actions ))
        self. reward_memory = np. zeros (self.mem_size)
       # self.terminal_memory = np.zeros(self.men_size, )

        
    def store_transition(self, state, action, reward, new_state):    ### done ===terminal flag 
        index = self.mem_cntr % self.mem_size                                      #####  once we have index ....then we can store tramsition
        self.state_memory [index] = state
        self.new_state_memory [index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        # self.terminal_memory[index]= 1- int(done)
        self.mem_cntr +=1    ###  incriment memory counter  by   1  
        #self.done= done
    #### value of terminal state is zero...reset thre episode ====back to the initial state 
              ###  incriment memory counter  by   1  
      
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random. choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        
        return states, actions, rewards, states_
    
    
    
    
    # MDP (Markov Decision Process) requires that the agent takes the best action based on the current state. 
    # This gives step reward and a new observation state. This problem is called MDP. We store these values
    # in a buffer called replay buffer.  
#What a reply buffer does is what makes this algorithm off policy.         
              