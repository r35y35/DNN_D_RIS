"""
Created on Tue Dec 26 13:13:44 2022

@author: WITS
"""

import tensorflow as tf      
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class CriticNetwork(keras.Model):                 ###    policy AEvaluatoion  (S<A) as input and QQ  as output
    def __init__(self, net, name='critic'):# each class can have its own session ????? but its more tidy to def session for each class 
        super(CriticNetwork, self).__init__()
        self.net = net
        self.fc=[]
        for i in range (len(self.net)):
            self.fc.append(Dense(self.net[i], activation='relu'))#########################
        self.q = Dense(1, activation=None) #relu......node output = activation(weighted sum of inputs)

    def call (self, state, action):       
        action_value = self.fc[0](tf.concat([state, action], axis=1))
        for i in range (1, len(self.net)):
            action_value = self.fc[i](action_value)
        q = self.q(action_value)
        
        return q

class ActorNetwork(keras.Model):  ## policy update   S as INPUT  and A as output   ann   se aata h ye sb

    def __init__(self, net, n_actions, name='actor'):    # added n 8 feb######################################
        super(ActorNetwork,self).__init__()
        self.net = net
        self.fc=[]
        for i in range (len(self.net)):
            self.fc.append(Dense(self.net[i], activation='relu', name='Hidden-'+str(i)))                  ##########################  DNN
        self.n_actions = n_actions        
        self.mu = Dense(n_actions, activation='sigmoid')                               ##     n_actions k aage se selgf maione htaya 8 feb ko   action_dim
        
    def call (self, state):
        prob = self.fc[0](state)
        for i in range (1, len(self.net)):
            prob = self.fc[i](prob)        
        mu = self.mu(prob)  
        return mu
    
    
    
    
    
    
    
    
    # The actor takes a decision based on a policy, critic evaluates state-action pair 
    # and give it a Q value. If the state-action pair is good according to critics it will have a higher Q value
    # and vice versa.
    
'''actor....Weight initialization is not necessary but generally, if we give it some init it learns faster.
Choosing an optimizer is very very important, the different optimizers can make lots of differences.
Now, how to choose the last activation function really depends on what kind of action space you are using, for example,
 if it is small and all values are like [-1,-2,-3] to [1,2,3] you can go ahead and tanh (squashing function),
 if you have [-2,-4000,-230] to [2,6000,560] you might want to change the activation function.'''