# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 17:38:38 2023

@author: WITS
"""

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class CriticNetwork(keras.Model):
    def __init__(self, net, name='critic'):
        super(CriticNetwork, self).__init__()
        self.net = net
        self.fc=[]
        for i in range (len(self.net)):
            self.fc.append(Dense(self.net[i], activation='relu'))
        self.q = Dense(1, activation=None)

    def call (self, state, action):       
        action_value = self.fc[0](tf.concat([state, action], axis=1))
        for i in range (1, len(self.net)):
            action_value = self.fc[i](action_value)
        q = self.q(action_value)
        return q

class ActorNetwork(keras.Model):
    def __init__(self, net, n, n_actions, name='actor'):
        super(ActorNetwork,self).__init__()
        self.net = net
        self.fc=[]
        for i in range (len(self.net)):
            self.fc.append(Dense(self.net[i], activation='relu', name='Hidden-'+str(i)))
            
        self.phi_mu = Dense(n_actions, activation='sigmoid', name='Output theta & phi')
        
    def call (self, state):

        prob = self.fc[0](state)
        for i in range (1, len(self.net)):
            prob = self.fc[i](prob)    
        phi_mu = self.phi_mu(prob)
        
        return phi_mu