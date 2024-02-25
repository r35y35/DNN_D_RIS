# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 13:56:16 2023

@author: WITS
"""

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
    
    def call (self, state):       
        action_value = self.fc[0](state)
        for i in range (1, len(self.net)):
            action_value = self.fc[i](action_value)
        q = self.q(action_value)
        return q

class ActorNetwork(keras.Model):
    def __init__(self, n, net, n_actions, name='actor'):
        super(ActorNetwork, self).__init__()

        self.net = net
        self.fc=[]
        for i in range (len(self.net)):
            self.fc.append(Dense(self.net[i], activation='relu', name='Hidden-'+str(i)))
        
        self.mu = Dense((n_actions), activation='sigmoid', name='Output theta & phi')
        self.deviation = Dense((n_actions), activation='softplus') #sigmoid 

    def call(self, state):
        prob = self.fc[0](state)
        for i in range (1, len(self.net)):
            prob = self.fc[i](prob)
        mu = self.mu(prob)
        deviation = self.deviation(prob)
        
        return mu, deviation
    




















# import tensorflow.keras as keras
# from tensorflow.keras.layers import Dense

# class CriticNetwork(keras.Model):
#     def __init__(self, net, name='critic'):
#         super(CriticNetwork, self).__init__()
#         self.net = net
#         self.fc=[]
#         for i in range (len(self.net)):
#             self.fc.append(Dense(self.net[i], activation='relu'))
#         self.q = Dense(1, activation=None)
    
#     def call (self, state):       
#         action_value = self.fc[0](state)
#         for i in range (1, len(self.net)):
#             action_value = self.fc[i](action_value)
#         q = self.q(action_value)
#         return q

# class ActorNetwork(keras.Model):
#     def __init__(self, n, net, n_actions, setting = None,name='actor'):
#         super(ActorNetwork, self).__init__()
#         self.setting = setting
#         self.net = net
#         self.fc=[]
#         for i in range (len(self.net)):
#             self.fc.append(Dense(self.net[i], activation='relu', name='Hidden-'+str(i)))
#         self.loc_mu = Dense((2*n), activation='tanh',name = 'Output X and Y axist of UAV')
#         self.phi_mu = Dense((n_actions-(2*n)), activation='sigmoid', name='Output theta & phi') #outputnya tidak boleh negatif
#         self.mu = Dense((n_actions), activation='sigmoid', name='Output theta & phi')
#         self.deviation = Dense((n_actions), activation='softplus') #sigmoid 

#     def call(self, state):
        
#         if self.setting is None:
#             prob = self.fc[0](state)
#             for i in range (1, len(self.net)):
#                 prob = self.fc[i](prob)
#             # loc_mu = self.loc_mu(prob)    
#             # phi_mu = self.phi_mu(prob)
#             mu = self.mu(prob)
#             deviation = self.deviation(prob)
            
#             return mu, deviation
#         else:
#             prob = self.fc[0](state)
#             for i in range (1, len(self.net)):
#                 prob = self.fc[i](prob)
#             loc_mu = self.loc_mu(prob)    
#             phi_mu = self.phi_mu(prob)
#             deviation = self.deviation(prob)
            
#             return loc_mu, phi_mu, deviation
        