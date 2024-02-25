# -*- coding: utf-8 -*-
"""
Created on Thu May  4 18:41:41 2023

@author: WITS

"""

import numpy as np
K=4
phi_array = np.zeros(K,dtype=np.complex)
for i in range(K):
    phi_array[i] = np.exp(-1j*2*np.pi*i*0.5*np.cos(np.random.rand(1)*2*np.pi))
