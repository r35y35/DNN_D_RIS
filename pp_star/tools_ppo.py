# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:56:42 2023

@author: WITS
"""

import numpy as np
import matplotlib.pyplot as plt
import os

class fungsi:
    
    def db2lin(db):
        lin = np.power(10,(db/10))
        return lin

class plotter_convergence:
    def __init__(self, steps, sample,num_fig):
        self.steps = steps
        self.num_fig = len(num_fig)
        self.data_memory = np.zeros([np.int(steps),np.int(sample),np.int(self.num_fig)])
        self.axis_memory = np.zeros(np.int(steps))
        self.num_sample = sample
        
        self.label = num_fig
        self.index = 0
        
    def record(self, rate, steps_th,sample_th,fig_th):
        self.data_memory[steps_th,sample_th,fig_th] = rate
        self.axis_memory[steps_th] = steps_th
                
    
    def plot(self, plotting = None,title="PPO Single-Agent", ax="Episodes", ay="Received SNR (dB)", grid=1, smoother=1, fig=1):
        if plotting is None:
            plotting = np.mean(self.data_memory,axis=1)
        plot_interval = np.int(self.steps/grid)
        rate_set = np.zeros(grid)
        axis_set = np.zeros((grid),dtype=np.int)
        if smoother < 0.1:
            smoother = 0.1
        smoother = np.log10(10*smoother) #To make logaritmic scale for ease to see as linear scale
        while plot_interval-(1-smoother)*plot_interval<1:
            smoother+=0.01
        
        plt.figure(fig)
        for i0 in range(self.num_fig):
                    
            for i in range(grid):
                rate_set[i]=np.sum(plotting[np.int((i+(1-smoother))*plot_interval):(i+1)*plot_interval,i0])/plot_interval/smoother
                axis_set[i]=self.axis_memory[i*plot_interval]
            
            plt.plot(axis_set, rate_set, linewidth=1, label="M = "+str(self.label[i0]))
            plt.legend(loc='lower right', fontsize=10)
            plt.xlim(np.min(axis_set)-(np.min(axis_set)*0.1), np.max(axis_set)+5)
            plt.ylim(np.min(plotting)-(np.max(plotting)-np.min(plotting))*0.1,
                      np.max(plotting)+(np.max(plotting)-np.min(plotting))*0.1)
            plt.autoscale(enable=False, axis='x')
            
    
            plt.xlabel(ax)
            plt.ylabel(ay)
            plt.minorticks_on()
            plt.grid(b=True, which='major')
            plt.grid(b=True, which='minor',alpha=0.4)
            plt.suptitle(title, fontsize='x-large', fontweight='bold')
        plt.show() 
        
    def average(self):
        plotting = np.mean(self.data_memory,axis=1)[-10:,:]
        plotting = np.mean(plotting,axis=0)
        
        return plotting
        
    def plot_result(self, title ="(PPO) Received SNR Vs BS-USer Horizontal Distance",
                    ax="BS-USer Horizontal Distance (m)", ay="Received SNR (dB)"):
        
        
        plotting = np.mean(self.data_memory,axis=1)[-10:,:]
        plotting = np.mean(plotting,axis=0)
        axis_set = self.label
        
        plt.figure()
        plt.plot(axis_set, plotting, linewidth=1,marker="o")
        plt.xlim(np.min(axis_set)-(np.min(axis_set)*0.1),
                  np.max(axis_set)+(np.min(axis_set)*0.1))
        plt.ylim(np.min(plotting)-(np.max(plotting)-np.min(plotting))*0.1,
                  np.max(plotting)+(np.max(plotting)-np.min(plotting))*0.1)
        plt.autoscale(enable=False, axis='x')
                
        plt.xlabel(ax)
        plt.ylabel(ay)
        plt.grid(which='major', axis='both')
        plt.suptitle(title, fontsize='large', fontweight='bold')     
        
        plt.show()
    
    def save_value(self,file_name):
        
        dirName = 'Value'
        fileName = file_name
        file = dirName+'/'+fileName+'_sample_'+str(self.num_sample)+'_episode_'+str(self.steps)+'.npy'
        
        
        if not os.path.exists(dirName):
            os.mkdir(dirName)
        value = np.mean(self.data_memory,axis=1)
        np.save(file,value)









'''import numpy as np
import matplotlib.pyplot as plt
import os

class plotter_convergence:
    def __init__(self, steps, sample,num_fig):
        self.steps = steps
        self.num_fig = len(num_fig)
        self.data_memory = np.zeros([np.int(steps),np.int(sample),np.int(self.num_fig)])
        self.axis_memory = np.zeros(np.int(steps))
        self.num_sample = sample
        
        self.label = num_fig
        self.index = 0
        
    def record(self, rate, steps_th,sample_th,fig_th):
        self.data_memory[steps_th,sample_th,fig_th] = rate
        self.axis_memory[steps_th] = steps_th
                
    
    def plot(self, plotting = None,title="ppo", ax="Episodes", ay="EE", legend = "K =" , grid=1, smoother=1, fig=1):
        if plotting is None:
            plotting = np.mean(self.data_memory,axis=1)
        plot_interval = np.int(self.steps/grid)
        rate_set = np.zeros(grid)
        axis_set = np.zeros((grid),dtype=np.int)
        if smoother < 0.1:
            smoother = 0.1
        smoother = np.log10(10*smoother) #To make logaritmic scale for ease to see as linear scale
        while plot_interval-(1-smoother)*plot_interval<1:
            smoother+=0.01
        
        plt.figure(fig)
        for i0 in range(self.num_fig):
                    
            for i in range(grid):
                rate_set[i]=np.sum(plotting[np.int((i+(1-smoother))*plot_interval):(i+1)*plot_interval,i0])/plot_interval/smoother
                axis_set[i]=self.axis_memory[i*plot_interval]
            
            plt.plot(axis_set, rate_set, linewidth=1, label="D = "+str(self.label[i0]))
            plt.legend(loc='lower right', fontsize=10)
            plt.xlim(np.min(axis_set)-(np.min(axis_set)*0.1), np.max(axis_set)+5)
            plt.ylim(np.min(plotting)-(np.max(plotting)-np.min(plotting))*0.1,
                      np.max(plotting)+(np.max(plotting)-np.min(plotting))*0.1)
            plt.autoscale(enable=False, axis='x')
            
    
            plt.xlabel(ax)
            plt.ylabel(ay)
            plt.minorticks_on()
            plt.grid(b=True, which='major')
            plt.grid(b=True, which='minor',alpha=0.4)
            plt.suptitle(title, fontsize='x-large', fontweight='bold')
        plt.show() 
        
    def average(self):
        plotting = np.mean(self.data_memory,axis=1)[-10:,:]
        plotting = np.mean(plotting,axis=0)
        
        return plotting
        
    def plot_result(self, title ="(DDPG) Received SNR Vs BS-USer Horizontal Distance",
                    ax="BS-USer Horizontal Distance (m)", ay="Received SNR (dB)"):
        
        plotting = np.mean(self.data_memory,axis=1)[-10:,:]
        plotting = np.mean(plotting,axis=0)
        axis_set = self.label
        
        plt.figure()
        plt.plot(axis_set, plotting, linewidth=1,marker="o")
        plt.xlim(np.min(axis_set)-(np.min(axis_set)*0.1),
                 np.max(axis_set)+(np.min(axis_set)*0.1))
        plt.ylim(np.min(plotting)-(np.max(plotting)-np.min(plotting))*0.1,
                  np.max(plotting)+(np.max(plotting)-np.min(plotting))*0.1)
        plt.autoscale(enable=False, axis='x')
                
        plt.xlabel(ax)
        plt.ylabel(ay)
        plt.grid(which='major', axis='both')
        plt.suptitle(title, fontsize='large', fontweight='bold')     
        
        plt.show()
    
    def save_value(self,file_name):
        
        dirName = 'Value'
        fileName = file_name
        file = dirName+'/'+fileName+'_sample_'+str(self.num_sample)+'_episode_'+str(self.steps)+'.npy'
             
        if not os.path.exists(dirName):
            os.mkdir(dirName)
        value = np.mean(self.data_memory,axis=1)
        np.save(file,value)'''
