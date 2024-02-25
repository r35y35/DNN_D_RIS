import numpy as np
from agent_td3 import TD3_agent
from env_Td3 import RIS_D, UEs_and_utilities
#from tools_td3 import plotter_convergence
import matplotlib.pyplot as plt

'''M=4  # 4   #BS ant    5,10,15,20  vs EE 333333333333333333333333    np.arange(2,12,2)[ 2,  4,  6,  8, 10]
L=4     # [8,16,24,32,40] # [4,8,12,16,20]   # No of IRS      2,4,6,10,12........VS EE wrt p_maX22222222222222222
K=1  # user    2,4,6,8,19,,VS EE  
J=1
N=6 #   [8,16,32,64]
#N_set= [2,4,6,8,10,12] #np.arange(2,12,2)      #  8 # ref ele Of IRS 2,4,6,8,10,12 VS EE  111111111111111111111

P_B= 39 # dbm 39
P_B= np.power(10,(P_B/10))/1000

P_K= 10  #dbm 10     10^(P_R/10)/10^3;
P_K= np.power(10,(P_K/10))/1000

P_J= 10  #dbm 10     10^(P_R/10)/10^3;
P_J= np.power(10,(P_J/10))/1000

P_R= 10  #dbm 10
P_R= np.power(10,(P_R/10))/1000   #db2lin(9)

Pmax=30    # [5,10,15,20,25,30]  #dbm   #10^(P_k/10)/10^3;   pt_set = [4,8,12] 
power_properties = np.array([[0],[np.power(10,(Pmax/10))/1000]])    #np.array([[0],[pmax]]) pmax ko lin me change kiya hai /////(2, 1)
phase_properties = np.array([[0],[2*np.pi]])

K0=5
K0= np.power(10,(K0/10))/1000

u= 1.25 #v=0.8
awgn_var = -104
awgn_var= np.power(10,(awgn_var/10))/1000
channel_noise_var=-104
channel_noise_var=np.power(10,(channel_noise_var/10))/1000

# DRL Hyperparameter
episodes = 1000       # The number of each episodehn
max_steps = 100                # The number of step in each episode 
                
alpha =0.0001  #3 0.001   #1e-3       #0.00001 perform better                 # The Learning Rate
beta        = 0.002   #2e-3       #0.002                     # The Learning Rate
gamma       = 0.9                                 # Discount factor
tau  = 0.0001   #0.0002 #0.0004                               # The soft update coefficient
nn_actor    = np.array([[512],[512]])
nn_critic   = np.array([[512],[542]])
batch_size  = 64
buffer_size=  100000  #50000
update_actor_interval= 2

ep_reward_list = []  
avg_reward_list = []    
avg_score = np.zeros(episodes)

if __name__ == '__main__':
        UEs = UEs_and_utilities(M,N,K,J,L)
        env = RIS_D( M, N, K,J,L, P_B, P_K,P_J ,P_R,K0, u,awgn_var,channel_noise_var,power_properties,phase_properties, UEs)         
        agent = TD3_agent(nn_actor, nn_critic, env,alpha, beta, gamma, buffer_size,tau,batch_size, update_actor_interval, noise=0.01)      
        x = np.zeros((episodes,episodes))       
        y = np.zeros((episodes,episodes), dtype = complex)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        for i in range(episodes):            
            done = False
            observation = env.reset_state()             #state                                                
            for i0 in range(max_steps):
                action = agent.choose_action(observation)   #  , evaluate= False)##  array of float(36).....36 row                       actnjmion ##i modified evaliuate= False before this term was not here ##   from ddpg                              
                reward, next_state = env.step(action)                 #next_stATE                    
                agent.remember(observation, action ,reward,next_state)             ## store transition## i have added done in extra #score += reward  ##v  frpm ddpg                
                agent.learn()                                 ## tarnsition to new state   ###  set currenrt state to new state  
                observation = next_state                          ##  state = next_ state ## transition to new state                              
                ep_reward_list.append(reward)  
            avg_score[i]=np.mean(ep_reward_list[-100:])
            # reward_set[i,cntr] = avg_score[i]
            print('episode',i,'reward %.1f' % reward, 'avg score %.1f' % avg_score[i]) 
                                        
plt.plot([i for i in range(episodes)],avg_score ,linewidth=1, label="M=8, L=8, K=1,N=8, Pmax= 20" )            #,label='N =' ) 
plt.legend(loc='lower right',fontsize = 7)
plt.xlabel("episodes")
plt.ylabel("score")# score
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor',alpha=0.4)
plt.title('TD3')
plt.show ()'''
              
  
    
      

M=4   #[10,20,30,40,50]   # 4   #BS ant    5,10,15,20  vs EE 333333333333333333333333    np.arange(2,12,2)[ 2,  4,  6,  8, 10]
L=4     # [8,16,24,32,40] # [4,8,12,16,20]   # No of IRS      2,4,6,10,12........VS EE wrt p_maX22222222222222222
K=1  # user    2,4,6,8,19,,VS EE  
J=1
N=8 #   [8,16,32,64]
#N_set= [2,4,6,8,10,12] #np.arange(2,12,2)      #  8 # ref ele Of IRS 2,4,6,8,10,12 VS EE  111111111111111111111

P_B= 39 # dbm 39
P_B= np.power(10,(P_B/10))/1000

P_K= 10  #dbm 10     10^(P_R/10)/10^3;
P_K= np.power(10,(P_K/10))/1000

P_J= 10  #dbm 10     10^(P_R/10)/10^3;
P_J= np.power(10,(P_J/10))/1000

P_R= 10  #dbm 10
P_R= np.power(10,(P_R/10))/1000   #db2lin(9)

Pmax=30  # [5,10,15,20,25,30]  #dbm   #10^(P_k/10)/10^3;   pt_set = [4,8,12]   30
Pmax_set=[5,10,15,20,25,30]
power_properties = np.array([[0],[np.power(10,(Pmax/10))/1000]])    #np.array([[0],[pmax]]) pmax ko lin me change kiya hai /////(2, 1)
phase_properties = np.array([[0],[2*np.pi]])

K0=5
K0= np.power(10,(K0/10))/1000

u= 1.25 #v=0.8
awgn_var = -104
awgn_var= np.power(10,(awgn_var/10))/1000
channel_noise_var=-104
channel_noise_var=np.power(10,(channel_noise_var/10))/1000

# DRL Hyperparameter
episodes = 1000       # The number of each episodehn
max_steps = 100               # The number of step in each episode 
                
alpha =0.0001  #3 0.001   #1e-3       #0.00001 perform better                 # The Learning Rate
beta        = 0.002   #2e-3       #0.002                     # The Learning Rate
gamma       = 0.9                                 # Discount factor
tau  = 0.0001   #0.0002 #0.0004                               # The soft update coefficient
nn_actor    = np.array([[512],[512]])
nn_critic   = np.array([[512],[542]])
batch_size  = 64
buffer_size=  100000  #50000
update_actor_interval= 2

ep_reward_list = []  
avg_reward_list = []    
avg_score = np.zeros(episodes)

reward_set=np.zeros((episodes,len(Pmax_set)))
score_pt = np.zeros((len(Pmax_set))) 

if __name__ == '__main__':
    for cntr in range(len(Pmax_set)):
        power_properties= Pmax_set[cntr]        
        
        UEs = UEs_and_utilities(M,N,K,J,L)
        env = RIS_D( M, N, K,J,L, P_B, P_K,P_J ,P_R,K0, u,awgn_var,channel_noise_var,power_properties,phase_properties, UEs)         
        agent = TD3_agent(nn_actor, nn_critic, env,alpha, beta, gamma, buffer_size,tau,batch_size, update_actor_interval, noise=0.01)      
        x = np.zeros((episodes,episodes))       
        y = np.zeros((episodes,episodes), dtype = complex)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
        for i in range(episodes):            
            done = False
            observation = env.reset_state()             #state                                                
            for i0 in range(max_steps):
                action = agent.choose_action(observation)   #  , evaluate= False)##  array of float(36).....36 row                       actnjmion ##i modified evaliuate= False before this term was not here ##   from ddpg                              
                reward, next_state = env.step(action)                 #next_stATE                    
                agent.remember(observation, action ,reward,next_state)             ## store transition## i have added done in extra #score += reward  ##v  frpm ddpg                
                agent.learn()                                 ## tarnsition to new state   ###  set currenrt state to new state  
                observation = next_state                          ##  state = next_ state ## transition to new state                              
                ep_reward_list.append(reward)  
            avg_score[i]=np.mean(ep_reward_list[-100:])
            reward_set[i,cntr] = avg_score[i]
            print('episode',i,'reward %.1f' % reward, 'avg score %.1f' % avg_score[i]) 
              
#average_score_pt = np.mean(reward_set,axis=0) 

for i in range(len(Pmax_set)):    
    plt.plot(np.arange(episodes),reward_set[:,i],linewidth=1,label='P_max ='+str(Pmax_set[i]))                                           
  # plt.plot([i for i in range(episodes)],avg_score ,linewidth=1, label="M=8, L=8, K=1,N=8, Pmax= 20" )            #,label='N =' ) 
plt.legend(loc='lower right',fontsize = 7)
plt.xlabel("episodes")
plt.ylabel("score")# score
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor',alpha=0.4)
plt.title('TD3')
plt.show ()