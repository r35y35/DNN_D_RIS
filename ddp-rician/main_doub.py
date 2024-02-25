import warnings
warnings.filterwarnings('ignore')
import numpy as np
from ddpg_doub import Agent 
from  env_doub import RIS_D, UEs_and_utilities
import matplotlib.pyplot as plt

def db2lin(db):                                                                                                       ## decible to   linear 
    lin = np.power(10,(db/10))
    return lin 
M=4    #[5,10]   #[4,8] # 4   #BS ant    5,10,15,20  vs EE 333333333333333333333333    np.arange(2,12,2)[ 2,  4,  6,  8, 10] 44444444444444444444444444444444444444444--best
L= 4 #[4,16,24,32,40]    #[np.transpose(np.conjugate( F[:,i])) ]  # No of IRS      2,4,6,10,12........VS EE wrt p_maX2222222222222222244444444444444444444444==beast
K=J=1#[4,8,12,16,20]  # user    2,4,6,8,19,,VS EE  444444444444444

N =8 #[8,16,32,64] 

P_B=39  #db2lin(39)/1000# 39 # dbm 39
P_B= np.power(10,(P_B/10))/1000

P_K=10     #db2lin(10)/1000# 10  #dbm 10     10^(P_R/10)/10^3;
P_K= np.power(10,(P_K/10))/1000

P_J =10     #db2lin(10)/1000# 10  #dbm 10     10^(P_R/10)/10^3;
P_J = np.power(10,(P_J/10))/1000

P_R=10   #db2lin(10)/1000# 10  #dbm 10
P_R= np.power(10,(P_R/10))/1000

K0=5 #3
K0= np.power(10,(K0/10))/1000
u= 1.25 #v=0.8
awgn_var = -104
awgn_var= np.power(10,(awgn_var/10))/100
channel_noise_var=-104
channel_noise_var= np.power(10,(channel_noise_var/10))/100
# channel_est_error=0

Pmax=20  #dbm    #10^(P_k/10)/10^3;   pt_set = [4,8,12]
 
#Pmax_set=[5,10,15,20,25,30,35] #[10,20,30,40,50] #[5,10,15,20,25,30,35,40,45,50]   #[5,10,15,20,25,30]
power_properties = np.array([[0],[np.power(10,(Pmax/10))/1000]])    #np.array([[0],[pmax]]) pmax ko lin me change kiya hai /////(2, 1)
phase_properties = np.array([[0],[2*np.pi]])    ###############################(2, 1)kl

# bs_loc = np.array((0,0,0),dtype=np.float32)
# irs_loc = np.array((100,0,50),dtype=np.float32)
# user_loc = np.array([[np.random.uniform(-100,100)],[np.random.uniform(-100,100)]])

net = np.array([[512],[512]]) #512
alpha = 0.001   #0.001  # 0.0001 perform better 
beta= 0.002  #0.001
gamma = 0.95  #.95best
tau =0.0001
batch_size = 64 #ty
max_size =1000000  # buffer size

# net = np.array([[128],[128]])    #nn_actor
# net  = np.array([[256],[256]])   #nn_critic

episodes=1000
max_steps=100
ep_reward_list = []  
avg_reward_list = []    
avg_score = np.zeros(episodes)

# reward_set = np.zeros((episodes,len(L_set)))                             
# score_pt = np.zeros((len(L_set)))    

# if __name__ == '__main__':
#     for cntr in range(len(L_set)):
#         L=L_set[cntr]
#         power_properties = np.array([[0],[np.power(10,(Pmax/10))/1000]])
              
#         UEs = UEs_and_utilities(M,N,K,J,L)                                                                                                                #pt = db2lin(pt)                                        
#         env = RIS_D( M, N, K,J,L, P_B, P_K, P_J, K0, P_R,u,awgn_var,channel_noise_var,power_properties,phase_properties,UEs )     
#         input_dims = env.state_dim                                                          #state_dim
#         action_dim = env.action_dim                                                         # action_dim
#         max_actioin= 1    
#         agent = Agent(net, env ,alpha,beta, gamma, max_size,tau, batch_size, noise = 0.1)                                                                                               #(input_dims , action_dim , net,env = env )             
#         x = np.zeros((episodes,episodes))       
#         y = np.zeros((episodes,episodes), dtype = complex)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
#         for i in range(episodes):            
#               done = False
#               observation = env.reset_state()             #state                                                
#               for i0 in range(max_steps):
#                   action = agent.choose_action(observation)   #  , evaluate= False)##  array of float(36).....36 row                       actnjmion ##i modified evaliuate= False before this term was not here ##   from ddpg                                
#                   reward, next_state = env.step(action)                 #next_stATE                            
#                   agent.remember(observation, action, reward, next_state)             ## store transition## i have added done in extra #score += reward  ##v  frpm ddpg                
#                   agent.learn()                                 ## tarnsition to new state   ###  set currenrt state to new state  
#                   observation = next_state                          ##  state = next_ state ## transition to new state                                
#                   ep_reward_list.append(reward)  
#               avg_score[i]=np.mean(ep_reward_list[-100:])
#               reward_set[i,cntr] = avg_score[i] #  dimension 5 h                                           
#               print('episode',i,'reward %.1f' % reward, 'avg score %.1f' % avg_score[i])  # (avg_score[i])\
# #average_score_pt = np.mean(reward_set,axis=0) 
# for i in range(len(L_set)):
#     # plt.plot(Pmax_set,average_score_pt,linewidth=1 ,label="P_max=10")  #  diff [2,4,5,7,9,]
#       plt.plot(np.arange(episodes), reward_set[:,i],linewidth=1,label='M= '+str(L_set[i]))  # for multiple graph ///  correct version   
# plt.legend(loc='lower right',fontsize = 10)
# plt.xlabel("episode")
# plt.ylabel("score")
# plt.title("DDPG (Episode VS Reward- star-irs)")
# plt.grid(b=True, which='major')
# plt.grid(b=True, which='minor',alpha=0.4)
# plt.show () 






if __name__ == '__main__':
        UEs = UEs_and_utilities(M,N,K,J,L)                                                                                                                #pt = db2lin(pt)                                        
        env = RIS_D( M, N, K,J,L, P_B, P_K, P_J, K0, P_R,u,awgn_var,channel_noise_var,power_properties,phase_properties,UEs )         
        input_dims = env.state_dim                                                          #state_dim
        action_dim = env.action_dim                                                         # action_dim
        max_actioin= 1        
        agent = Agent(net, env ,alpha, beta, gamma, max_size,tau, batch_size, noise = 0.1)                                                                                                   #(input_dims , action_dim , net,env = env )                 
        for i in range(episodes):            
              done = False
              observation = env.reset_state()             #state                                                
              for i0 in range(max_steps):
                action = agent.choose_action(observation)   #  , evaluate= False)##  array of float(36).....36 row                       actnjmion ##i modified evaliuate= False before this term was not here ##   from ddpg                                
                reward, next_state = env.step(action)                 #next_stATE                             
                agent.remember(observation, action, reward, next_state)             ## store transition## i have added done in extra #score += reward  ##v  frpm ddpg                
                agent.learn()                                 ## tarnsition to new state   ###  set currenrt state to new state  
                observation = next_state                          ##  state = next_ state ## transition to new state                                
                ep_reward_list.append(reward)  
              avg_score[i]=np.mean(ep_reward_list[-100:])  
              # reward_set[i,cntr] = avg_score[i]     
              print('episode',i,'reward %.1f' % reward, 'avg score %.1f' % avg_score[i])  # (avg_score[i])\
# for i in range(len(pt_set)):                                           
plt.plot([i for i in range(episodes)],avg_score ,linewidth=1, label="M=8, L=8, K=1,N=6, Pmax= 10" )            #,label='N =' ) 
#plt.plot([i for i in range(episode)],avg_score ,linewidth=1, label="pt= 30 , learning_rate= 0.001, decay_rate= 0.00001")  
plt.legend(loc='lower right',fontsize = 7)
plt.xlabel("episode")
plt.ylabel("score")# score
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor',alpha=0.4)
plt.show () 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
# if __name__ == '__main__':
#     for cntr in range(len(Pmax_set)):
#         Pmax=Pmax_set[cntr]
#         power_properties = np.array([[0],[np.power(10,(Pmax/10))/1000]])
        
#         UEs = UEs_and_utilities(M,N,K,J,L)                                                                                                                #pt = db2lin(pt)                                        
#         env = RIS_D( M, N, K,J,L, P_B, P_K, P_J, P_R,u,awgn_var,power_properties,phase_properties, UEs )     
#         input_dims = env.state_dim                                                          #state_dim
#         action_dim = env.action_dim                                                         # action_dim
#         max_actioin= 1    
#         agent = Agent(net, env ,alpha, beta, gamma, max_size,tau, batch_size, noise = 0.1)                                                                                               #(input_dims , action_dim , net,env = env )             
#         x = np.zeros((episodes,episodes))       
#         y = np.zeros((episodes,episodes), dtype = complex)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
#         for i in range(episodes):            
#               done = False
#               observation = env.reset_state()             #state                                                
#               for i0 in range(max_steps):
#                   action = agent.choose_action(observation)   #  , evaluate= False)##  array of float(36).....36 row                       actnjmion ##i modified evaliuate= False before this term was not here ##   from ddpg                                
#                   reward, next_state = env.step(action)                 #next_stATE                            
#                   agent.remember(observation, action, reward, next_state)             ## store transition## i have added done in extra #score += reward  ##v  frpm ddpg                
#                   agent.learn()                                 ## tarnsition to new state   ###  set currenrt state to new state  
#                   observation = next_state                          ##  state = next_ state ## transition to new state                                
#                   ep_reward_list.append(reward)  
#               avg_score[i]=np.mean(ep_reward_list[-100:])
#               reward_set[i,cntr] = avg_score[i] #  dimension 5 h                                           
#               print('episode',i,'reward %.1f' % reward, 'avg score %.1f' % avg_score[i])  # (avg_score[i])\
# #average_score_pt = np.mean(reward_set,axis=0) 
# for i in range(len(Pmax_set)):
#    # plt.plot(Pmax_set,average_score_pt,linewidth=1 ,label="P_max=10")  #  diff [2,4,5,7,9,]
#      plt.plot(np.arange(episodes), reward_set[:,i],linewidth=1,label='N = '+str(Pmax_set[i]))  # for multiple graph ///  correct version 
#    # plt.plot(Pmax_set,average_score_pt,linewidth=1,label='Pmax ='+str(Pmax_set[i]))    #   for x_axis   {N_set}////////////////// correct verson ////////////
# # plt.plot(np.arange(episodes),avg_score,linewidth=1,label='N = '+str(N_set[i]))  
#  # plt. plot(N_set,score_pt,linewidth=1,label='Pt = '+str(N_set[i]))
# # plt.plot(np.arange(episodes),reward_set[:,i],linewidth=1,label='lr ='+str(N_set[i]))                                           
# #plt.plot([i for i in range(episodes)],avg_score ,linewidth=1, label="M=8, L=8, K=1,J=1, N=4, Pmax= 30" )            #,label='N =' ) 
#  #plt.plot([i for i in range(episode)],avg_score ,linewidth=1, label="pt= 30 , learning_rate= 0.001, decay_rate= 0.00001")  
# plt.legend(loc='lower right',fontsize = 10)

# plt.xlabel("episode")
# plt.ylabel("score")
# plt.title("DDPG (Episode VS Reward- star-irs)")
# plt.grid(b=True, which='major')
# plt.grid(b=True, which='minor',alpha=0.4)
# plt.show () 








# if __name__ == '__main__':
#     for cntr in range(len(N_set)):
#         N=N_set[cntr]
          
#         UEs = UEs_and_utilities(M,N,K,L)
#                                                                                                                 #pt = db2lin(pt)                                        
#         env = RIS_D( M, N, K,L, P_B, P_K, P_R,u,awgn_var,power_properties,phase_properties, UEs ) 
        
#         input_dims = env.state_dim                                                          #state_dim
#         action_dim = env.action_dim                                                         # action_dim
#         max_actioin= 1
        
#         agent = Agent(net, env ,alpha, beta, gamma, max_size,tau, batch_size, noise = 0.1) 
#                                                                                                   #(input_dims , action_dim , net,env = env ) 
                
#         x = np.zeros((episodes,episodes))       
#         y = np.zeros((episodes,episodes), dtype = complex)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                     
#         for i in range(episodes):            
#               done = False
#               observation = env.reset_state()             #state                                                
#               for i0 in range(max_steps):
#                   action = agent.choose_action(observation)   #  , evaluate= False)##  array of float(36).....36 row                       actnjmion ##i modified evaliuate= False before this term was not here ##   from ddpg                
                
#                   reward, next_state = env.step(action)                 #next_stATE
                             
#                   agent.remember(observation, action, reward, next_state)             ## store transition## i have added done in extra #score += reward  ##v  frpm ddpg                
#                   agent.learn()                                 ## tarnsition to new state   ###  set currenrt state to new state  
#                   observation = next_state                          ##  state = next_ state ## transition to new state  
                              
#                   ep_reward_list.append(reward)  
#               avg_score[i]=np.mean(ep_reward_list[-100:])
#               reward_set[i,cntr] = avg_score[i] #  dimension 5 h                                           
#         print('episode',i,'reward %.1f' % reward, 'avg score %.1f' % avg_score[i])  # (avg_score[i])\
# average_score_pt = np.mean(reward_set,axis=0) 
# for i in range(len(N_set)):
#      plt.plot(np.arange(episodes), reward_set[:,i],linewidth=1,label='N = '+str(N_set[i]))    #multiple graph 
#     # plt.plot(np.arange(episodes),avg_score,linewidth=1,label='N = '+str(N_set[i]))  
#    #  plt.plot(N_set,average_score_pt,linewidth=1,label='lr ='+str(N_set[i]))
# # plt.plot(np.arange(episodes),avg_score,linewidth=1,label='N = '+str(N_set[i]))  
#  # plt. plot(N_set,score_pt,linewidth=1,label='Pt = '+str(N_set[i]))
# # plt.plot(np.arange(episodes),reward_set[:,i],linewidth=1,label='lr ='+str(N_set[i]))                                           
# # plt.plot([i for i in range(episodes)],avg_score ,linewidth=1, label="M=8, L=8, K=1,N=6, Pmax= 40" )            #,label='N =' ) 
#  #plt.plot([i for i in range(episode)],avg_score ,linewidth=1, label="pt= 30 , learning_rate= 0.001, decay_rate= 0.00001")  
#      plt.legend(loc='lower right',fontsize = 7)
#      plt.xlabel("episode")
#      plt.ylabel("score")# score
#      plt.grid(b=True, which='major')
#      plt.grid(b=True, which='minor',alpha=0.4)
# plt.show () 













# =============================================================================N=2,4,6,8,10,12
# sampling    = 3
# 
# ddpg_reward_temp = np.zeros((episodes,max_steps))
# ddpg_plotter = plotter_convergence(episodes,sampling,N_set)
# 
# for cntr in range(len(N_set)):
#     N = N_set[cntr]
#     UEs = UEs_and_utilities(M,N,K,L)    
#     
#     for sample_ in range(sampling):
#         env = RIS_D( M, N, K,L, P_B, P_K, P_R,u,awgn_var,power_properties,phase_properties, UEs ) 
#                 
#         input_dims = env.state_dim                                                          #state_dim
#         action_dim = env.action_dim                                                         # action_dim
#         max_actioin= 1
#         
#         agent = Agent(net, env ,alpha, beta, gamma, max_size,tau, batch_size, noise = 0.1) 
#         
#        
#         for i0 in range(episodes):
#             state = agent.env.reset_state() # shape of state from lingkungan (state_dim,)
#             
#             for i in range(max_steps):
#                 action = agent.choose_action(state)
#                 reward, new_state = env.step(action)
#                 agent.remember(state, action, reward, new_state)
#                 agent.learn()
#                 ddpg_reward_temp [i0,:]= reward
#                 state = new_state                                                                             
#                           
#             avg_score = np.mean(ddpg_reward_temp[i0,:])
#             ddpg_plotter.record(avg_score,i0,sample_,cntr)
#             print('D =',N,'Sample',sample_+1,'Episode',i0+1, 'reward %.1f' % (reward), 'avg score %.1f' % avg_score)
#         print('\nScenario\n','Sampling ',str(sample_+1),'Scenario Done!!!--- \n')
# 
# ddpg_plotter.plot(grid=episodes,title="DDPG")
# ddpg_plotter.plot_result(title ="(DDPG) Received SNR Vs BS-USer Horizontal Distance")
# 
# =============================================================================

























# reward_set = np.zeros((episode,len(P_max_set)))                             
# score_pt = np.zeros((len(Pmax_set)))    

# if __name__ == '__main__':////correct vala hai
#         UEs = UEs_and_utilities(M,N,K,L)
#                                                                                                                 #pt = db2lin(pt)                                        
#         env = RIS_D( M, N, K,L, P_B, P_K, P_R,u,awgn_var,power_properties,phase_properties, UEs ) 
        
#         input_dims = env.state_dim                                                          #state_dim
#         action_dim = env.action_dim                                                         # action_dim
#         max_actioin= 1
        
#         agent = Agent(net, env ,alpha, beta, gamma, max_size,tau, batch_size, noise = 0.1) 
#                                                                                                   #(input_dims , action_dim , net,env = env ) 
                
#         x = np.zeros((episode,episode))       
#         y = np.zeros((episode,episode), dtype = complex)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                     
#         for i in range(episode):            
#               done = False
#               observation = env.reset_state()             #state                                                
#               for i0 in range(max_steps):
#                 action = agent.choose_action(observation)   #  , evaluate= False)##  array of float(36).....36 row                       actnjmion ##i modified evaliuate= False before this term was not here ##   from ddpg                
                
#                 reward, next_state = env.step(action)                 #next_stATE
                             
#                 agent.remember(observation, action, reward, next_state)             ## store transition## i have added done in extra #score += reward  ##v  frpm ddpg                
#                 agent.learn()                                 ## tarnsition to new state   ###  set currenrt state to new state  
#                 observation = next_state                          ##  state = next_ state ## transition to new state  
                              
#                 ep_reward_list.append(reward)  
#               avg_score[i]=np.mean(ep_reward_list[-100:])  
#               # reward_set[i,cntr] = avg_score[i]     
#               print('episode',i,'reward %.1f' % reward, 'avg score %.1f' % avg_score[i])  # (avg_score[i])\
# # for i in range(len(pt_set)):                                           
# plt.plot([i for i in range(episode)],avg_score ,linewidth=1, label="M=8, L=8, K=1,N=6, Pmax= 10" )            #,label='N =' ) 
# #plt.plot([i for i in range(episode)],avg_score ,linewidth=1, label="pt= 30 , learning_rate= 0.001, decay_rate= 0.00001")  
# plt.legend(loc='lower right',fontsize = 7)
# plt.xlabel("episode")
# plt.ylabel("score")# score
# plt.grid(b=True, which='major')
# plt.grid(b=True, which='minor',alpha=0.4)
# plt.show () 



# best result
# alpha _bita =0.001  
# tau=0.0001
# gamma=0.99











# import numpy as np
# from ddpg_doub import Agent 
# from  env_doub import RIS_D
# import matplotlib.pyplot as plt
# def db2lin(db):                                                                                                       ## decible to   linear 
#     lin = np.power(10,(db/10))
#     return lin 
# M= 2   #BS ant
# N= 4  # ref ele of IRS
# K= 4    # user
# L= 2  # NO Of IRS 2,4
# Pmax_set = [10,15,20 ,25,30.40]  #dbm    #10^(P_k/10)/10^3;   pt_set = [4,8,12] 
# power_properties = np.array([[0],[np.power(10,(Pmax/10))/1000]])

# P_B= 7.94   # dbm 39
# P_K= .01  #dbm 10
# P_R= .01   #dbm 10
# u= 1.25 #v=0.8
# awgn_var = 0.01

# net = np.array([[128],[128]]) #512
# alpha=0.001 
# beta=0.001
# gamma =0.99
# tau =0.0001
# batch_size =64
# max_size =100000  # buffer size
# net = np.array([[256],[256]])  #512 
# # net = np.array([[128],[128]])    #nn_actor
# # net  = np.array([[256],[256]])   #nn_critic

# episode = 50
# max_steps=  150
# ep_reward_list = []  
# avg_reward_list = []    
# avg_score = np.zeros(episode)

# reward_set = np.zeros((episode,len(P_max_set)))                              #  len(pt_set))) //////array o float (6,3))  6..row 3,column 
# score_pt = np.zeros((len(Pmax_set)))    

# if __name__ == '__main__':           
#                                                                                                                 #pt = db2lin(pt)                                        
#         env = RIS_D( M, N, K,L, P_B, P_K, P_R, u,awgn_var,power_properties) 
        
#         input_dims = env.state_dim                                                          #state_dim
#         action_dim = env.action_dim                                                         # action_dim
#         max_actioin= 1
        
#         agent = Agent(input_dims , action_dim , net,env = env ) 
                
#         x = np.zeros((episode,episode))       
#         y = np.zeros((episode,episode), dtype = complex)
                     
#         for i in range(episode):            
#               done = False
#               observation = env.reset_state()             #state                                                
#               for i0 in range(max_steps):
#                 action = agent.choose_action(observation, evaluate= False)##  array of float(36).....36 row                       actnjmion ##i modified evaliuate= False before this term was not here ##   from ddpg                
                
#                 reward, observation_ = env.step(action)                 #next_stATE
                             
#                 agent.remember(observation, action, reward, observation_)             ## store transition## i have added done in extra #score += reward  ##v  frpm ddpg                
#                 agent.learn()                                 ## tarnsition to new state   ###  set currenrt state to new state  
#                 observation = observation_                          ##  state = next_ state ## transition to new state  
                              
#                 ep_reward_list.append(reward)  
#               avg_score[i]=np.mean(ep_reward_list[-100:])  
#               print('episode',i,'reward %.1f' % reward, 'avg score %.1f' % avg_score[i])  # (avg_score[i])\
                                          
# plt.plot([i for i in range(episode)],avg_score ,linewidth=1, label="M=2, N=4, K=4,L=2, pmax=20" )            #,label='N =' ) 
# #plt.plot([i for i in range(episode)],avg_score ,linewidth=1, label="pt= 30 , learning_rate= 0.001, decay_rate= 0.00001")  
# plt.legend(loc='lower right',fontsize = 7)
# plt.xlabel("episode")
# plt.ylabel("score")# score
# plt.grid(b=True, which='major')
# plt.grid(b=True, which='minor',alpha=0.4)
# plt.show () 


#======================  1 february
# import numpy as np

# class UEs_and_utilities():
        
#     def __init__(self,M,N,K,L):
#         self.K = K
#         self.L = L
#         self.bs_loc = np.array([[0],[0]])   ##(0 ,0)
#         self.generate_irs_loc()              ##(100,0,50)
#         self.generate_user_loc()                 # #random( 54,32)
      
#     def generate_irs_loc(self):
#         irs_loc = np.zeros((3,L),dtype = float)
#         for i in range(self.L):
#             irs_loc[:,i] = np.array([[np.cos(2*i*np.pi/self.L)*100],[np.sin(2*i*np.pi/self.L)*100],[50]])   #(3, 1)  #sirf ro k liye 
#         self.irs_loc = irs_loc
        
#     def generete_user_loc(self):   #rand(Num_User,2)*lengA;
#         user_loc = np.zeros((2,self.K))
#         for i in range(self.K):
#             user_loc[:,i] = np.array([[np.random.uniform(-100,100)],[np.random.uniform(-100,100)]])     #random( 54,32)
#         self.user_loc = user_loc        
# #dist 
#     def generate_BS_IRS(self):       ## BS_IRS
#         d_1 = np.zeros((self.N,self.M, self.L),dtype=np.float32)     #
#         for i in range(self.N):
#             for i0 in range(self.M):
#                 for i00 in range(self.L):
#                      G[i,i0,i00] = np.sqrt((self.irs_loc[0]-self.bs_loc[0])**2 + 
#                                        (self.irs_loc[1]-self.bs_loc[1])**2 + (self.irs_loc[2])**2)   #@@@@@@@@@@@
                
                
#        def get_IRS_US(self):     # irs_user   ### fix 
#         d_2  = np.zeros((self.N,self.K,self.L),dtype=np.complex)  #h
#         for i in range(self.N):
#             for i0 in range(self.k):
#                 for i00 in range(L):
#                     h[:,0,i0,i] = np.sqrt(fn.db2lin(self.PL0)*np.power(self.h[i,i0],-self.a2)
#                                           )*((np.sqrt(self.b1/(1+self.b1))*self.phi)+(np.sqrt(1/(2))*self.h)) 
                
                
                    
#        def get_BS_US(self):     # BS _User  ##  fix
#         d_3 = np.zeros((self.M,self.K),dtype=np.complex)  #g
#         for i in range(self.M):
#             for i0 in range(self.K):
#                 g[:,0,i0,i] = np.sqrt(fn.db2lin(self.b0)*np.power(self.d_irsm[i,i0],-self.a2)
#                                           )*((np.sqrt(self.b1/(1+self.b1))*self.phi_array)+(np.sqrt(1/(2))*self.h_nlos))
                
# #pathloss  distance bw vala  constant vala   and  difrent diffrent pl vala ause hoga  
# # PL0(pl at refrence distance) -10a(PL exponent)log10(distance(d1,d2,d3)/distance d0 (constant) )

#         PL1 = lin(lin(self.PL0) - 10*self.PLexp1*np.log10(self.d_1/self.D0))    #  BS-IRs 
#         PL2 = lin(lin(self.PL0) - 10*self.PLexp2*np.log10(self.d_2/self.D0))       #  IRs-User 
#         PLd = lin(lin(self.PL0) - 10*self.PLexp2*np.log10(self.d_3/self.D0))       #  BS-User
                       
     
#     def generate_channel(self):    # distance + pathloss parame              
    
#             self.g= np.random.rand(self.M, 1,self.K) +(1j * (np.random.rand(self.M,1,self.K)))    # BS to user      1*M
#             self.G= np.random.rand(self.N, self.M,self.L) +(1j * (np.random.rand(self.N, self.M,self.L)))                  #N*M
#             self.h= np.random.rand(self.N,self.K,self.L) +(1j * (np.random.rand(self.N,self.K,self.L)))      # Ris to user 1*N 
    
    
#             PL1 = lin(lin(self.a) - 10*self.PLexp1*np.log10(self.d_1/self.b))    #  BS-IRs 
#             PL2 = lin(lin(self.a) - 10*self.PLexp2*np.log10(d_2/self.b))       #  IRs-User 
#             PLd = lin(lin(self.a) - 10*self.PLexp2*np.log10(d_3/self.b))       #  BS-User
    
    
#         g   = np.sqrt(PL1)*((np.sqrt(self.K1/(self.K1+1))*Gbar)+(np.sqrt(1/(self.K1+1))*ghead))      # Channel G (BS-IRs)   
#         G  = np.sqrt(PL2)*((np.sqrt(self.K2/(self.K2+1))*hr_bar)+(np.sqrt(1/(self.K1+1))*hrhead))   # Channel hr (IRs-user)
#         h  = np.sqrt(PL3)*hdhead 
        
#         return g,G,h
                    
   
    
   
    

# class RIS_D(object):
       
#     def __init__(self, M, N, K,L,P_B, P_K, P_R,u,awgn_var,power_properties, bita, k1,k2,k3,D0,lamda):        
#         self.M = M        # BS station ante
#         self.N = N        #  IRS element N_l
#         self.K = K         # user  
#         self.L= L           # no of irs       
#       # channel 
#         # self.g= np.random.rand(self.M, 1,self.K) +(1j * (np.random.rand(self.M,1,self.K)))    # BS to user      1*M
#         # self.G= np.random.rand(self.N, self.M,self.L) +(1j * (np.random.rand(self.N, self.M,self.L)))                  #N*M
#         # self.h= np.random.rand(self.N,self.K,self.L) +(1j * (np.random.rand(self.N,self.K,self.L)))      # Ris to user 1*N            
#         self.g= np.ones((self.M,1,self.K),dtype=complex)
#         self.G= np.ones((self.N,self.M,self.L),dtype=complex)
#         self.h= np.ones((self.N,self.K,self.L),dtype=complex)
#         # x = np.zeros(self.L)
        
# #pathloss parameter
#        self.d_ver  = d_ver
#        self.d_hor  = d_hor
#        self.d_fix  = d_fix
#        self.Do     = Do
#        self.PLo    = PLo
#        self.PLexp1 = PLexp1
#        self.PLexp2 = PLexp2
        
#         self.PL0=PL0    # PL at the refrence distance of D0   constant
#         self.D0=D0  # refrence distance  constant
#         self.PLexp1 = PLexp1  # PL exponent
#         self.PLexp2 = PLexp2        
                
#         # power
#         self.u= u       #1/v = .`W11/0.8
#         self.Power_max = np.max(power_properties)     #100Watt  
#         self.P_B= P_B
#         self.P_K= P_K
#         self.P_R= P_R         #x.N.P_r      
#       # self.x=x              #x_l  [0,1]        
#       # action
#         self.theta = np.random.rand(self.N, self.L)*2*np.pi                           #np.random.rand(self.N)*2*np.pi                      
#         self.Phi = np.zeros(self.N, dtype=complex)                                #np.zeros(self.M,self.K   dtype= complex) 
#         self.w = np.zeros((self.M, self.K), dtype= complex) #BF vector for user K
#         self.x = np.zeros(self.L)      
      
#         self.action_dim = self.M*self.K + (self.N*self.L)+(self.L)+1                               # theta(N) + w(M,1) + X((0,1 ))/theta(matrix) + W(vector) + x(vector)      
#         self.state_dim =   2*self.M*self.K      
      
#         # self.phase_normalize = np.max(phase_properties) - np.min(phase_properties) 
#         # self.phase_properties = phase_properties/self.phase_normalize
# # path_loss       
#         self.bita0 = bita0 
#         self.k1= k1
#         self.k2= k2
#         self.k3= k3
#         self.D0= D
#         self.lambda= lamda
        
#         self.awgn_var = awgn_var
#         self.done = None                 



        
# #dist 
#         def get_dis(self):       ## BS_IRS      
#             d_1 = np.zeros((self.N,self.M, self.L),dtype=np.float32)     #BS_IRS
#             d_2 = np.zeros((self.N),dtype=np.float32)               #BS_US
#             #d_3 = np.zeros((self.N,self.M),dtype=np.float32)   #IRS_US
#             for i in range(self.N):
#                     for i00 in range(self.L):
#                          d_1[i,i0,i00] = np.sqrt((self.irs_loc[0]-self.bs_loc[0])**2 + 
#                                            (self.irs_loc[1]-self.bs_loc[1])**2 + (self.irs_loc[2])**2)   #@@@@@@@@@@@ 
                         
#                     for i0 in range(self.M):
                
#                 d_2[i,i0] = np.sqrt((bs_loc[0]-self.user_loc[0])**2 +
#                                      (bs_loc[1]-self.user_loc[0])**2 + (self.bs_loc[2])**2)     # @@@@@@@@@@
            
#         return d_1, d_nm
    
    
#      def dis_irs_US(self):    #   IRS_ US
#         d_3 = np.zeros((self.N,self.M),dtype=np.float32)
#         for i in range(self.N):
#             for i0 in range(self.M):
#                 d_3[i,i0] = np.sqrt((self.irs_loc[0]-self.user_loc[i0,0,i])**2 + 
#                                        (self.irs_loc[1]-self.user_loc[i0,1,i])**2 + (self.irs_loc[2])**2)                
#         return d_3
    
# #chanel                                              
#      def get_h_irsm(self):
#          h_irsm = np.zeros((self.K,1,self.M,self.N),dtype=np.complex)
#          for i in range(self.N):
#             for i0 in range(self.M):
#                 h_irsm[:,0,i0,i] = np.sqrt(fn.db2lin(self.bita0)*np.power(self.d_1[i,i0],-self.k1)
#                                           )*((np.sqrt(self.b1/(1+self.b1))*self.phi_array)+(np.sqrt(1/(2))*self.h_nlos))
                
#         return h_1  
    
#     def get_channel(self,x_uav=None,y_uav=None):
#         d_1, d_2 = self.get_dis()        
#         h_1 = np.zeros((self.T,self.K,self.N),dtype=np.complex)
#         h_2 = np.zeros((self.T,1,self.M,self.N),dtype=np.complex)
        
#         for i in range(self.N):
#             for i0 in range(self.T):
#                 h_1[i0,:,i] = np.sqrt(fn.db2lin(self.bita0)*(np.power(d_nirs[i],-self.k1)))*self.phi_array
            
#             for i0 in range(self.M):
#                 for i1 in range(self.T):
#                     h_nm[i1,0,i0,i] = np.sqrt(fn.db2lin(self.bita0)*(np.power(d_nm[i,i0],-self.k3)))*self.h_nm_random[i1]
                     
#         return h_1, h_2
                    
        
        
        
        
        
        
#     def reset_state(self):
#        # G= np.random.rand(self.N, self.M,self.L) +(1j * (np.random.rand(self.N, self.M,self.L))) 
        
#         theta = np.random.rand(self.N,self.L)*2*np.pi          #(4,4)                             
#         phi = np.zeros((self.N,self.N,self.L) ,dtype=complex)         
#         for i in range(self.L):
#             phi[:,:,i] = np.diag(np.exp(1j*theta[:,i]))                       

#         s = np.zeros((self.M,self.K), dtype= np.complex)    ##2 dec///
#                                           ##  it will update at every episode 
#         x = np.zeros(self.L ) #, dtype=int)                                                                  ## int part I added 10 jan
#         for i in range(self.L):     #initiate xl
#             x[i]= np.random.randint(0,2)  #@@@@    (100,)
              
#         for i in range(self.K): 
#             s1= self.g[:,:,i].conj().T  #hkl_H   wk 
#             for i0 in  range(self.L):
#                 s1 += np.matmul(np.matmul(x[i0]*self.h[:,i,i0].conj().T,phi[:,:,i0]),self.G[:,:,i0]) #(4,)                 #[0.+0.j 0.+0.j 0.+0.j 0.+0.j]
#             s[:,i] = s1      #(4,)    
#         s_reshape = s.reshape(-1,1)                 # (16, 1)  only column me 
#         state = np.concatenate((np.real(s_reshape),np.imag(s_reshape)), axis=0)     #(32, 1)
#         return state[:,0]         #(16,)   
        
#     def step(self, action):           #(19,)          
#         x = np.zeros(self.L )                     #x =L     (4,)                                                     # I add dtyep 12 jan
#         theta_w = np.zeros((self.M*self.K))       # w= M,K     exp(j(theta))
#         theta_irs = np.zeros(self.N*self.L)       # theta (irs) (0,2*pi)  (16,)                     
#         for i in range(len(action)):
#             if i < 1:
#                 power_scale = action[i]*self.Power_max    #46
#             elif i < self.L+1:  #x
#                x[i-1] = int(np.around(action[i]) )        #                                            bef==int(np.around(action[i]))====i remove int 
#             elif i < (self.M*self.K)+self.L+1:   #w=exp(j(theta))               
#                 theta_w[i-self.L-1] = action[i]*2*np.pi   #0=0.6524216
#             else:                                
#                 theta_irs[i-(self.M*self.K) -self.L-1] = np.exp(1j*action[i]*2*np.pi)     #theta  
          
#         W = np.zeros((self.M,self.K),dtype=complex) #,dtype = int)    
#         cntr = 0  
#         for i in range(self.M):
#             for i0 in range(self.K):                                
#                 W[i,i0] = np.exp(1j*theta_w[cntr]) 
#                 cntr += 1                                    
#         W= (W/np.linalg.norm(W))*power_scale  
#         theta_irs = theta_irs.reshape((self.N,self.L))     #np.diag(theta_irs) (4,4)
#         phi = np.zeros((self.N,self.N,self.L))
#         for i in range(self.L):
#             phi[:,:,i] = np.diag(theta_irs[:,i])
#         # for i in range(self.L):            
#         #      phi[:,i] = np.diag(np.exp(1j*phi[:,i])) 
        
#         reward=0
#         s = np.zeros((self.M,self.K), dtype= np.complex)
#         sinr = np.zeros((self.K),dtype=np.float32)
#         rate = np.zeros((self.K),dtype=np.float32)  
                   
#         for i in range(self.K):                        
#             s1= self.g[:,:,i].conj().T    #(1, 4)          
#             for i0 in  range(self.L):                
#                 s1 += np.matmul(np.matmul(x[i0]*self.h[:,i,i0].conj().T,phi[:,:,i0]),self.G[:,:,i0])     #phi[:,i0] /// (4,)          
#                 s[:,i] = s1   #(4,)  (1,)
#                 num = np.matmul(s[:,i],W[:,i])    #(1,)                                                            #before==np.matmul(s1,W[:,i]).....Ichange == 16 jan np.matmul(s[:,i],W[:,i])
#                 num = np.power(np.abs(num),2)   # 156
                
#                 for i1 in range(self.K):                    
#                     if i1 != i:                        
#                         s2 = self.g[:,:,i].conj().T    #(1, 4)
#                         for i2 in range(self.L):                            
#                             s2 += np.matmul(np.matmul((x[i2]*self.h[:,i,i2].conj().T),phi[:,:,i2]),self.G[:,:,i2]) #phi[:,i2]//(4,)
#                             s[:,i]=s2
#                         denum_ = np.matmul(s[:,i],W[:,i1])     # (1,4),(4,)== (1,)           # before==np.matmul(s2,W[:,i1]) 
#                         denum = np.power(np.abs(denum_),2)  # 22.00
#                         denum += self.awgn_var
                        
#                 sinr[i] = num/denum   #int 0.823
#                 rate[i] = np.log2(1+(sinr[i]))       #0.86  sinr[i]...maine i remove ki
#                        #          np.log2(1+sinr[i])    #(4,)                               
#         #power: 
#         P_t=0                  
#         P_1 = 0
#         for K00 in range(self.K):                    
#            # w_k= W[:,K00]   #(4,)
#             w_kh= W[:,K00].conj().T   #(4,)
#             P_1 += np.abs(np.matmul(W[:,K00], w_kh)) *self.u   # complex    712        

#         P_3=0
#         for i in range(self.K):
#             P_3 += self.P_K 
                 
#         P_4 = 0     
#         for L0 in range (self.L):                                                          
#             P_4 += (x[L0]*self.N)*self.P_R    # 40   x[L0]=0                     
#         P_t += P_1 + self.P_B + P_3 + P_4   # 54
                            
#         reward = np.sum(rate)/P_t   # maine i htaya h rate[i]                                                             #  before==sum(rate)/P_T /// after==  16 jansum(rate[i])/P_T                                                   # np.sum(rate)/(P_t) 
#         s1_reshape = s.reshape(-1,1)   #(4, 1)
#         new_state = np.concatenate((np.real(s1_reshape),np.imag(s1_reshape)),axis=0)     #(8, 1)                                                            #np.sum(rate) /(P_t )             
#         return reward, new_state[:,0]            #return state,reward,  (32,)
   
