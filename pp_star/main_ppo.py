import numpy as np
from ppo_agent import Agent 
from  env_ppo import RIS_D, UEs_and_utilities
import matplotlib.pyplot as plt
#from tools import plotter_convergence
def db2lin(db):                                                                                                       ## decible to   linear 
    lin = np.power(10,(db/10))
    return lin 
#M_set= [5,10,15,20] # vs EE 333333333333333333333333    np.arange(2,12,2)[ 2,  4,  6,  8, 10]
M=8
L=8  # No of IRS      2,4,6,10,12........VS EE wrt p_maX22222222222222222
K =1  
J=1
#K_set= [ 2,4,6,8,10]       #   user ,,VS EE  444444444444444
N=4
#N_set= [ 8,16,32,64] #np.arange(2,12,2)yu      #  8 # ref ele Of IRS 2,4,6,8,10,12 VS EE  111111111111111111111

P_B= 39 # dbm 39
P_B= np.power(10,(P_B/10))/1000

P_K= 10  #dbm 10     10^(P_R/10)/10^3;
P_K= np.power(10,(P_K/10))/1000

P_J= 10  #dbm 10     10^(P_R/10)/10^3;
P_J= np.power(10,(P_J/10))/1000

P_R= 10  #dbm 10
P_R= np.power(10,(P_R/10))/1000
#P_R= db2lin(10)    #np.power(10,(P_R/10))/1000   #db2lin(9)

u= 1.25 #v=0.8
awgn_var = -104#110
awgn_var= np.power(10,(awgn_var/10))/100

Pmax= 30 #dbm    #10^(P_k/10)/10^3;   '/
power_properties = np.array([[0],[np.power(10,(Pmax/10))/1000]])    #np.array([[0],[pmax]]) pmax ko lin me change kiya hai /////(2, 1)
phase_properties = np.array([[0],[2*np.pi]])    ###############################(2, 1)

#hyperparameter
alpha=0.001   #1e-5  0.00001.
gamma =0.9  #.95  # Discount factor 0.9
batch_size = 64 #ty 128
max_size =100000  # buffer size
gae_lambda= 0.99  #0.95//0.9
clip= 0.2
epoch=10
nn_actor= np.array([[512],[512]])
nn_critic=np.array([[512],[512]])

# net = np.array([[128],[128]])    #nn_actor
# net  = np.array([[256],[256]])   #nn_critic

episodes = 1000
max_steps= 100
ep_reward_list = []  
avg_reward_list = []    
avg_score = np.zeros(episodes)

# reward_set=np.zeros((episodes,len(K_set)))
# score_pt = np.zeros((len(K_set))) 
# for cntr in range(len(K_set)):
#     K=K_set[cntr]
#============================================d=============================================================   
if __name__ == '__main__':
    
               
    UEs = UEs_and_utilities(M,N,K,L)
    
    env = RIS_D( M, N, K,L, P_B, P_K, P_J, P_R,u,awgn_var,power_properties,phase_properties, UEs )         
    agent = Agent(env, gamma, gae_lambda, batch_size, alpha, clip, epoch, nn_actor, nn_critic)  
        
    x = np.zeros((episodes,episodes))       
    y = np.zeros((episodes,episodes), dtype = complex)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    for i in range(episodes):            
        done = False
        observation = env.reset_state()             #state                                                
        for i0 in range(max_steps):
            action,logprob = agent.choose_action(observation)   #  , evaluate= False)##  array of float(36).....36 row                       actnjmion ##i modified evaliuate= False before this term was not here ##   from ddpg                              
            reward, next_state = env.step(action)                 #next_stATE                            
            agent.remember(observation, action,logprob ,reward)             ## store transition## i have added done in extra #score += reward  ##v  frpm ddpg                
            agent.learn()                                 ## tarnsition to new state   ###  set currenrt state to new state  
            observation = next_state                          ##  state = next_ state ## transition to new state                          
            ep_reward_list.append(reward)  
        avg_score[i]=np.mean(ep_reward_list[-100:])
     #   reward_set[i,cntr] = avg_score[i] 
        print('episode',i,'reward %.1f' % reward, 'avg score %.1f' % avg_score[i])
                  # reward_set[i,cntr] = avg_score[i]                                         
              #              print('episode',i,'reward %.1f' % reward, 'avg score %.1f' % avg_score[i])\                
              #              x[i]=i
              #              y[i]=avg_score
              # reward_set[i,cntr] = avg_score[i]     
            #  print('episode',i,'reward %.1f' % reward, 'avg score %.1f' % avg_score[i])  # (avg_score[i])\
#average_score_pt = np.mean(reward_set,axis=0) 

#for i in range(len(K_set)):
    
#plt.plot(N_set,average_score_pt,linewidth=1,label='P_max =10')
    # plt.plot(N_set,average_score_pt,linewidth=1,label='lr ='+str(N_set[i]))  #///correct for x axix N_Set [2,4,6,8,10]
    # plt.plot(np.arange(episodes),avg_score,linewidth=1,label='N = '+str(N_set[i]))  
    # plt. plot(N_set,score_pt,linewidth=1,label='Pt = '+str(N_set[i]))
    ################## plt.plot(np.arange(episodes),reward_set[:,i],linewidth=1,label='K ='+str(K_set[i]))                                           
plt.plot([i for i in range(episodes)],avg_score ,linewidth=1 )#, label='N= '+str(N_set[i] ) )           #,label='N =' ) 
    #plt.plot([i for i in range(episode)],avg_score ,linewidth=1, label="pt= 30 , learning_rate= 0.001, decay_rate= 0.00001")  
plt.legend(loc='lower right',fontsize = 5)
plt.xlabel("episodes")
plt.ylabel("EE")# score
plt.title("(PPO)  EE Vs Episiodes star-irs")
plt.grid(b=True, which='major')
plt.grid(b=True, which='minor',alpha=0.4)
plt.show () 







# ============================================================  best result on this =================
# import numpy as np
# from ppo_agent import Agent 
# from  env_ppo import RIS_D, UEs_and_utilities
# import matplotlib.pyplot as plt
# #from tools import plotter_convergence
# def db2lin(db):                                                                                                       ## decible to   linear 
#     lin = np.power(10,(db/10))
#     return lin 
# M= 8    #BS ant    5,10,15,20  vs EE 333333333333333333333333    np.arange(2,12,2)[ 2,  4,  6,  8, 10]
# L= 8    # No of IRS      2,4,6,10,12........VS EE wrt p_maX22222222222222222
# K= 1  # user    2,4,6,8,19,,VS EE  444444444444444
# N=4
# #N_set= [ 2,4,6,8,10,12] #np.arange(2,12,2)yu      #  8 # ref ele Of IRS 2,4,6,8,10,12 VS EE  111111111111111111111
# 
# P_B= 39 # dbm 39
# P_B= np.power(10,(P_B/10))/1000
# 
# P_K= 10  #dbm 10     10^(P_R/10)/10^3;
# P_K= np.power(10,(P_K/10))/1000
# 
# P_R= 10  #dbm 10
# P_R= np.power(10,(P_R/10))/1000
# #P_R= db2lin(10)    #np.power(10,(P_R/10))/1000   #db2lin(9)
# 
# u= 1.25 #v=0.8
# awgn_var = -104
# awgn_var= np.power(10,(awgn_var/10))/100
# 
# Pmax=50  #dbm    #10^(P_k/10)/10^3;   pt_set = [4,8,12] 
# power_properties = np.array([[0],[np.power(10,(Pmax/10))/1000]])    #np.array([[0],[pmax]]) pmax ko lin me change kiya hai /////(2, 1)
# phase_properties = np.array([[0],[2*np.pi]])    ###############################(2, 1)
# 
# #hyperparameter
# alpha=0.0001   #1e-5  0.00001.
# gamma =0.95  #.95  # Discount factor
# batch_size = 64 #ty
# max_size =100000  # buffer size
# gae_lambda= 0.9  #0.95//0.9
# clip= 0.2
# epoch=10
# nn_actor= np.array([[512],[512]])
# nn_critic=np.array([[512],[512]])
# 
# # net = np.array([[128],[128]])    #nn_actor
# # net  = np.array([[256],[256]])   #nn_critic
# 
# episodes = 500
# max_steps= 50
# ep_reward_list = []  
# avg_reward_list = []    
# avg_score = np.zeros(episodes)
# # reward_set=np.zeros((episodes,len(N_set)))
# # score_pt = np.zeros((len(N_set))) 
# #=========================================================================================================   
# if __name__ == '__main__':       
#     # for cntr in range(len(N_set)):
#     #     N=N_set[cntr]
#     UEs = UEs_and_utilities(M,N,K,L)
#     env = RIS_D( M, N, K,L, P_B, P_K, P_R,u,awgn_var,power_properties,phase_properties, UEs )         
#     agent = Agent(env, gamma, gae_lambda, batch_size, alpha, clip, epoch, nn_actor, nn_critic)     
#     x = np.zeros((episodes,episodes))       
#     y = np.zeros((episodes,episodes), dtype = complex)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
#     for i in range(episodes):            
#         done = False
#         observation = env.reset_state()             #state                                                
#         for i0 in range(max_steps):
#             action,logprob = agent.choose_action(observation)   #  , evaluate= False)##  array of float(36).....36 row                       actnjmion ##i modified evaliuate= False before this term was not here ##   from ddpg                              
#             reward, next_state = env.step(action)                 #next_stATE                            
#             agent.remember(observation, action,logprob ,reward)             ## store transition## i have added done in extra #score += reward  ##v  frpm ddpg                
#             agent.learn()                                 ## tarnsition to new state   ###  set currenrt state to new state  
#             observation = next_state                          ##  state = next_ state ## transition to new state                          
#             ep_reward_list.append(reward)  
#         avg_score[i]=np.mean(ep_reward_list[-100:])
#       #  reward_set[i,cntr] = avg_score[i] 
#         print('episode',i,'reward %.1f' % reward, 'avg score %.1f' % avg_score[i])
#                  # reward_set[i,cntr] = avg_score[i]                                         
#               #              print('episode',i,'reward %.1f' % reward, 'avg score %.1f' % avg_score[i])\                
#               #              x[i]=i
#               #              y[i]=avg_score
#               # reward_set[i,cntr] = avg_score[i]     
#             #  print('episode',i,'reward %.1f' % reward, 'avg score %.1f' % avg_score[i])  # (avg_score[i])\
# # average_score_pt = np.mean(reward_set,axis=0) 
# # for i in range(len(N_set)):
# #plt.plot(N_set,average_score_pt,linewidth=1,label='P_max =10')
#    # plt.plot(N_set,average_score_pt,linewidth=1,label='lr ='+str(N_set[i]))  #///correct for x axix N_Set [2,4,6,8,10]
#     # plt.plot(np.arange(episodes),avg_score,linewidth=1,label='N = '+str(N_set[i]))  
#     # plt. plot(N_set,score_pt,linewidth=1,label='Pt = '+str(N_set[i]))
#     # plt.plot(np.arange(episodes),reward_set[:,i],linewidth=1,label='lr ='+str(N_set[i]))                                           
# plt.plot([i for i in range(episodes)],avg_score ,linewidth=1, label="N= 4" )            #,label='N =' ) 
#     #plt.plot([i for i in range(episode)],avg_score ,linewidth=1, label="pt= 30 , learning_rate= 0.001, decay_rate= 0.00001")  
# plt.legend(loc='lower right',fontsize = 5)
# plt.xlabel("episodes")
# plt.ylabel("avg_score")# score
# plt.title("(PPO)  EE Vs Episiodes")
# plt.grid(b=True, which='major')
# plt.grid(b=True, which='minor',alpha=0.4)
# plt.show () 
# =============================================================================
   












# if __name__ == '__main__':
          
#         UEs = UEs_and_utilities(M,N,K,L)
                # env = RIS_D( M, N, K,L, P_B, P_K, P_R,u,awgn_var,power_properties,phase_properties, UEs )         
                   # agent = Agent(env, gamma, gae_lambda, batch_size, alpha, clip, epoch, nn_actor, nn_critic)                                                                                                       #pt = db2lin(pt)                                        
#         env = RIS_D( M, N, K,L, P_B, P_K, P_R,u,awgn_var,power_properties,phase_properties, UEs )         
#         agent = Agent(env, gamma, gae_lambda, batch_size, alpha, clip, epoch, nn_actor, nn_critic)                                                                                              #(input_dims , action_dim , net,env = env )                 
#         x = np.zeros((episodes,episodes))       
#         y = np.zeros((episodes,episodes), dtype = complex)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                 
#         for i in range(episodes):            
#               done = False
#               observation = env.reset_state()             #state                                                
#               for i0 in range(max_steps):
#                   action,logprob = agent.choose_action(observation)   #  , evaluate= False)##  array of float(36).....36 row                       actnjmion ##i modified evaliuate= False before this term was not here ##   from ddpg                                    
#                   reward, next_state = env.step(action)                 #next_stATE                             
#                   agent.remember(observation, action,logprob ,reward)             ## store transition## i have added done in extra #score += reward  ##v  frpm ddpg                
#                   agent.learn()                                 ## tarnsition to new state   ###  set currenrt state to new state                   
#                   observation = next_state                          ##  state = next_ state ## transition to new state  
#                 # ppo_reward_list[i0,i] = reward
#                   ppo_reward_list.append(reward)  
#               avg_score[i]=np.mean(ppo_reward_list[-100:])
              
#                 # reward_set[i,cntr] = avg_score[i]     
#               print('episode',i,'reward %.1f' % reward, 'avg score %.1f' % avg_score[i])  # (avg_score[i])\ 
                 
#               # np.append(ppo_reward_temp[i0,i],(reward))  
#               # avg_score=np.mean(ppo_reward_temp[i0,:])  
          
       
# # for i in range(len(pt_set)):
                                           
# plt.plot([i for i in range(episodes)],avg_score ,linewidth=1, label="M=8, L=8, K=1,N=6, Pmax= 40" )            #,label='N =' ) 
# #plt.plot([i for i in range(episode)],avg_score ,linewidth=1, label="pt= 30 , learning_rate= 0.001, decay_rate= 0.00001")  
# plt.legend(loc='lower right',fontsize = 7)
# plt.xlabel("episode")
# plt.ylabel("EE(PPO)")# score
# plt.grid(b=True, which='major')
# plt.grid(b=True, which='minor',alpha=0.4)
# plt.show () 


     
'''import numpy as np
from ppo_agent import Agent 
from  env_ppo import RIS_D, UEs_and_utilities
import matplotlib.pyplot as plt
from tools_ppo import plotter_convergence
def db2lin(db):                                                                                                       ## decible to   linear 
    lin = np.power(10,(db/10))
    return lin 
M= 8    #BS ant    5,10,15,20  vs EE 333333333333333333333333    np.arange(2,12,2)[ 2,  4,  6,  8, 10]
L= 8    # No of IRS      2,4,6,10,12........VS EE wrt p_maX22222222222222222
K= 1  # user    2,4,6,8,19,,VS EE  444444444444444
N=8
N_set= np.arange(2,12,2)      #  8 # ref ele Of IRS 2,4,6,8,10,12 VS EE  111111111111111111111


P_B= 39 # dbm 39
P_B= np.power(10,(P_B/10))/1000

P_K= 10  #dbm 10     10^(P_R/10)/10^3;
P_K= np.power(10,(P_K/10))/1000

#P_R= 10  #dbm 10
P_R= db2lin(10)    #np.power(10,(P_R/10))/1000   #db2lin(9)

u= 1.25 #v=0.8
awgn_var = -104
awgn_var= np.power(10,(awgn_var/10))/100


Pmax= 10  #dbm    #10^(P_k/10)/10^3;   pt_set = [4,8,12] 
power_properties = np.array([[0],[np.power(10,(Pmax/10))/1000]])    #np.array([[0],[pmax]]) pmax ko lin me change kiya hai /////(2, 1)
phase_properties = np.array([[0],[2*np.pi]])    ###############################(2, 1)


# bs_loc = np.array((0,0,0),dtype=np.float32)
# irs_loc = np.array((100,0,50),dtype=np.float32)
# user_loc = np.array([[np.random.uniform(-100,100)],[np.random.uniform(-100,100)]])

net = np.array([[512],[512]]) #512
alpha=0.001   #1e-5 
beta=0.001
gamma =0.99  #.95  # Discount factor
tau =0.0001
batch_size = 64 #ty
max_size =100000  # buffer size

sampling=2
gae_lambda= 0.99  #0.95//0.9
clip= 0.2
epoch=10
nn_actor= np.array([[512],[512]])
nn_critic=np.array([[512],[512]])

# net = np.array([[128],[128]])    #nn_actor
# net  = np.array([[256],[256]])   #nn_critic

episodes = 5
max_steps= 5
ppo_reward_list = []  
avg_reward_list = []    
avg_score = np.zeros(episodes)

ppo_reward_temp = np.zeros((episodes,max_steps))
ppo_plotter     = plotter_convergence(episodes,sampling,N_set)
ppo_reward_step = np.zeros((max_steps,len(N_set),sampling))
#ppo_reward_temp = np.zeros((episodes,max_steps))
# reward_set = np.zeros((episodes,len(P_max_set)))                             
# score_pt = np.zeros((len(Pmax_set)))    



for cntr in range(len(N_set)):
    N=N_set[cntr]
      
    UEs = UEs_and_utilities(M,N,K,L) 
    for sample in range(sampling):                                                                                              #pt = db2lin(pt)                                        
        env = RIS_D( M, N, K,L, P_B, P_K, P_R,u,awgn_var,power_properties,phase_properties, UEs )         
        agent = Agent(env, gamma, gae_lambda, batch_size, alpha, clip, epoch, nn_actor, nn_critic)                                                                                              #(input_dims , action_dim , net,env = env )                 
        x = np.zeros((episodes,episodes))       
        y = np.zeros((episodes,episodes), dtype = complex)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                 
        for i0 in range(episodes):            
              done = False
              observation = env.reset_state()             #state                                                
              for i in range(max_steps):
                  action,logprob = agent.choose_action(observation)   #  , evaluate= False)##  array of float(36).....36 row                       actnjmion ##i modified evaliuate= False before this term was not here ##   from ddpg                                    
                  reward, next_state = env.step(action)                 #next_stATE                             
                  agent.remember(observation, action,logprob ,reward)             ## store transition## i have added done in extra #score += reward  ##v  frpm ddpg                
                  agent.learn()                                 ## tarnsition to new state   ###  set currenrt state to new state                   
                  ppo_reward_temp[i0,i] = reward
                  observation = next_state                          ##  state = next_ state ## transition to new state 
                  if i0 == 0:
                       ppo_reward_step [i,cntr,sample] = reward 
               
               # print('step',i,'reward = ', reward)
              avg_score = np.mean(ppo_reward_temp[i0,:])
              ppo_plotter.record(avg_score,i0,sample,cntr)
              print('N =',N,'Sample',sample+1,'Episode',i0+1, 'reward %.1f' % (reward), 'avg score %.1f' % avg_score)
        print('Sampling ',str(sample+1),' Done!!!--- \n')

ppo_plotter.plot(grid=episodes,title="PPO", ax="Episodes")
ppo_plotter.plot_result(title ="(PPO) Received SNR Vs BS-USer Horizontal Distance")
ppo_plot_step = np.mean(ppo_reward_step,axis=2)

#PPO plot step
for i0 in range(len(N_set)):
    plt.plot(np.arange(episodes), ppo_plot_step[:,i0], linewidth=1, label="N")
    plt.legend(loc='lower right', fontsize=10)
    plt.minorticks_on()
    plt.title("PPO Convergence")
    plt.xlabel("episodes")
    plt.ylabel("EE")
    plt.grid(b=True, which='major')
    plt.grid(b=True, which='minor',alpha=0.4)
plt.show()'''

              
              
              
#             # ppo_reward_list[i0,i] = reward
#               ppo_reward_list.append(reward)  
#         avg_score=np.mean(ppo_reward_list[-100:])
        
#           # reward_set[i,cntr] = avg_score[i]     
#         print('episode',i,'reward %.1f' % reward, 'avg score %.1f' % avg_score)  # (avg_score[i])\ 
                 
#               # np.append(ppo_reward_temp[i0,i],(reward))  
#               # avg_score=np.mean(ppo_reward_temp[i0,:])  
          
       
# # for i in range(len(pt_set)):
                                           
# plt.plot([i for i in range(episodes)],avg_score ,linewidth=1, label="M=8, L=8, K=1,N=6, Pmax= 40" )            #,label='N =' ) 
# #plt.plot([i for i in range(episode)],avg_score ,linewidth=1, label="pt= 30 , learning_rate= 0.001, decay_rate= 0.00001")  
# plt.legend(loc='lower right',fontsize = 7)
# plt.xlabel("episode")
# plt.ylabel("EE(PPO)")# score
# plt.grid(b=True, which='major')
# plt.grid(b=True, which='minor',alpha=0.4)
# plt.show () 








# import numpy as np
# from ppo_agent import Agent 
# from  env_ppo import RIS_D, UEs_and_utilities
# import matplotlib.pyplot as plt
# #from tools import plotter_convergence
# def db2lin(db):                                                                                                       ## decible to   linear 
#     lin = np.power(10,(db/10))
#     return lin 
# M= 8    #BS ant    5,10,15,20  vs EE 333333333333333333333333    np.arange(2,12,2)[ 2,  4,  6,  8, 10]
# L= 8    # No of IRS      2,4,6,10,12........VS EE wrt p_maX22222222222222222
# K= 1  # user    2,4,6,8,19,,VS EE  444444444444444
# N=8
# N_set=  np.arange(2,12,2)      #  8 # ref ele Of IRS 2,4,6,8,10,12 VS EE  111111111111111111111


# P_B= 39 # dbm 39
# P_B= np.power(10,(P_B/10))/1000

# P_K= 10  #dbm 10     10^(P_R/10)/10^3;
# P_K= np.power(10,(P_K/10))/1000

# #P_R= 10  #dbm 10
# P_R= db2lin(10)    #np.power(10,(P_R/10))/1000   #db2lin(9)

# u= 1.25 #v=0.8
# awgn_var = -104
# awgn_var= np.power(10,(awgn_var/10))/100


# Pmax= 10  #dbm    #10^(P_k/10)/10^3;   pt_set = [4,8,12] 
# power_properties = np.array([[0],[np.power(10,(Pmax/10))/1000]])    #np.array([[0],[pmax]]) pmax ko lin me change kiya hai /////(2, 1)
# phase_properties = np.array([[0],[2*np.pi]])    ###############################(2, 1)


# # bs_loc = np.array((0,0,0),dtype=np.float32)
# # irs_loc = np.array((100,0,50),dtype=np.float32)
# # user_loc = np.array([[np.random.uniform(-100,100)],[np.random.uniform(-100,100)]])

# net = np.array([[512],[512]]) #512
# alpha=0.001   #1e-5 
# beta=0.001
# gamma =0.99  #.95  # Discount factor
# tau =0.0001
# batch_size = 64 #ty
# max_size =100000  # buffer size

# sampling=2
# gae_lambda= 0.99  #0.95//0.9
# clip= 0.2
# epoch=10
# nn_actor= np.array([[512],[512]])
# nn_critic=np.array([[512],[512]])

# # net = np.array([[128],[128]])    #nn_actor
# # net  = np.array([[256],[256]])   #nn_critic

# episodes = 50
# max_steps= 5
# ppo_reward_list = []  
# avg_reward_list = []    
# avg_score = np.zeros(episodes)


# #ppo_reward_temp = np.zeros((episodes,max_steps))
# # reward_set = np.zeros((episodes,len(P_max_set)))                             
# # score_pt = np.zeros((len(Pmax_set)))    

# if __name__ == '__main__':          
#         UEs = UEs_and_utilities(M,N,K,L)                                                                                                     #pt = db2lin(pt)                                        
#         env = RIS_D( M, N, K,L, P_B, P_K, P_R,u,awgn_var,power_properties,phase_properties, UEs )         
#         agent = Agent(env, gamma, gae_lambda, batch_size, alpha, clip, epoch, nn_actor, nn_critic)                                                                                              #(input_dims , action_dim , net,env = env )                 
#         x = np.zeros((episodes,episodes))       
#         y = np.zeros((episodes,episodes), dtype = complex)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
                 
#         for i in range(episodes):            
#               done = False
#               observation = env.reset_state()             #state                                                
#               for i0 in range(max_steps):
#                  action,logprob = agent.choose_action(observation)   #  , evaluate= False)##  array of float(36).....36 row                       actnjmion ##i modified evaliuate= False before this term was not here ##   from ddpg                                    
#                  reward, next_state = env.step(action)                 #next_stATE                             
#                  agent.remember(observation, action,logprob ,reward)             ## store transition## i have added done in extra #score += reward  ##v  frpm ddpg                
#                  agent.learn()                                 ## tarnsition to new state   ###  set currenrt state to new state                   
#                  observation = next_state                          ##  state = next_ state ## transition to new state  
#                 # ppo_reward_list[i0,i] = reward
#                  ppo_reward_list.append(reward)  
#               avg_score=np.mean(ppo_reward_list[-100:])
              
#                # reward_set[i,cntr] = avg_score[i]     
#               print('episode',i,'reward %.1f' % reward, 'avg score %.1f' % avg_score)  # (avg_score[i])\ 
                 
#              # np.append(ppo_reward_temp[i0,i],(reward))  
#              # avg_score=np.mean(ppo_reward_temp[i0,:])  
          
       
# # for i in range(len(pt_set)):
                                           
# plt.plot([i for i in range(episodes)],avg_score ,linewidth=1, label="M=8, L=8, K=1,N=6, Pmax= 40" )            #,label='N =' ) 
# #plt.plot([i for i in range(episode)],avg_score ,linewidth=1, label="pt= 30 , learning_rate= 0.001, decay_rate= 0.00001")  
# plt.legend(loc='lower right',fontsize = 7)
# plt.xlabel("episode")
# plt.ylabel("EE(PPO)")# score
# plt.grid(b=True, which='major')
# plt.grid(b=True, which='minor',alpha=0.4)
# plt.show () 


 





#  ep_reward_list.append(reward)  
# avg_score[i]=np.mean(ep_reward_list[-100:])  








'''import numpy as np
from ppo_agent import Agent
from env_ppo  import RIS_D, UEs_and_utilities
import matplotlib.pyplot as plt
from tools_ppo import plotter_convergence   #, fungsi as fn
# import os
def db2lin(db):                                                                                                       
    lin = np.power(10,(db/10))
    return lin 
M= 8    #BS ant    5,10,15,20  vs EE 333333333333333333333333
L= 8    # No of IRS      2,4,6,10,12........VS EE wrt p_maX22222222222222222
K= 1  # user    2,4,6,8,19,,VS EE  444444444444444
N= 8 # ref ele Of IRS 2,4,6,8,10,12 VS EE  111111111111111111111
N_set=[2,6,8,10,12]     #////x_axis
parameter = N_set

P_B= 39 # dbm 39
P_B= np.power(10,(P_B/10))/1000

P_K= 10  #dbm 10     10^(P_R/10)/10^3;
P_K= np.power(10,(P_K/10))/1000

P_R= 10  #dbm 10
P_R= np.power(10,(P_R/10))/1000

u= 1.25 #v=0.8
awgn_var = -104
awgn_var= np.power(10,(awgn_var/10))/100

Pmax= 10  #dbm    #10^(P_k/10)/10^3;   pt_set = [4,8,12] 
power_properties = np.array([[0],[np.power(10,(Pmax/10))/1000]])    #np.array([[0],[pmax]]) pmax ko lin me change kiya hai /////(2, 1)
phase_properties = np.array([[0],[2*np.pi]])    ###############################(2, 1)

net = np.array([[512],[512]]) #512
alpha=0.001 
beta=0.001
gamma =0.99  #.95
tau =0.0001
batch_size = 64 #ty
max_size =100000  # buffer size

sampling=2
gamma =0.99  #.95    # Discount factor
gae_lambda= 0.99  #0.95//0.9
batch_size = 64   #16///32
alpha= 1e-5 
clip= 0.2
epoch=10
max_size =100000  # buffer size
nn_actor= np.array([[512],[512]])
nn_critic=np.array([[512],[512]])

episodes = 10
max_steps= 2  # The number of step in each episode   

ppo_reward_temp = np.zeros((episodes,max_steps))
ppo_plotter = plotter_convergence(episodes,sampling,parameter)

for cntr in range(len(parameter)):
    N = parameter[cntr]
    info = N
    UEs = UEs_and_utilities(M,N,K,L)

    for sample_ in range(sampling):                                                                                                                #pt = db2lin(pt)                                        
        env = RIS_D( M, N, K,L, P_B, P_K, P_R,u,awgn_var,power_properties,phase_properties,UEs )         
        agent = Agent(env, gamma, gae_lambda, batch_size, alpha, clip, epoch, nn_actor, nn_critic) 
       # state = agent.env.reset_state()         
             
        for i in range(episodes):            
              state = agent.env.reset_state()             #state 
                                               
              for i0 in range(max_steps):
                action,logprob = agent.choose_action(state)                                            
                reward, next_state = env.step(action)                 #next_stATE                             
                agent.remember(state, action, logprob, reward)             ## store transition## i have added done in extra #score += reward  ##v  frpm ddpg                
                agent.learn()                                 ## tarnsition to new state   ###  set currenrt state to new state  
                
              #  ppo_reward_temp[i0,i] = reward
                observation = next_state                          ##  state = next_ state ## transition to new state  
                              
                #ep_reward_list.append(reward)
              avg_score = np.mean(ppo_reward_temp[i0,:])
              ppo_plotter.record(avg_score,i0,sample_,cntr)
              print('N =',info,'Sample',sample_+1,'Episode',i0+1, 'reward %.1f' % (reward), 'avg score %.1f' % avg_score)
        print('\n MOVING UAV Scenario\n',' Sampling ',str(sample_+1),'MOVING UAV Scenario Done!!!--- \n')
ppo_plotter.plot(grid=episodes,title="PPO Single-Agent",legend = "M = ")  

ppo_plotter.plot_result(title ="(PPO) Energy Efficiency", ax="Number of UEs on Each// Cluster")'''



