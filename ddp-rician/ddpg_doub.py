import tensorflow as tf
import tensorflow. keras as keras
from tensorflow. keras. optimizers import Adam   ## keras is inbuilt in TF(nice rapper)   
from replay_buff_doub import ReplayBuffer
from network_doub import ActorNetwork, CriticNetwork
class Agent:
    def __init__(self, net, env=None ,alpha=0.001, beta=0.001, gamma =0.99, max_size =100000,
                 tau =0.0001, batch_size =64, noise = 0.1) :   #tau=0.001
        self.env = env
        #tarining parameter
        self.gamma = gamma         #gamma is the discount factor   for future reward
        self.tau = tau              # decaying  rate
        
        self.n_actions = env.action_dim   #############
       # self.input_dims= env.state_dim      ## state_dim
        
        self.memory = ReplayBuffer(max_size,env.state_dim, self.n_actions)          ##state k apace pe ..input_dims tha maine  state kiya hai
        self.batch_size = batch_size        
        self.noise = noise
        
        self.max_action = 1  #env.max_action
        self.min_action = 0    #env.min_action
        
        #Initialize actor//critic  networks and=== optimizer
        self.actor = ActorNetwork(net,self.n_actions, name ='actor')               ################################ env.L   8 feb
        self.critic = CriticNetwork(net, name='critic')         
        self.target_actor = ActorNetwork(net,self.n_actions,name='target_actor')
        self.target_critic = CriticNetwork(net,name='target_critic')
        
        self.actor.compile(optimizer=Adam( learning_rate=alpha)) # alpha is the actor_learning_rate
        self.critic.compile(optimizer=Adam( learning_rate=beta)) #beta is the critic_learning_rate        
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta)) 
        
        self.update_network_parameters (tau=1) 
        
    def update_network_parameters(self, tau=None):
        if tau is None:
           tau =self.tau          
                                    
        weights = []   
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):  
            weights.append(weight*tau+targets[i]*(1-tau))
        self.target_actor.set_weights(weights)  #
        weights =[]                         
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight*tau+targets[i]*(1-tau))
        self.target_critic.set_weights(weights)   
             
    def remember(self, state, action, reward, new_state):
        self.memory.store_transition(state, action, reward, new_state) 
                                                           #_transition() missing 1 required positional argument: 'done'# I removed done from here     
    def choose_action(self, observation, evaluate=False):  #observation is state####################################################################
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)                           ###########    # if not evaluate:                                                                       
        actions += tf.random.normal(shape =[self.n_actions],mean =0.0, stddev=self. noise)
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        return actions[0].numpy()
        
    # def learn(self):        
    #     if self.memory.mem_cntr < self.batch_size:
    #         return
    #     state, action, reward, new_state = self.memory.sample_buffer(self.batch_size) ## sample agwnts  memory## from  replay buffer
    #     states = tf.convert_to_tensor(state, dtype=tf.float32)  ## handle all therse with tf 
    #     states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)                                
    #     actions = tf.convert_to_tensor(action, dtype=tf.float32)
    #     reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        
        
    #     with tf.GradientTape() as tape:                        ###      calculation of our gradient        #######    /////critic loss   ??? MSE
    #         # loc_mu,phi_mu = self.actor(states)  ################%%%%%%%%%%%%%%%^^^^^^^^^^^^^^^^   target_actions me changes krna hai 
    #         target_actions = self.target_actor(states_)
    #         critic_value_ = tf.squeeze(self.target_critic(states_, target_actions), 1)         #critc value for new  state and //pass it in new taget action
    #         critic_value = tf.squeeze(self. critic(states, actions ), 1)             ## critc value for current state and //actions the sgent actually took
            
    #         target = reward+self.gamma*critic_value_  # //calculate our actual target value //## 1- done   means if,  true then 1-1 = 0  
    #         critic_loss = keras.losses.MSE(target, critic_value)  ##   ///////   Critic_loss  loss    ///////////    between the target and critic vaue   
    #     critic_network_gradient = tape.gradient(critic_loss,self.critic.trainable_variables)         
    #     self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))  ## ///////critic loss 
        
    #     with tf.GradientTape() as tape:   #                 ///    Actor Loss     ///////////  Backpropagation
    #        # loc_mu,phi_mu = self.actor(states)   #####################%%%%%%%%%%%%%%%%%%%%%%%%########### new_policy actions me changes krne hai
    #         new_policy_actions = self.actor(states)
    #         actor_loss = -self. critic(states, new_policy_actions)## state is choosen acording to actor network ### ////////backpropagation
    #         actor_loss = tf.math.reduce_mean(actor_loss)
    #     actor_network_gradient = tape.gradient(actor_loss,self.actor.trainable_variables)
    #     self. actor. optimizer.apply_gradients (zip( actor_network_gradient, self.actor.trainable_variables))        
    #     self.update_network_parameters()        #/// update our network parameter    ///## tau  ///////    soft update 
                     
        
        
        
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        # target_actions = self.target_actor(states_)
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_) #mu & sigma

            # target_actions = tf.concat((loc_mu,phi_mu),axis=1)
            critic_value_ = tf.squeeze(self.target_critic(states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1) #Q
            target = rewards + self.gamma*critic_value_ #r+gama*Q_next
            critic_loss = keras.losses.MSE(target, critic_value)
        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            # new_policy_actions = tf.concat((loc_mu,phi_mu),axis=1)
            actor_loss = -self.critic(states, new_policy_actions)
            actor_loss = tf.math.reduce_mean(actor_loss)
        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))
        
        self.update_network_parameters()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
''' So, initially, our agent starts with random actions and we keep feeding it in memory
        until it reaches the batch size. Then from this point, the agent takes 
        all the (states, actions, rewards, termination, new state) info from the replay buffer
        and learns on this. This is called an off-policy algorithm, when your agent learns based on past experiences
        and not from the current actions it is taking. An on-policy algorithm means it will learn on the fly, 
        it will make a batch, learn from it and then dump that batch. So, we create our replay buffer –'''
        
        
        
        
'''def reset_state(self):                                                                                             # env needs to reset its state
              
              theta = np.random.rand(self.N,self.L)*2*np.pi          #(4,4) 
              
              x = np.zeros(self.L, dtype=int)                                                               # int part I added 10 jan
              for i in range(self.L):     #initiate xl
                  x[i]= np.random.randint(0,2)
              
              phi = np.zeros((self.N,self.N,self.L) ,dtype=complex)         
              for i in range(self.L):
                  phi[:,:,i] = np.diag(np.exp(1j*theta[:,i]))                       

              s = np.zeros((self.M,self.K), dtype= np.complex)    ##2 dec///////////////
                                              ##  it will update at every episode 
              x = np.zeros(self.L, dtype=int)
              for i in range(self.L):     #initiate xl
                  x[i]= np.random.randint(0,2)  #@@@@    (100,)
                  
              for i in range(self.K): 
                  s1= self.g[:,:,i].conj().T  #hkl_H   wk 
                  for i0 in  range(self.L):
                      s1 += np.matmul(np.matmul((x[i0]*self.h[:,i,i0].conj().T),phi[:,:,i0]),self.G[:,:,i0])
                  s[:,i] = s1          
              s_reshape = s.reshape(-1,1)                 #  only column me 
              state = np.concatenate((np.real(s_reshape),np.imag(s_reshape)), axis=0)   
              return state[:,0] 

        def step(self, action):          
            x = np.zeros(self.L)                     # x
            theta_w = np.zeros((self.M*self.K))      # w
            theta_irs = np.zeros(self.N*self.L)       # theta (irs)
                        
            for i in range(len(action)):
                if i < self.L:
                    x[i] = int(np.around(action[i]))                
                elif i < (self.M*self.K)+self.L:                
                    theta_w[i-self.L] = action[i]
                else:                                
                    theta_irs[i-(self.M*self.K) -self.L] = action[i]
              
            W = np.zeros((self.M,self.K),dtype = np.complex)    #/// ## ////shape(4,4,)//    sizze  16## GGGG//////////////
            cntr = 0  
            for i in range(self.M):
                for i0 in range(self.K):                                
                    W[i,i0] = np.exp(1j*theta_w[cntr]) # ///givinfg a complex valeu (1+2j)//
                    cntr += 1  
                       
              
               phi = theta_irs.reshape((self.N,self.L))
              
                for i in range(self.K): 
                    s1= self.g[:,:,i].conj().T  #hkl_H   wk 
                    for i0 in  range(self.L):
                        s1 += np.matmul(np.matmul((x[i0]*self.h[:,i,i0].conj().T),phi[:,:,i0]),self.G[:,:,i0])
                    s[:,i] = s1
                    num = np.matmul(s1,W[:,i])
                    num = np.power(np.abs(num),2)
                    
                    for i1 in range(self.K):
                        if i1 != i:
                            s2 = self.g[:,:,i1].conj().T
                            for i0 in range(self.L):
                                s2 += np.matmul(np.matmul((x[i0]*self.h[:,i1,i0].conj().T),phi[:,:,i0]),self.G[:,:,i0])
                            denum = np.matmul(s2,W[:,i1])
                            denum = np.power(np.abs(denum),2)
                            denum += self.awgn_var
                            
                    sinr[i] = num/denum
                    rate[i] = np.log2(1+siinr[i])
                    
                    
                    #power: 
                     P_t=0 
                     
                     P_1 = 0
                     for K00 in range(self.K):
                         w_k= self.w[K00]
                         w_kh= self.w[K00].conj().T
                         P_1 += np.matmul (np.matmul(w_k, w_kh),self.u)   #Pmax= w.w_H
                                  
                     P_4 = 0     
                     for L0 in range (self.L) :                     
                      # N[L0]= np.zeros(self.L)===================           
                       P_4 += np.matmul(np.matmul(x[L0],self.N) , self.P_R)    # Nl
                                 
                     P_t += P_1 + self.P_B + self.P_K + P_4
                       
                reward = sum(rate)/P_t'''
                