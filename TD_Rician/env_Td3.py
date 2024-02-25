import numpy as np
import os 
class UEs_and_utilities():                   
    def __init__(self,M,N,K,J,L):
        self.K = K
        self.J=J
        self.L =L
        self.N=N
        self.M=M
        self.bs_loc = np.array([[0],[0]])   ##(0,0,0)
        self.generate_irs_loc()                 ##(100,0,50)
        self.generete_user_loc1()                ##random( 54,32)
        self.generete_user_loc2()
        self.nlos= self.get_NLOS_array()
        self.U1_los, self.U2_los , self.bs_ris= self.get_random_channel()
      
    def generate_irs_loc(self):
        irs_loc = np.zeros((2,self.L),dtype = float)
        for i in range(self.L):
            irs_loc[:,i] = np.array([[np.cos(2*i*np.pi/self.L)*100],[np.sin(2*i*np.pi/self.L)*100]])[0]   #(3, 1)  #sirf ro k liye 
        self.irs_loc = irs_loc
       # return irs_loc
        
    def generete_user_loc1(self):    #rand(Num_User,2)*lengA;
        user_loc1 = np.zeros((2,self.K))
        for i in range(self.K):
            user_loc1[:,i] = np.array([[np.random.uniform(-100,100)],[np.random.uniform(-100,100)]])[0]     #random( 54,32)
        self.user_loc1 = user_loc1   
    def generete_user_loc2(self):   #rand(Num_User,2)*lengA;
        user_loc2 = np.zeros((2,self.J))
        for j in range(self.J):
            user_loc2[:,j] = np.array([[np.random.uniform(-100,100)],[np.random.uniform(-100,100)]])[0]     #random( 54,32)
        self.user_loc2 = user_loc2

    def get_NLOS_array(self):##############################################################################################################
        nlos = np.zeros((self.L),dtype=np.complex)
        for i in range(self.L):            
                nlos[i] = np.exp(-1j*2*np.pi*i*0.5*np.sin(np.random.rand(1)*2*np.pi))   ## d/lambda=1/2=0.5///      np.random.randint(0,2*np.pi)
        return nlos 
    
    def get_random_channel(self):
        U1_los = np.random.randn(self.N,self.K,self.L)+(1j*(np.random.randn(self.N,self.K,self.L)))
        U2_los = (np.random.randn(self.N,self.J,self.L)+(1j*(np.random.randn(self.N,self.J,self.L))))
        bs_ris = (np.random.randn(self.N,self.M,self.L)+(1j*(np.random.randn(self.N,self.M,self.L))))
            
        return U1_los, U2_los, bs_ris #with x and y axist 

class RIS_D(object):               
    def __init__(self, M, N, K,J,L, P_B, P_K,P_J ,P_R,K0, u,awgn_var,channel_noise_var,power_properties,phase_properties, UEs):        
        self.M = M        # BS station ante
        self.N = N        # IRS element N_l
        self.K = K         # user  
        self.J = J
        self.L= L           # no of irs       
# power
        self.u= u       #1/v = .`W11/0.8
        self.P_B= P_B
        self.P_K= P_K
        self.P_J= P_J
        self.P_R= P_R         #x.N.P_r
        self.K0=K0
        self.awgn_var = awgn_var
        self.channel_noise_var=channel_noise_var
        self.u=u
        
        self.power_properties = power_properties   #
        self.phase_properties = phase_properties
        
        self.P_max = np.max(power_properties)     #100Watt   Power_normal   // P_max
        self.Phase_max= np.max(phase_properties)     # phase normali
      # self.x=x              #x_l  [0,1]        
  # action
        self.theta = np.random.rand(self.N, self.L)*2*np.pi                           #np.random.rand(self.N)*2*np.pi                      
        self.Phi = np.zeros(self.N, dtype=complex)                                #np.zeros(self.M,self.K   dtype= complex) 
        self.w = np.zeros((self.M, self.K), dtype= complex) #BF vector for user K
        self.x = np.zeros(self.L)              
        self.UEs = UEs
                
        # self.h_nlos = np.random.randn(self.N)+(1j*(np.random.randn(self.N)))     #IRS_USER  ##  # fix # NLOS
        # self.h_random = np.random.randn(self.K)+(1j*(np.random.randn(self.K)))  # BS_USer      #   fix   ### NLOS ###########################**************
        #self.squire_array= self.get_squire_array()
        self.d_1, self.d_2 = self.get_dis() 
        self.d_3, self.d_4 = self.get_dis_IUS()     #IRS_US
        
        self.nlos = UEs.nlos
        self.U1_los = UEs.U1_los
        self.U2_los = UEs.U2_los
        self.bs_ris = UEs.bs_ris
        
        # self.irs_loc = UEs.irs_loc
        # self.bs_loc = UEs.bs_loc
        # self.user_loc1 = UEs.user_loc1
        
        self.h1 = self.get_h_channel()
        self.h2 = self.get_h_channel()
        self.F= self.get_channel()
        self.G= self.get_channel()
        
        self.D= self._compute_D()
      
#        self.state = self.reset_state()
    
        self.action_space = self.M*self.K*self.J + (self.N*self.L)+(self.L)+1         # theta(N) + w(M,1) + X((0,1 ))/theta(matrix) + W(vector) + x(vector)      
        self.observation_space =   2*(self.M*self.K *self.J)   #+2*(self.M*self.J )
 
#        self.min_action, self.max_action = self.get_bound_action() 
    def get_squire_array(self):  #h_LOS   # irs ref ele
        squire_array = np.zeros(self.N,dtype=np.complex)  # irs   #(4,)
        for i in range(self.N):
            squire_array[i] = np.exp(-1j*2*np.pi*i*0.5*np.cos(np.random.rand(1)*2*np.pi))   # complex array (0j) ==  (1,)            
        return squire_array        
#distance
    def get_dis(self):       ## BS_IRS   # BS_USer_1
        # irs_loc = self.UEs.irs_loc
        d_1 = np.zeros((self.L),dtype=np.float32)               #BS_IRS   #(L,N,M)  .....L ka for loop
        d_2 = np.zeros((self.K),dtype=np.float32)               #BS_US_1 (1,M)   ...no for loop        
        # d_3 = np.zeros((self.N,self.M),dtype=np.float32)       #IRS_US
        for i in range(self.L):
            d_1[i]= np.sqrt((self.UEs.irs_loc[0,i]-self.UEs.bs_loc[0])**2 + 
                                  (self.UEs.irs_loc[1,i]-self.UEs.bs_loc[1])**2 )   # + (self.UEs.irs_loc[2,i]-self.UEs.bs_loc[2])**2)
        for i0 in range(self.K):                
            d_2[i0] = np.sqrt((self.UEs.bs_loc[0]-self.UEs.user_loc1[0,i0])**2 + (self.UEs.bs_loc[1]-self.UEs.user_loc1[1,i0])**2 )# + (self.UEs.bs_loc[2]-self.UEs.user_loc[2,i0])**2)                
        return d_1, d_2    #122,104
           
    def get_dis_IUS(self):    #   IRS_ US_1,      US_2
        d_3 = np.zeros((self.L,self.K),dtype=np.float32)   #Nk jagah pe--K hoga   KLN...KL ka for loop 
        d_4 = np.zeros((self.L,self.J),dtype=np.float32)   #Nk jagah pe--K hoga   KLN...KL ka for loop
        for i in range(self.L):
            for i0 in range(self.K):
                d_3[i,i0] = np.sqrt((self.UEs.irs_loc[0,i]-self.UEs.user_loc1[0,i0])**2 + 
                                        (self.UEs.irs_loc[1,i]-self.UEs.user_loc1[1,i0])**2 )  #+ (self.UEs.irs_loc[2,i]-self.UEs.user_loc[2,i0])**2)  
        for j in range(self.L):
            for j0 in range(self.J):
                d_4[j,j0] = np.sqrt((self.UEs.irs_loc[0,j]-self.UEs.user_loc2[0,j0])**2 + (self.UEs.irs_loc[1,j]-self.UEs.user_loc2[1,j0])**2 )                        
        return d_3 ,d_4   #235
    
#/////// chanel 
# BS_IRS //  BS_USER_1  //// RANOM_NLOS
    def get_channel(self):  # BS_IRS  //  BS_USER     
        G = np.zeros((self.N,self.M,self.L),dtype=np.complex)   # BS_IRS   HHHHHHHHHHHH
        F = np.zeros((self.M,self.K),dtype=np.complex)    #BS_USER  K   hhhhhhhhhhhhh
        
        for i in range(self.L):  #L=8     (4,2)  2D    bs_ris 
            G[:,:,i] = np.power(10,-3.53)/(np.power(self.d_1[i],0.9))*((np.sqrt(self.K0/(1+self.K0))*self.nlos[i]) +(np.sqrt(1/(2))*self.bs_ris[:,:,i]))
                                                                                      
            
        for i0 in range(self.K):  ##///// NO--NLOS////// RANOM_NLOS
            F[:,i0] =  (np.power(10,-3.53)/(np.power(self.d_2[i0], 3.76)))*np.sqrt(0.5)*((np.random.randn(self.M) + 1j*(np.random.randn(self.M)))) #K /(8,)               
        return G, F  
    
# IRS_USER_1, IRS_US_2   ##+++  \\\\NLOS                                               
    def get_h_channel(self):
        h1 = np.zeros((self.N,self.K,self.L),dtype=complex)  ###gggggggggggggggg
        h2=np.zeros((self.N,self.J,self.L),dtype=complex)    ### fffffffffffff
        for i in range(self.L):
            for i0 in range(self.K):
                h1[:,i0,i] = np.power(10,-3.53)/(np.power(self.d_3[i,i0],3.76))*((np.sqrt(self.K0/(1+self.K0))*self.nlos[i])
                +(np.sqrt(1/(2))*self.U1_los[:,i0,i]))  #(4,) 
                
        for j in range(self.L):
            for j0 in range (self.J):
                h2[:,j0,j]=np.power(10,-3.53)/(np.power(self.d_4[j,j0],3.76))*((np.sqrt(self.K0/(1+self.K0))*self.nlos[j])+(np.sqrt(1/(2))*self.U1_los[:,j0,j]))
        return h1, h2                            
    def _compute_D(self):#================================================================================
        G,F = self.get_channel()       # NML// MK
        h1,h2= self.get_h_channel()   # NKL/  NJL
        
        D = np.zeros((self.N,self.M,self.K, self.L), dtype= np.complex)#L   4,1,4
        D_D= np.zeros((self.N,self.M,self.K, self.L), dtype= np.complex)#
        
        # for i0 in range (self.K):
        #     for i in range(self.L):
        #         D[:,i,:] = np.matmul(np.diag(h1[:,i, i0]), G[:,:,i])  #3 NKL// NML (8,8 ),  (8, 4)///  (8, 4)
        #         D += np.random.normal(0,np.sqrt(self.channel_noise_var / 2), D.shape) + 1j * np.random.normal(0, np.sqrt(self.channel_noise_var / 2), D.shape)
        # return D   
        for i in range (self.L):
            for i0 in range (self.K):
                D[:,i,i0,:]=np.matmul(np.diag(h1[:,i0, i]),G[:,:,i])+ D_D[:,i,i0,:]  #3 NKL// NML (8,8 ),  (8, 4)///  (8, 4)
                D+=np.random.normal(0,np.sqrt(self.channel_noise_var / 2), D.shape) + 1j * np.random.normal(0, np.sqrt(self.channel_noise_var / 2), D.shape)
        return D
            
        
    def reset_state(self):
        D= self._compute_D()  #NMLK 
        G,F = self.get_channel()       # L # K
        h1,h2= self.get_h_channel()
        theta = np.random.rand(self.N,self.L)*2*np.pi          #(4,4)                             
        phi = np.zeros((self.N,self.N,self.L) ,dtype=complex)         
        for i in range(self.L):
            phi[:,:,i] = np.diag(np.exp(1j*theta[:,i]))                               
        x = np.zeros(self.L ) #, dtype=int)                                                                  ## int part I added 10 jan
        for i in range(self.L):     #initiate xl
            x[i]= np.random.randint(0,2)       #@@@@    (100,)
            
        s1 = np.zeros((self.L,self.K), dtype= np.complex)    ##2 dec///       ##  it will update at every episode (4,1)
        for i in range(self.K): 
            s_1= np.transpose(np.conjugate( F[:,i]))  #   .conj().T      #hkl_H   wk (4,)
            for i0 in range(self.L):
                s_1+= np.matmul(x[i0]*(phi[:,i0,:].T), D[:,i0,i,i0])# (8,8)       ///#(8,)  D[:,i0,i0,i]
               # s_1 += np.matmul(np.matmul(x[i0]*h1[:,i,i0].conj().T,phi[:,:,i0]),G[:,:,i0])                   
            s1[:,i] = s_1      #(4,)
            
        s2 = np.zeros((self.M,self.J), dtype= np.complex)    ##2 dec///       ##  it will update at every episode 
        for j in range(self.J): 
            for j0 in  range(self.L):
                s_2 = np.matmul(x[i0]*(phi[:,i0,:].T), D[:,j0,j,j0])
                #s_2 = np.matmul(np.matmul(x[j0]*h2[:,j,j0].conj().T,phi[:,:,j0]),G[:,:,j0]) #(8,)                   
            s2[:,j] = s_2 
            
        s = np.add(s1, s2)    # to add array    (4,1)         
        s_reshape = s.reshape(-1,1)                 # (16, 1)  only column me                
        state = np.concatenate((np.real(s_reshape),np.imag(s_reshape)), axis=0)     #(32, 1)
        return state[:,0]         #(8,)   
    
    def extarct_action(self,action):
        #` power_BS= np.zeros((self.L) ,dtype=np.float32)  # BS power 
        x = np.zeros(self.L )                     #x =L     (4,)                                                     # I add dtyep 12 jan
        theta_w1 = np.zeros((self.M*self.K))       # w= M,K     exp(j(theta))`
        theta_w2 = np.zeros((self.M*self.J))       # w= M,J     exp(j(theta))`        
        theta_irs = np.zeros(self.N*self.L)   # theta (irs) (0,2*pi)  (16,)
        
        for i in range(len(action)):
            if i < 1:
                power_BS = action[i]*self.P_max    #  BS power 
            elif i < self.L+1:  #x
                x[i-1] = int(np.around(action[i]))                #          bef==int(np.around(action[i]))====i remove int 
            elif i < (self.M*self.K)+self.L+1:   #w=exp(j(theta))               
                theta_w1[i-self.L-1] = action[i]*2*np.pi   #0=0.6524216
                
            elif i < ((self.M*self.J)+(self.M*self.K)+self.L+1):   #w=exp(j(theta))               
                theta_w2[i-self.M*self.K-self.L-1] = action[i]*2*np.pi   #0=0.6524216                 
           
            else:                                
                theta_irs[i-(self.M*(self.J+self.K)) -self.L-1] = np.exp(1j*action[i]*2*np.pi)     #theta    phase_normali//power_max
        return power_BS, x ,theta_w1, theta_w2, theta_irs
                
    
    def step(self, action):           #(41,) 
        #` Power_BS= np.zeros((self.L) ,dtype=np.float32)  # BS power   
        power_BS, x, theta_w1, theta_w2,theta_irs= self.extarct_action(action)
        W1 = np.zeros((self.M,self.K),dtype=np.complex)              #,dtype = int)    
        W2= np.zeros((self.M,self.J),dtype=np.complex)              #,dtype = int)    
        cntr1 = 0  
        cntr2 = 0 
        for i in range(self.M):
            for i0 in range(self.K):                                
                W1[i,i0] = np.exp(1j*theta_w1[cntr1]) 
                cntr1 += 1 
            for j0 in range(self.J):
                W2[i,j0] = np.exp(1j*theta_w2[cntr2]) 
                cntr2 += 1                                  
        W1 = (W1/np.linalg.norm(W1))*power_BS                      # BS Power (8,2)    ///  norm==||w||
        W2 = (W2/np.linalg.norm(W2))*power_BS     
        
        theta_irs = theta_irs.reshape((self.N,self.L))     #np.diag(theta_irs) (4,4)
        phi = np.zeros((self.N,self.N,self.L))
        for i in range(self.L):
            phi[:,:,i] = np.diag(theta_irs[:,i])
        # for i in range(self.L):            
        #      phi[:,i] = np.diag(np.exp(1j*phi[:,i])) 
###### ###////////////////////////// for 1

        D=self._compute_D()  ##  NML
        G,F = self.get_channel()   #L # K
        h1,h2 = self.get_h_channel()   # KL
        sr= np.zeros((self.M,self.K), dtype= np.complex)   #s  ==sr+st
        sinr1 = np.zeros((self.K),dtype=np.float32)   #LK
        rate1 = np.zeros((self.K),dtype=np.float32)     #LK
      ##  P = np.zeros((self.K),dtype=np.float32)
      
#///////////////////////   for 1       
        for i in range(self.K):                        
            s1_= F[:,i].conj().T    #(1, 8)          
            for i0 in  range(self.L):
                s1_+= np.matmul(x[i0]*(phi[:,i0,:].T),D[:,i0,i,i0])
                #s1_+= np.matmul(np.matmul(x[i0]*h1[:,i,i0].conj().T,phi[:,:,i0]),G[:,:,i0])     #phi[:,i0] /// (4,)          
                sr[:,i] = s1_   #(8,)  
                num_ = np.matmul(sr[:,i],W1[:,i])    #(8,) (8,)                                                     #before==np.matmul(s1,W[:,i]).....Ichange == 16 jan np.matmul(s[:,i],W[:,i])
                num = np.power(np.abs(num_),2)   # 156
                
                denum= self.awgn_var
                for i1 in range(self.K):                    
                    if i1 != i:                        
                        s2_ = F[:,i].conj().T    #(1, 4)   
                        for i2 in range(self.L):                            
                            s2_+= np.matmul(x[i2]*(phi[:,i2,:].T),D[:,i2,:]) #phi[:,i2]//(4,)
                            sr[:,i]=s2_
                        s_4 = np.matmul(sr[:,i],W1[:,i1])     # (1,4),(4,)== (1,)           # before==np.matmul(s2,W[:,i1]) 
                        denum+= np.power(np.abs(s_4),2)  # 22.00
                sinr1[i] = num/(denum)  #int 0.823
                rate1[i] = np.log2(1+sinr1[i])           #denum += self.awgn_var 
               
#//////////////////////////for 2       
        # s = np.zeros((self.M,self.J,self.K), dtype= np.complex) 
        sinr2= np.zeros((self.J),dtype=np.float32)   #LK
        rate2 = np.zeros((self.J),dtype=np.float32)    
        st = np.zeros((self.M,self.J), dtype= np.complex)               
        for j in range(self.J):                                           
            for j0 in  range(self.L):
                s2=np.matmul(x[j0]*(phi[:,j0,:].T),D[:,j0,j,j0])                
                #s2= np.matmul(np.matmul(x[j0]*h2[:,j,j0].conj().T,phi[:,:,j0]),G[:,:,j0])     #phi[:,i0] /// (4,)          
                st[:,j] = s2   #(8,)  
                num_ = np.matmul(st[:,j],W2[:,j])    #(8,) (8,)                                                     #before==np.matmul(s1,W[:,i]).....Ichange == 16 jan np.matmul(s[:,i],W[:,i])
                num = np.power(np.abs(num_),2)   # 156       
                
                denum= self.awgn_var
                for j1 in range(self.J):                    
                    if j1 != i:                        
                        for j2 in range(self.L):
                            s2= np.matmul(x[j2]*(phi[:,j2,:].T),D[:,j1,j2])                            
                           # s2 = np.matmul(np.matmul((x[j2]*h2[:,j,j2].conj().T),phi[:,:,j2]),G[:,:,j2]) #phi[:,i2]//(4,)
                            st[:,j]=s2
                        s_4 = np.matmul(st[:,j],W2[:,j1])     # (1,4),(4,)== (1,)           # before==np.matmul(s2,W[:,i1]) 
                        denum+= np.power(np.abs(s_4),2)  # 22.00
                        #denum += self.awgn_var
                        
                sinr2[j] = num/(denum)  #int 0.823
                rate2[j] = np.log2(1+sinr2[j])       #0.86  sinr[i]...maine i remove ki 
        
        s = np.add(sr,st)    # state   (4, 2)//////////////////////////////////////////////////////////////////////////////////////////
        rate= np.sum(rate1+rate2)                       
                          ## P[i] = np.linalg.norm(W)                 
# power:          
        P_1=0
        for K0 in range(self.K):
            P_11=np.abs(np.matmul(W1[:,K0], (np.transpose(np.conjugate(W1[:,K0]))))) *self.u   # power of bs     //power_bs  /////power_BS*self.u 
        for J0 in range(self.J):
            P_12=np.abs(np.matmul(W2[:,J0], (np.transpose(np.conjugate(W2[:,J0]))))) *self.u   # power of bs     //power_bs  /////power_BS*self.u    
        P_1+= P_11 +P_12
        
        P_3=0         
        for i in range(self.K):
            P_31 = self.P_K 
        for j in range(self.J):
            P_32 = self.P_J
        P_3+= P_31+ P_32
                        
        P_4 = 0     
        for l in range (self.L):                                                          
            P_4+= (x[l]*self.N)*self.P_R    # 40   x[L0]=0    
                     
        P_t = np.sum( self.P_B + P_3 + P_4 )  # 54
        
        # Power_BS[i] = np.linalg.norm(W[:,0,:,i])                    
        reward= rate/ P_t       #pen if W_h.W<= P_max  //(np.sum(power_BS)*self.u+ self.P_B+self.P_K+P_4 )                                                       #  before==sum(rate)/P_T /// after==  16 jansum(rate[i])/P_T                                                   # np.sum(rate)/(P_t) 
        s1_reshape = s.reshape(-1,1)   #(8, 1)
        new_state = np.concatenate((np.real(s1_reshape),np.imag(s1_reshape)),axis=0)     #(16, 1)                                                            #np.sum(rate) /(P_t )             
        return reward, new_state[:,0]          #(8,)   Power_BS
    
    
    































'''import numpy as np
# def db2lin(db):
#     lin = np.power(10,(db/10))
#     return lin  
class UEs_and_utilities():                   
    def __init__(self,M,N,K,J,L):
        self.K = K
        self.J =J
        self.L = L
        self.N = N
        self.M = M
        self.bs_loc = np.array([[0],[0]])   ##(0,0,0)
        self.generate_irs_loc()                 ##(100,0,50)
        self.generete_user_loc1()                ##random( 54,32)
        self.generete_user_loc2()
        self.nlos= self.get_NLOS_array()
        self.U1_los, self.U2_los , self.bs_ris= self.get_random_channel()
      
    def generate_irs_loc(self):
        irs_loc = np.zeros((2,self.L),dtype = float)
        for i in range(self.L):
            irs_loc[:,i] = np.array([[np.cos(2*i*np.pi/self.L)*100],[np.sin(2*i*np.pi/self.L)*100]])[0]   #(3, 1)  #sirf ro k liye 
        self.irs_loc = irs_loc
        
    def generete_user_loc1(self):    #rand(Num_User,2)*lengA;
        user_loc1 = np.zeros((2,self.K))
        for i in range(self.K):
            user_loc1[:,i] = np.array([[np.random.uniform(-100,100)],[np.random.uniform(-100,100)]])[0]     #random( 54,32)
        self.user_loc1 = user_loc1   
    def generete_user_loc2(self):   #rand(Num_User,2)*lengA;
       user_loc2 = np.zeros((2,self.J))
       for j in range(self.J):
           user_loc2[:,j] = np.array([[np.random.uniform(-100,100)],[np.random.uniform(-100,100)]])[0]     #random( 54,32)
       self.user_loc2 = user_loc2

    def get_NLOS_array(self):##############################################################################################################
        nlos = np.zeros((self.L),dtype=np.complex)
        for i in range(self.L):            
                nlos[i] = np.exp(-1j*2*np.pi*i*0.5*np.sin(np.random.rand(1)*2*np.pi))
        return nlos     
    def get_random_channel(self):
        U1_los = np.random.randn(self.N,self.K,self.L)+(1j*(np.random.randn(self.N,self.K,self.L)))
        U2_los = (np.random.randn(self.N,self.J,self.L)+(1j*(np.random.randn(self.N,self.J,self.L))))
        bs_ris = (np.random.randn(self.N,self.M,self.L)+(1j*(np.random.randn(self.N,self.M,self.L))))
            
        return U1_los, U2_los, bs_ris #with x and y axist                

class RIS_D(object):               
    def __init__(self, M, N, K,J,L, P_B, P_K, P_J, P_R,K0,u,awgn_var,power_properties,phase_properties,UEs ):        
        self.M = M        # BS station ante
        self.N = N        # IRS element N_l
        self.K = K         # user  
        self.J = J
        self.L= L           # no of irs       
# power
        self.u= u       #1/v = .`W11/0.8
        self.P_B= P_B
        self.P_K= P_K
        self.P_J= P_J
        self.P_R= P_R         #x.N.P_r 
        self.awgn_var = awgn_var
        self.K0 = K0
        
        self.power_properties = power_properties   #
        self.phase_properties = phase_properties
        
        self.P_max = np.max(power_properties)     #100Watt   Power_normal   // P_max
        self.Phase_max= np.max(phase_properties)     # phase normali
      # self.x=x              #x_l  [0,1]        
 # action
        self.theta = np.random.rand(self.N, self.L)*2*np.pi                           #np.random.rand(self.N)*2*np.pi                      
        self.Phi = np.zeros(self.N, dtype=complex)                                #np.zeros(self.M,self.K   dtype= complex) 
        self.w = np.zeros((self.M, self.K), dtype= complex) #BF vector for user K
        self.x = np.zeros(self.L)              
        self.UEs = UEs
        
        # self.h_nlos = np.random.randn(self.N)+(1j*(np.random.randn(self.N)))     #IRS_USER  ##  # fix # NLOS
        # self.h_random = np.random.randn(self.K)+(1j*(np.random.randn(self.K)))  # BS_USer      #   fix   ### NLOS ###########################**************
        #self.squire_array= self.get_squire_array()
        self.d_1, self.d_2 = self.get_dis() 
        self.d_3, self.d_4 = self.get_dis_IUS()     #IRS_US
        
        self.nlos = UEs.nlos
        self.U1_los = UEs.U1_los
        self.U2_los = UEs.U2_los
        self.bs_ris = UEs.bs_ris
        
        self.h1 = self.get_h_channel()
        self.h2 = self.get_h_channel()
        self.F= self.get_channel()
        self.G= self.get_channel()
        
#        self.state = self.reset_state()
    
        self.action_space = self.M*self.K + (self.M*self.J ) + (self.N*self.L) + (self.L) + 1                          # theta(N) + w(M,1) + X((0,1 ))/theta(matrix) + W(vector) + x(vector)      
        self.observation_space =   2*(self.M*self.K*self.J)
 
#        self.min_action, self.max_action = self.get_bound_action() 
    def get_squire_array(self):  #h_LOS   # irs ref ele
        squire_array = np.zeros(self.N,dtype=np.complex)  # irs   #(4,)
        for i in range(self.N):
            squire_array[i] = np.exp(-1j*2*np.pi*i*0.5*np.cos(np.random.rand(1)*2*np.pi))   # complex array (0j) ==  (1,)            
        return squire_array        
#distance
    def get_dis(self):       ## BS_IRS   # BS_USer_1
        d_1 = np.zeros((self.L),dtype=np.float32)               #BS_IRS   #(L,N,M)  .....L ka for loop
        d_2 = np.zeros((self.K),dtype=np.float32)               #BS_US_1 (1,M)   ...no for loop        
        #d_3 = np.zeros((self.N,self.M),dtype=np.float32)       #IRS_US
        for i in range(self.L):
            d_1[i]= np.sqrt((self.UEs.irs_loc[0,i]-self.UEs.bs_loc[0])**2 + 
                                 (self.UEs.irs_loc[1,i]-self.UEs.bs_loc[1])**2 )   # + (self.UEs.irs_loc[2,i]-self.UEs.bs_loc[2])**2)
        for i0 in range(self.K):                
            d_2[i0] = np.sqrt((self.UEs.bs_loc[0]-self.UEs.user_loc1[0,i0])**2 + (self.UEs.bs_loc[1]-self.UEs.user_loc1[1,i0])**2 )# + (self.UEs.bs_loc[2]-self.UEs.user_loc[2,i0])**2)                
        return d_1, d_2    #122,104
           
    def get_dis_IUS(self):    #   IRS_ US_1,      US_2
        d_3 = np.zeros((self.L,self.K),dtype=np.float32)   #Nk jagah pe--K hoga   KLN...KL ka for loop 
        d_4 = np.zeros((self.L,self.J),dtype=np.float32)   #Nk jagah pe--K hoga   KLN...KL ka for loop
        for i in range(self.L):
            for i0 in range(self.K):
                d_3[i,i0] = np.sqrt((self.UEs.irs_loc[0,i]-self.UEs.user_loc1[0,i0])**2 + 
                                       (self.UEs.irs_loc[1,i]-self.UEs.user_loc1[1,i0])**2 )  #+ (self.UEs.irs_loc[2,i]-self.UEs.user_loc[2,i0])**2)  
        for j in range(self.L):
            for j0 in range(self.J):
                d_4[j,j0] = np.sqrt((self.UEs.irs_loc[0,j]-self.UEs.user_loc2[0,j0])**2 + (self.UEs.irs_loc[1,j]-self.UEs.user_loc2[1,j0])**2 )                        
        return d_3 ,d_4   #235
    
#/////// chanel 
# BS_IRS //  BS_USER_1  
    def get_channel(self):  # BS_IRS  //  BS_USER     
        G = np.zeros((self.N,self.M,self.L),dtype=np.complex)   # BS_IRS   L
        F = np.zeros((self.M,self.K),dtype=np.complex)    #BS_USER  K
        
        for i in range(self.L):  #L=8     (4,2)  2D     
            G[:,:,i] =(np.power(10,-3.53)/(np.power(self.d_1[i],0.9)))*((np.sqrt(self.K0/(1+self.K0))*self.nlos[i]) +(np.sqrt(1/(2))*self.bs_ris[:,:,i]))
            
        for i0 in range(self.K):  
            F[:,i0] = (np.random.randn(self.M) + 1j*(np.random.randn(self.M)))*(np.power(10,-3.53)/(np.power(self.d_2[i0], 3.76)))*np.sqrt(0.5) #K /(8,)               
        return G, F  
    
# IRS_USER_1, IRS_US_2                                                  
    def get_h_channel(self):
        h1 = np.zeros((self.N,self.K,self.L),dtype=complex)
        h2=np.zeros((self.N,self.J,self.L),dtype=complex)
        
        
        for i in range(self.L):
            for i0 in range(self.K):
                h1[:,i0,i] = (np.power(10,-3.53)/(np.power(self.d_3[i,i0],3.76)))*((np.sqrt(self.K0/(1+self.K0))*self.nlos[i]) 
                                                                                   +(np.sqrt(1/(2))*self.U1_los[:,i0,i]))  #(4,) 
        for j in range(self.L):
            for j0 in range (self.J):
                h2[:,j0 ,j]=(np.power(10,-3.53)/(np.power(self.d_4[j,j0],3.76)))*((np.sqrt(self.K0/(1+self.K0))*self.nlos[j])+(np.sqrt(1/(2))*self.U1_los[:,j0,j]))
        return h1, h2                            
                
    def reset_state(self):       
        G,F = self.get_channel()       # L # K
        h1,h2= self.get_h_channel()
        theta = np.random.rand(self.N,self.L)*2*np.pi          #(4,4)                             
        phi = np.zeros((self.N,self.N,self.L) ,dtype=complex)         
        for i in range(self.L):
            phi[:,:,i] = np.diag(np.exp(1j*theta[:,i]))                               
        x = np.zeros(self.L ) #, dtype=int)                                                                  ## int part I added 10 jan
        for i in range(self.L):     #initiate xl
            x[i]= np.random.randint(0,2)       #@@@@    (100,)
            
        s1 = np.zeros((self.M,self.K), dtype= np.complex)    ##2 dec///       ##  it will update at every episode 
        for i in range(self.K): 
            s_1= np.transpose(np.conjugate( F[:,i]))  #   .conj().T      #hkl_H   wk 
            for i0 in  range(self.L):
                s_1+= np.matmul(np.matmul(x[i0]*h1[:,i,i0].conj().T,phi[:,:,i0]),G[:,:,i0]) #(8,)                   
            s1[:,i] = s_1      #(4,)
            
        s2 = np.zeros((self.M,self.J), dtype= np.complex)    ##2 dec///       ##  it will update at every episode 
        for j in range(self.J): 
            for j0 in  range(self.L):
                s_2 = np.matmul(np.matmul(x[j0]*h2[:,j,j0].conj().T,phi[:,:,j0]),G[:,:,j0]) #(8,)                   
            s2[:,j] = s_2 
            
        s = np.add(s1, s2)    # to add array            
        s_reshape = s.reshape(-1,1)                 # (16, 1)  only column me                
        state = np.concatenate((np.real(s_reshape),np.imag(s_reshape)), axis=0)     #(32, 1)
        return state[:,0]         #(32),)   
    
    def extarct_action(self,action):
        #` power_BS= np.zeros((self.L) ,dtype=np.float32)  # BS power 
        x = np.zeros(self.L )                     #x =L     (4,)                                                     # I add dtyep 12 jan
        theta_w1 = np.zeros((self.M*self.K))       # w= M,K     exp(j(theta))`
        theta_w2 = np.zeros((self.M*self.J))       # w= M,J     exp(j(theta))`        
        theta_irs = np.zeros(self.N*self.L)   # theta (irs) (0,2*pi)  (16,)
        
        for i in range(len(action)):
            if i < 1:
                power_BS = action[i]*self.P_max    #  BS power 
            elif i < self.L+1:  #x
                x[i-1] = int(np.around(action[i]))                #          bef==int(np.around(action[i]))====i remove int 
            elif i < (self.M*self.K)+self.L+1:   #w=exp(j(theta))               
                theta_w1[i-self.L-1] = action[i]*2*np.pi   #0=0.6524216
                
            elif i < (self.M*self.J)+(self.M*self.K)+self.L+1:   #w=exp(j(theta))               
                theta_w2[i-self.M*self.K-self.L-1] = action[i]*2*np.pi   #0=0.6524216    
            else:                                
                theta_irs[i-(self.M*(self.K+ self.J)) -self.L-1] = np.exp(1j*action[i]*2*np.pi)     #theta    phase_normali//power_max
        return power_BS, x ,theta_w1, theta_w2, theta_irs
                
    
    def step(self, action):           #(41,) 
       #` Power_BS= np.zeros((self.L) ,dtype=np.float32)  # BS power   
        power_BS, x, theta_w1, theta_w2,theta_irs= self.extarct_action(action)
        W1 = np.zeros((self.M,self.K),dtype=np.complex)              #,dtype = int)    
        W2= np.zeros((self.M,self.J),dtype=np.complex)              #,dtype = int)    
        cntr1 = 0  
        cntr2 = 0 
        for i in range(self.M):
            for i0 in range(self.K):                                
                W1[i,i0] = np.exp(1j*theta_w1[cntr1]) 
                cntr1 += 1 
            for j0 in range(self.J):
                W2[i,j0] = np.exp(1j*theta_w2[cntr2]) 
                cntr2 += 1                                  
        W1 = (W1/np.linalg.norm(W1))*power_BS                      # BS Power (8,2)    ///  norm==||w||
        W2 = (W2/np.linalg.norm(W2))*power_BS     
        
        theta_irs = theta_irs.reshape((self.N,self.L))     #np.diag(theta_irs) (4,4)
        phi = np.zeros((self.N,self.N,self.L))
        for i in range(self.L):
            phi[:,:,i] = np.diag(theta_irs[:,i])
        # for i in range(self.L):            
        #      phi[:,i] = np.diag(np.exp(1j*phi[:,i])) 
###### ###////////////////////////// for 1
        G,F = self.get_channel()   #L # K
        h1,h2 = self.get_h_channel()   # KL
        sr = np.zeros((self.M,self.K), dtype= np.complex)   #s  ==sr+st
        sinr1 = np.zeros((self.K),dtype=np.float32)   #LK
        rate1 = np.zeros((self.K),dtype=np.float32)     #LK
      ##  P = np.zeros((self.K),dtype=np.float32)
#///////////////////////   for 1       
        for i in range(self.K):                        
            s1= F[:,i].conj().T    #(1, 8)          
            for i0 in  range(self.L):                
                s1+= np.matmul(np.matmul(x[i0]*h1[:,i,i0].conj().T,phi[:,:,i0]),G[:,:,i0])     #phi[:,i0] /// (4,)          
                sr[:,i] = s1   #(8,)  
                num_ = np.matmul(sr[:,i],W1[:,i])    #(8,) (8,)                                                     #before==np.matmul(s1,W[:,i]).....Ichange == 16 jan np.matmul(s[:,i],W[:,i])
                num = np.power(np.abs(num_),2)   # 156
                
                denum= self.awgn_var
                for i1 in range(self.K):                    
                    if i1 != i:                        
                        s2 = F[:,i].conj().T    #(1, 4)   
                        for i2 in range(self.L):                            
                            s2 += np.matmul(np.matmul((x[i2]*h1[:,i,i2].conj().T),phi[:,:,i2]),G[:,:,i2]) #phi[:,i2]//(4,)
                            sr[:,i]=s2
                        s_4 = np.matmul(sr[:,i],W1[:,i1])     # (1,4),(4,)== (1,)           # before==np.matmul(s2,W[:,i1]) 
                        denum+= np.power(np.abs(s_4),2)  # 22.00
                sinr1[i] = num/(denum)  #int 0.823
                rate1[i] = np.log2(1+sinr1[i])           #denum += self.awgn_var 
               
#//////////// //////////////for 2       
       # s = np.zeros((self.M,self.J,self.K), dtype= np.complex) 
        sinr2= np.zeros((self.J),dtype=np.float32)   #LK
        rate2 = np.zeros((self.J),dtype=np.float32)    
        st = np.zeros((self.M,self.J), dtype= np.complex)               
        for j in range(self.J):                                           
            for j0 in  range(self.L):                
                s2= np.matmul(np.matmul(x[j0]*h2[:,j,j0].conj().T,phi[:,:,j0]),G[:,:,j0])     #phi[:,i0] /// (4,)          
                st[:,j] = s2   #(8,)  
                num_ = np.matmul(st[:,j],W2[:,j])    #(8,) (8,)                                                     #before==np.matmul(s1,W[:,i]).....Ichange == 16 jan np.matmul(s[:,i],W[:,i])
                num = np.power(np.abs(num_),2)   # 156       
                
                denum= self.awgn_var
                for j1 in range(self.J):                    
                    if j1 != i:                        
                        for j2 in range(self.L):                            
                            s2 = np.matmul(np.matmul((x[j2]*h2[:,j,j2].conj().T),phi[:,:,j2]),G[:,:,j2]) #phi[:,i2]//(4,)
                            st[:,j]=s2
                        s_4 = np.matmul(st[:,j],W2[:,j1])     # (1,4),(4,)== (1,)           # before==np.matmul(s2,W[:,i1]) 
                        denum+= np.power(np.abs(s_4),2)  # 22.00
                        #denum += self.awgn_var
                        
                sinr2[j] = num/(denum)  #int 0.823
                rate2[j] = np.log2(1+sinr2[j])       #0.86  sinr[i]...maine i remove ki 
        
        s = np.add(sr,st)    # state 
        rate= np.sum(rate1+rate2)                       
                         ## P[i] = np.linalg.norm(W)                 
# power:
       
    
        P_1=0
        for K0 in range(self.K):
            P_11=np.abs(np.matmul(W1[:,K0], (np.transpose(np.conjugate(W1[:,K0]))))) *self.u   # power of bs     //power_bs  /////power_BS*self.u            
        for J0 in range(self.J):
            P_12=np.abs(np.matmul(W2[:,J0], (np.transpose(np.conjugate(W2[:,J0]))))) *self.u   # power of bs     //power_bs  /////power_BS*self.u    
        P_1+= P_11+P_12
        
        P_3=0         
        for i in range(self.K):
            P_31 = self.P_K 
        for j in range(self.J):
            P_32 = self.P_J
        P_3+= P_31+ P_32
                        
        P_4 = 0     
        for l in range (self.L):                                                          
            P_4+= (x[l]*self.N)*self.P_R    # 40   x[L0]=0    
                     
        P_t = np.sum( P_1+self.P_B + P_3 + P_4)  # 54
        
       # Power_BS[i] = np.linalg.norm(W[:,0,:,i])                    
        reward =rate/ P_t       #pen if W_h.W<= P_max  //(np.sum(power_BS)*self.u+ self.P_B+self.P_K+P_4 )                                                       #  before==sum(rate)/P_T /// after==  16 jansum(rate[i])/P_T                                                   # np.sum(rate)/(P_t) 
        s1_reshape = s.reshape(-1,1)   #(4, 1)
        new_state = np.concatenate((np.real(s1_reshape),np.imag(s1_reshape)),axis=0)     #(8, 1)                                                            #np.sum(rate) /(P_t )             
        return reward, new_state[:,0]          #(32,)   Power_BS'''
    
    
    
   
    
   
    
   