

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
    
        self.action_dim = self.M*self.K*self.J + (self.N*self.L)+(self.L)+1         # theta(N) + w(M,1) + X((0,1 ))/theta(matrix) + W(vector) + x(vector)      
        self.state_dim =   2*(self.M*self.K )   #+2*(self.M*self.J )
 
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
    def get_hk_channel(self):
        h_k = np.zeros((self.M,self.K),dtype=np.complex)    #BS_USER  K   hhhhhhhhhhhhh
        for i0 in range(self.K):  ##///// NO--NLOS////// RANOM_NLOS
            h_k[:,i0] =  (np.power(10,-3.53)/(np.power(self.d_2[i0], 3.76)))*np.sqrt(0.5)*((np.random.randn(self.M) + 1j*(np.random.randn(self.M)))) #K /(8,)               
            h_k+= np.random.normal(0,np.sqrt(self.channel_noise_var / 2), h_k.shape) + 1j * np.random.normal(0, np.sqrt(self.channel_noise_var / 2), h_k.shape)
        return h_k                      
    def _compute_D(self):#================================================================================
        G,F = self.get_channel()       # NML// MK
        h1,h2= self.get_h_channel()   # NKL/  NJL
        # D=(np.random.randn(self.N,self.M,self.K, self.L) + 1j*(self.N,self.M,self.K, self.L))
        # D_D=(np.random.randn(self.N,self.M,self.K, self.L) + 1j*(self.N,self.M,self.K, self.L))
        D = np.zeros((self.N,self.M,self.K, self.L), dtype= np.complex)#L   4,1,4
        D_D= np.zeros((self.N,self.M,self.K, self.L), dtype= np.complex)#
        
        # for i0 in range (self.K):
        #     for i in range(self.L):
        #         D[:,i,:] = np.matmul(np.diag(h1[:,i, i0]), G[:,:,i])  #3 NKL// NML (8,8 ),  (8, 4)///  (8, 4)
        #         D += np.random.normal(0,np.sqrt(self.channel_noise_var / 2), D.shape) + 1j * np.random.normal(0, np.sqrt(self.channel_noise_var / 2), D.shape)
        # return D
        for i in range (self.L):
            for i0 in range(self.K):
                D[:,i,i0,] = np.matmul(np.diag(h1[:,i0, i]), G[:,:,i])+ D_D[:,i,i0,:]  #3 NKL// NML (8,8 ),  (8, 4)///  (8, 4)
                D += np.random.normal(0,np.sqrt(self.channel_noise_var / 2), D.shape) + 1j * np.random.normal(0, np.sqrt(self.channel_noise_var / 2), D.shape)
        return D
        # for i in range (self.L):
        #     for i0 in range (self.K):
        #         D[:,i,i0,:]=np.add(np.matmul(np.diag(h1[:,i0, i]),G[i,:,:]), D_D[:,i,i0,:] ) #3 NKL// NML (8,8 ),  (8, 4)///  (8, 4)
        #         D+=np.random.normal(0,np.sqrt(self.channel_noise_var / 2), D.shape) + 1j * np.random.normal(0, np.sqrt(self.channel_noise_var / 2), D.shape)
        # return D
            
        
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
                            s2_+= np.matmul(x[i2]*(phi[:,i2,:].T),D[:,i2,i1,i2]) #phi[:,i2]//(4,)
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
                            s2= np.matmul(x[j2]*(phi[:,j2,:].T),D[:,j2,j1,j2])                            
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
import os
# def db2lin(db):
#     lin = np.power(10,(db/10))
#     return lin  
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
    def __init__(self, M, N, K,J,L, P_B, P_K, P_J, K0, P_R,u,awgn_var,channel_noise_var,power_properties,phase_properties,UEs):        
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
    
        self.action_dim = self.M*self.K*self.J + (self.N*self.L)+(self.L)+1         # theta(N) + w(M,1) + X((0,1 ))/theta(matrix) + W(vector) + x(vector)      
        self.state_dim =   2*(self.M*self.K *self.J)   #+2*(self.M*self.J )lm+lk
 
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
    # def _compute_D(self):
    #     D = np.diag(self.H_2[:, 0]) @ self.H_1

    #     for column_idx in np.arange(1, self.H_2.shape[1]):
    #         D = np.vstack((D, np.diag(self.H_2[:, column_idx] @ self.H_1)))

    #     if self.channel_est_error:
    #         D += np.random.normal(0, np.sqrt(self.channel_noise_var / 2), D.shape) + 1j * np.random.normal(0, np.sqrt(self.channel_noise_var / 2), D.shape)

    #     return D                        
    def _compute_D(self):#================================================================================
        G ,F = self.get_channel()       # NML// MK
        h1,h2= self.get_h_channel()   # NKL/  NJL
        #np.real(np.diag(self.G.conjugate().T @ self.G)).reshape(1, -1) ** 2
        D = np.zeros((self.N,self.M,self.K, self.L), dtype= np.complex)#L   4,1,4
        D_D= np.zeros((self.N,self.M,self.K, self.L), dtype= np.complex)#        
        for i in range (self.L):
            for i0 in range(self.K):
                D[:,i,i0,] = np.matmul(np.diag(h1[:,i0, i]), G[:,:,i])+ D_D[:,i,i0,:]  #3 NKL// NML (8,8 ),  (8, 4)///  (8, 4)
                D += np.random.normal(0,np.sqrt(self.channel_noise_var / 2), D.shape) + 1j * np.random.normal(0, np.sqrt(self.channel_noise_var / 2), D.shape)
        return D   
    #     for i in range (self.L):
    #         for i0 in range (self.K):
    #           #  s=D_D[:,i,i0,:]
    #             #s_11=np.diag(h1[:,i0, i])
    #            # s=np.matmul(s_11,G[:,:,i])     #+ D_D[:,i,i0,:]   # M-set
    #             D[:,i,i0,:]= np.real(np.diag(self.h1[:,i0, i].conjugate().T @ self.G[:,:,i])).reshape(1, -1) ** 2
    #             #D[:,i,i0,:]=np.real(np.diag(h1[:,i0, i]conjugate().T, G[:,:,i])).reshape(1, -1) ** 2  #+ D_D[:,i,i0,:]  #3 NKL// NML (8,8 ),  (8, 4)///  (8, 4)  N-set __max_set k liye
    #           #  D[:,i,i0,]=np.matmul(s_11,G[:,:,i])
    #             D+=np.random.normal(0,np.sqrt(self.channel_noise_var / 2), D.shape)+1j * np.random.normal(0, np.sqrt(self.channel_noise_var / 2), D.shape)
    #     return D
            
    # def _compute_D(self):#================================================================================
    #      return np.diag(self.h1.conjugate().T) @ self.G
       
        # G = self.get_channel()       # NML// MK
        # h1= self.get_h_channel()   # NKL/  NJL
        # D= np.zeros((self.N,self.M,self.K, self.L), dtype= np.complex)
        # for i in range (self.L):
        #     for i0 in range (self.K):
        #         D[:,i,i0,]= np.diag(h1[:,i0, i].conjugate().T) , G[:,:,i]
        #         D += np.random.normal(0, np.sqrt(self.channel_noise_var / 2), D.shape) + 1j * np.random.normal(0, np.sqrt(self.channel_noise_var / 2), D.shape)

        # return D
       
    def reset_state(self):
        D= self._compute_D()  #NMLK 
        F,G = self.get_channel()       #F- mk
        #h1,h2= self.get_h_channel()
        theta = np.random.rand(self.N,self.L)*2*np.pi          #(4,4)                             
        phi = np.zeros((self.N,self.N,self.L) ,dtype=complex)         
        for i in range(self.L):
            phi[:,:,i] = np.diag(np.exp(1j*theta[:,i]))                               
        x = np.zeros(self.L ) #, dtype=int)                                                                  ## int part I added 10 jan
        for i in range(self.L):     #initiate xl
            x[i]= np.random.randint(0,2)       #@@@@    (100,)
            
        s1 = np.zeros((self.L,self.K), dtype= np.complex)    ##2 dec///       ##  it will update at every episode (4,1)
        for i in range(self.K): 
            s_1= np.transpose(np.conjugate( F[:,i]))                     #( F[:,i]))  #   .conj().T      #hkl_H   wk (4,)
            for i0 in range(self.L):
                s_1+= np.matmul(x[i0]*(phi[:,i0,:].T),D[i,:,i0,:])# (8,8)       ///#(8,)  D[:,i0,i0,i]
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
        return reward, new_state[:,0]          #(8,)   Power_BS'''
    
    
    
   
    
 
    
 

'''import numpy as np
import os
# def db2lin(db):
#     lin = np.power(10,(db/10))
#     return lin  
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
    def __init__(self, M, N, K,J,L, P_B, P_K, P_J, K0, P_R,u,awgn_var,channel_est_error, channel_noise_var,power_properties,phase_properties,UEs, ):        
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
        self.channel_est_error = channel_est_error
        self.channel_noise_var = channel_noise_var
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
        
        self.h1 = self.get_h_channel()
        self.h2 = self.get_h_channel()
        self.F= self.get_channel()
        self.G= self.get_channel()
        self.D= self._compute_D()
        
#        self.state = self.reset_state()
    
        self.action_dim = self.M*self.K*self.J + (self.N*self.L)+(self.L)+1         # theta(N) + w(M,1) + X((0,1 ))/theta(matrix) + W(vector) + x(vector)      
        self.state_dim =   2*(self.M*self.K *self.J)   #+2*(self.M*self.J )
        
 
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
# BS_IRS //  BS_USER_1  //// RANOM_NLOS
    def get_channel(self):  # BS_IRS  //  BS_USER     
        G = np.zeros((self.N,self.M,self.L),dtype=np.complex)   # BS_IRS   L
        F = np.zeros((self.M,self.K),dtype=np.complex)    #BS_USER  K
        
        for i in range(self.L):  #L=8     (4,2)  2D    bs_ris 
            G[:,:,i] = np.power(10,-3.53)/(np.power(self.d_1[i],0.9))*((np.sqrt(self.K0/(1+self.K0))*self.nlos[i]) +(np.sqrt(1/(2))*self.bs_ris[:,:,i]))
                                                                                      
            
        for i0 in range(self.K):  ##///// NO--NLOS////// RANOM_NLOS
            F[:,i0] =  (np.power(10,-3.53)/(np.power(self.d_2[i0], 3.76)))*np.sqrt(0.5)*((np.random.randn(self.M) + 1j*(np.random.randn(self.M)))) #K /(8,)               
        return G, F  
    
# IRS_USER_1, IRS_US_2   ##+++  \\\\\\NLOS                                               
    def get_h_channel(self):
        h1 = np.zeros((self.N,self.K,self.L),dtype=complex)
        h2=np.zeros((self.N,self.J,self.L),dtype=complex)
        for i in range(self.L):
            for i0 in range(self.K):
                h1[:,i0,i] = np.power(10,-3.53)/(np.power(self.d_3[i,i0],3.76))*((np.sqrt(self.K0/(1+self.K0))*self.nlos[i])
                +(np.sqrt(1/(2))*self.U1_los[:,i0,i]))  #(4,) 
                
        for j in range(self.L):
            for j0 in range (self.J):
                h2[:,j0,j]=np.power(10,-3.53)/(np.power(self.d_4[j,j0],3.76))*((np.sqrt(self.K0/(1+self.K0))*self.nlos[j])+(np.sqrt(1/(2))*self.U1_los[:,j0,j]))
        return h1, h2                            
    def _compute_D(self):
        G,F = self.get_channel()       # L # K
        h1,h2= self.get_h_channel()
        D = np.zeros((self.L,self.M,self.K), dtype= np.complex)#L
        for i in range (self.K):
            for i0 in range (self.L):
                D = (np.diag(h1[:,i, i0])@ G[:,:,i0])  #3 NKL// NML
            D += np.random.normal(0,np.sqrt(self.channel_noise_var / 2), D.shape) + 1j * np.random.normal(0, np.sqrt(self.channel_noise_var / 2), D.shape)
        return D
# D_1 = np.zeros((self.N,self.M, self.K), dtype= np.complex)   #========================================================================================
# #G = np.zeros((self.N,self.M,self.L),dtype=np.complex)
# #h1 = np.zeros((self.N,self.K,self.L)
# #D_2= 
# for i in range(self.K):
#     for i0 in range(self.L):
#         D_1=np.diag(h1[:,i,i0],G[:,:,i0] )  ### NKL  NML
#     D=D_1+D_2    #==========================================================================================================================            
    def reset_state(self):
        D=  self._compute_D()
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
                s_1+= x[i0]*(np.matmul(phi[:,:,i0]),D[:,:,i0]) #(8,)
                #s_1+= np.matmul(np.matmul(x[i0]*h1[:,i,i0].conj().T,phi[:,:,i0]),G[:,:,i0]) #(8,)                   
            s1[:,i] = s_1      #(4,)
            
        s2 = np.zeros((self.M,self.J), dtype= np.complex)    ##2 dec///       ##  it will update at every episode 
        for j in range(self.J): 
            for j0 in  range(self.L):
                s_2 = np.matmul(np.matmul(x[j0]*h2[:,j,j0].conj().T,phi[:,:,j0]),G[:,:,j0]) #(8,)                   
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
        G,F = self.get_channel()   #L # K
        h1,h2 = self.get_h_channel()   # KL
        sr = np.zeros((self.M,self.K), dtype= np.complex)   #s  ==sr+st
        sinr1 = np.zeros((self.K),dtype=np.float32)   #LK
        rate1 = np.zeros((self.K),dtype=np.float32)     #LK
      ##  P = np.zeros((self.K),dtype=np.float32)
      
#///////////////////////   for 1
# s1 = np.zeros((self.N,self.M, self.L), dtype= np.complex)
# G = np.zeros((self.N,self.M,self.L),dtype=np.complex) 
# for i in range(self.L):
#     for i0 in range(self.N):
#     D_1=np.diag(h1[i,:,i0].conj().T,G[i,i0,:] )  ### NKL  NML
#     D_2=np.add(D_1,G)
 
# D=D_1+D_2      
        for i in range(self.K):                        
            s1_= F[:,i].conj().T    #(1, 8)          
            for i0 in  range(self.L):
                #s1_+= np.matmul(np.matmul(x[i0]*phi[:,:,i0]),D[:,:,i0])
                s1_+= np.matmul(np.matmul(x[i0]*h1[:,i,i0].conj().T,phi[:,:,i0]),G[:,:,i0])     #phi[:,i0] /// (4,)          
                sr[:,i] = s1_   #(8,)  
                num_ = np.matmul(sr[:,i],W1[:,i])    #(8,) (8,)                                                     #before==np.matmul(s1,W[:,i]).....Ichange == 16 jan np.matmul(s[:,i],W[:,i])
                num = np.power(np.abs(num_),2)   # 156
                
                denum= self.awgn_var
                for i1 in range(self.K):                    
                    if i1 != i:                        
                        s2_ = F[:,i].conj().T    #(1, 4)   
                        for i2 in range(self.L):                            
                            s2_+= np.matmul(np.matmul((x[i2]*h1[:,i,i2].conj().T),phi[:,:,i2]),G[:,:,i2]) #phi[:,i2]//(4,)
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
        reward =rate/ P_t       #pen if W_h.W<= P_max  //(np.sum(power_BS)*self.u+ self.P_B+self.P_K+P_4 )                                                       #  before==sum(rate)/P_T /// after==  16 jansum(rate[i])/P_T                                                   # np.sum(rate)/(P_t) 
        s1_reshape = s.reshape(-1,1)   #(8, 1)
        new_state = np.concatenate((np.real(s1_reshape),np.imag(s1_reshape)),axis=0)     #(16, 1)                                                            #np.sum(rate) /(P_t )             
        return reward, new_state[:,0]          #(8,)   Power_BS'''
    
    
    
   
    
 
    
 #################################### for same J&K=====KKKKKKKKKK only k bna kr 
# import numpy as np
# # def db2lin(db):
# #     lin = np.power(10,(db/10))
# #     return lin  
# class UEs_and_utilities():                   
#     def __init__(self,M,N,K,J,L):
#         self.K = K
#         self.L = L
#         self.bs_loc = np.array([[0],[0]])   ##(0,0,0)
#         self.generate_irs_loc()                 ##(100,0,50)
#         self.generete_user_loc1()                ##random( 54,32)
#         self.generete_user_loc2()
      
#     def generate_irs_loc(self):
#         irs_loc = np.zeros((2,self.L),dtype = float)
#         for i in range(self.L):
#             irs_loc[:,i] = np.array([[np.cos(2*i*np.pi/self.L)*100],[np.sin(2*i*np.pi/self.L)*100]])[0]   #(3, 1)  #sirf ro k liye 
#         self.irs_loc = irs_loc
        
#     def generete_user_loc1(self):    #rand(Num_User,2)*lengA;
#         user_loc1 = np.zeros((2,self.K))
#         for i in range(self.K):
#             user_loc1[:,i] = np.array([[np.random.uniform(-100,100)],[np.random.uniform(-100,100)]])[0]     #random( 54,32)
#         self.user_loc1 = user_loc1   
#     def generete_user_loc2(self):   #rand(Num_User,2)*lengA;
#        user_loc2 = np.zeros((2,self.K))
#        for j in range(self.K):
#            user_loc2[:,j] = np.array([[np.random.uniform(-100,100)],[np.random.uniform(-100,100)]])[0]     #random( 54,32)
#        self.user_loc2 = user_loc2               

# class RIS_D(object):               
#     def __init__(self, M, N, K,L, P_B, P_K, P_J, P_R,u,awgn_var,power_properties,phase_properties,UEs ):        
#         self.M = M        # BS station ante
#         self.N = N        # IRS element N_l
#         self.K = K         # user  
#         self.L= L           # no of irs       
# # power
#         self.u= u       #1/v = .`W11/0.8
#         self.P_B= P_B
#         self.P_K= P_K
#         self.P_J= P_J
#         self.P_R= P_R         #x.N.P_r 
#         self.awgn_var = awgn_var
        
#         self.power_properties = power_properties   #
#         self.phase_properties = phase_properties
        
#         self.P_max = np.max(power_properties)     #100Watt   Power_normal   // P_max
#         self.Phase_max= np.max(phase_properties)     # phase normali
#       # self.x=x              #x_l  [0,1]        
#  # action
#         self.theta = np.random.rand(self.N, self.L)*2*np.pi                           #np.random.rand(self.N)*2*np.pi                      
#         self.Phi = np.zeros(self.N, dtype=complex)                                #np.zeros(self.M,self.K   dtype= complex) 
#         self.w = np.zeros((self.M, self.K), dtype= complex) #BF vector for user K
#         self.x = np.zeros(self.L)              
#         self.UEs = UEs
        
#         # self.h_nlos = np.random.randn(self.N)+(1j*(np.random.randn(self.N)))     #IRS_USER  ##  # fix # NLOS
#         # self.h_random = np.random.randn(self.K)+(1j*(np.random.randn(self.K)))  # BS_USer      #   fix   ### NLOS ###########################**************
#         #self.squire_array= self.get_squire_array()
#         self.d_1, self.d_2 = self.get_dis() 
#         self.d_3, self.d_4 = self.get_dis_IUS()     #IRS_US
        
#         self.h1 = self.get_h_channel()
#         self.h2 = self.get_h_channel()
#         self.F= self.get_channel()
#         self.G= self.get_channel()
        
# #        self.state = self.reset_state()
    
#         self.action_dim = (self.M*self.K  + (self.N*self.L)+(self.L)+1)          # theta(N) + w(M,1) + X((0,1 ))/theta(matrix) + W(vector) + x(vector)      
#         self.state_dim =   2*(self.M*self.K ) #+2*(self.M*self.J )
 
# #        self.min_action, self.max_action = self.get_bound_action() 
#     def get_squire_array(self):  #h_LOS   # irs ref ele
#         squire_array = np.zeros(self.N,dtype=np.complex)  # irs   #(4,)
#         for i in range(self.N):
#             squire_array[i] = np.exp(-1j*2*np.pi*i*0.5*np.cos(np.random.rand(1)*2*np.pi))   # complex array (0j) ==  (1,)            
#         return squire_array        
# #distance
#     def get_dis(self):       ## BS_IRS   # BS_USer_1
#         d_1 = np.zeros((self.L),dtype=np.float32)               #BS_IRS   #(L,N,M)  .....L ka for loop
#         d_2 = np.zeros((self.K),dtype=np.float32)               #BS_US_1 (1,M)   ...no for loop        
#         #d_3 = np.zeros((self.N,self.M),dtype=np.float32)       #IRS_US
#         for i in range(self.L):
#             d_1[i]= np.sqrt((self.UEs.irs_loc[0,i]-self.UEs.bs_loc[0])**2 + 
#                                  (self.UEs.irs_loc[1,i]-self.UEs.bs_loc[1])**2 )   # + (self.UEs.irs_loc[2,i]-self.UEs.bs_loc[2])**2)
#         for i0 in range(self.K):                
#             d_2[i0] = np.sqrt((self.UEs.bs_loc[0]-self.UEs.user_loc1[0,i0])**2 + (self.UEs.bs_loc[1]-self.UEs.user_loc1[1,i0])**2 )# + (self.UEs.bs_loc[2]-self.UEs.user_loc[2,i0])**2)                
#         return d_1, d_2    #122,104
           
#     def get_dis_IUS(self):    #   IRS_ US_1,      US_2
#         d_3 = np.zeros((self.L,self.K),dtype=np.float32)   #Nk jagah pe--K hoga   KLN...KL ka for loop 
#         d_4 = np.zeros((self.L,self.K),dtype=np.float32)   #Nk jagah pe--K hoga   KLN...KL ka for loop
#         for i in range(self.L):
#             for i0 in range(self.K):
#                 d_3[i,i0] = np.sqrt((self.UEs.irs_loc[0,i]-self.UEs.user_loc1[0,i0])**2 + (self.UEs.irs_loc[1,i]-self.UEs.user_loc1[1,i0])**2 )  #+ (self.UEs.irs_loc[2,i]-self.UEs.user_loc[2,i0])**2)                    
#                 d_4[i,i0] = np.sqrt((self.UEs.irs_loc[0,i]-self.UEs.user_loc2[0,i0])**2 + (self.UEs.irs_loc[1,i]-self.UEs.user_loc2[1,i0])**2 )                        
#         return d_3 ,d_4   #235
    
# #/////// chanel 
# # BS_IRS //  BS_USER_1  
#     def get_channel(self):  # BS_IRS  //  BS_USER     
#         G = np.zeros((self.N,self.M,self.L),dtype=np.complex)   # BS_IRS   L
#         F = np.zeros((self.M,self.K),dtype=np.complex)    #BS_USER  K
        
#         for i in range(self.L):  #L=8     (4,2)  2D     
#             G[:,:,i] = (np.random.randn(self.N,self.M) + 1j*np.random.randn(self.N,self.M))*(np.power(10,-3.53)/(np.power(self.d_1[i],0.9)))*np.sqrt(0.5) #(4, 8)
            
#         for i0 in range(self.K):  
#             F[:,i0] = (np.random.randn(self.M) + 1j*(np.random.randn(self.M)))*(np.power(10,-3.53)/(np.power(self.d_2[i0], 3.76)))*np.sqrt(0.5) #K /(8,)               
#         return G, F  
    
# # IRS_USER_1, IRS_US_2                                                  
#     def get_h_channel(self):
#         h1 = np.zeros((self.N,self.K,self.L),dtype=complex)
#         h2=np.zeros((self.N,self.K,self.L),dtype=complex)
        
        
#         for i in range(self.L):
#             for i0 in range(self.K):
#                 h1[:,i0,i] = (np.random.randn(self.N) + 1j*(np.random.randn(self.N)))*(np.power(10,-3.53)/(np.power(self.d_3[i,i0],3.76)))*np.sqrt(0.5) #(4,)        
#                 h2[:,i0 ,i]=(np.random.randn(self.N) + 1j*(np.random.randn(self.N)))*(np.power(10,-3.53)/(np.power(self.d_4[i,i0],3.76)))*np.sqrt(0.5)
#         return h1, h2                            
                
#     def reset_state(self):       
#         G,F = self.get_channel()       # L # K
#         h1,h2= self.get_h_channel()
#         theta = np.random.rand(self.N,self.L)*2*np.pi          #(4,4)                             
#         phi = np.zeros((self.N,self.N,self.L) ,dtype=complex)         
#         for i in range(self.L):
#             phi[:,:,i] = np.diag(np.exp(1j*theta[:,i]))                               
#         x = np.zeros(self.L ) #, dtype=int)                                                                  ## int part I added 10 jan
#         for i in range(self.L):     #initiate xl
#             x[i]= np.random.randint(0,2)       #@@@@    (100,)
            
#         s_1 = np.zeros((self.M,self.K), dtype= np.complex)    ##2 dec///       ##  it will update at every episode 
#         s_2 = np.zeros((self.M,self.K), dtype= np.complex)
#         for i in range(self.K): 
#             s1= np.transpose(np.conjugate( F[:,i]))  #   .conj().T      #hkl_H   wk 
#             for i0 in  range(self.L):
#                 s1 += np.matmul(np.matmul(x[i0]*h1[:,i,i0].conj().T,phi[:,:,i0]),G[:,:,i0]) #(8,)                   
#             s_1[:,i] = s1      #(4,)
#             s2 = np.matmul(np.matmul(x[i0]*h2[:,i,i0].conj().T,phi[:,:,i0]),G[:,:,i0]) #(8,)                   
#             s_2[:,i] = s2 
            
#         s = np.add(s1, s2)    # to add array            
#         s_reshape = s.reshape(-1,1)                 # (16, 1)  only column me                
#         state = np.concatenate((np.real(s_reshape),np.imag(s_reshape)), axis=0)     #(32, 1)
#         return state[:,0]         #(32),)   
    
#     def extarct_action(self,action):
#         #` power_BS= np.zeros((self.L) ,dtype=np.float32)  # BS power 
#         x = np.zeros(self.L )                     #x =L     (4,)                                                     # I add dtyep 12 jan
#         theta_w1 = np.zeros((self.M*self.K))       # w= M,K     exp(j(theta))`
#         theta_w2 = np.zeros((self.M*self.K))       # w= M,J     exp(j(theta))`        
#         theta_irs = np.zeros(self.N*self.L)   # theta (irs) (0,2*pi)  (16,)
        
#         for i in range(len(action)):
#             if i < 1:
#                 power_BS = action[i]*self.P_max    #  BS power 
#             elif i < self.L+1:  #x
#                 x[i-1] = int(np.around(action[i]))                #          bef==int(np.around(action[i]))====i remove int 
#             elif i < (self.M*self.K)+self.L+1:   #w=exp(j(theta))               
#                 theta_w1[i-self.L-1] = action[i]*2*np.pi   #0=0.6524216
                
#             elif i < ((self.M*self.K)+(self.M*self.K)+self.L+1):   #w=exp(j(theta))               
#                 theta_w2[i-self.M*self.K-self.L-1] = action[i]*2*np.pi   #0=0.6524216                            
#             else:                                
#                 theta_irs[i-(self.M*self.K) -self.L-1] = np.exp(1j*action[i]*2*np.pi)     #theta    phase_normali//power_max
#         return power_BS, x ,theta_w1, theta_w2, theta_irs
                
    
#     def step(self, action):           #(41,) 
#        #` Power_BS= np.zeros((self.L) ,dtype=np.float32)  # BS power   
#         power_BS, x, theta_w1, theta_w2,theta_irs= self.extarct_action(action)
#         W1 = np.zeros((self.M,self.K),dtype=np.complex)              #,dtype = int)    
#         W2= np.zeros((self.M,self.K),dtype=np.complex)              #,dtype = int)    
#         cntr1 = 0  
#         cntr2 = 0 
#         for i in range(self.M):
#             for i0 in range(self.K):                                
#                 W1[i,i0] = np.exp(1j*theta_w1[cntr1]) 
#                 cntr1 += 1             
#                 W2[i,i0] = np.exp(1j*theta_w2[cntr2]) 
#                 cntr2 += 1                                  
#         W1 = (W1/np.linalg.norm(W1))*power_BS                      # BS Power (8,2)    ///  norm==||w||
#         W2 = (W2/np.linalg.norm(W2))*power_BS     
        
#         theta_irs = theta_irs.reshape((self.N,self.L))     #np.diag(theta_irs) (4,4)
#         phi = np.zeros((self.N,self.N,self.L))
#         for i in range(self.L):
#             phi[:,:,i] = np.diag(theta_irs[:,i])
#         # for i in range(self.L):            
#         #      phi[:,i] = np.diag(np.exp(1j*phi[:,i])) 
# ###### ###////////////////////////// for 1
#         G,F = self.get_channel()   #L # K
#         h1,h2 = self.get_h_channel()   # KL
#         sr = np.zeros((self.M,self.K), dtype= np.complex)   #s  ==sr+st
#         sinr1 = np.zeros((self.K),dtype=np.float32)   #LK
#         rate1 = np.zeros((self.K),dtype=np.float32)     #LK
#       ##  P = np.zeros((self.K),dtype=np.float32)
# #///////////////////////   for 1       
#         for i in range(self.K):                        
#             s1= F[:,i].conj().T    #(1, 8)          
#             for i0 in  range(self.L):                
#                 s1 += np.matmul(np.matmul(x[i0]*h1[:,i,i0].conj().T,phi[:,:,i0]),G[:,:,i0])     #phi[:,i0] /// (4,)          
#                 sr[:,i] = s1   #(8,)  
#                 num_ = np.matmul(sr[:,i],W1[:,i])    #(8,) (8,)                                                     #before==np.matmul(s1,W[:,i]).....Ichange == 16 jan np.matmul(s[:,i],W[:,i])
#                 num = np.power(np.abs(num_),2)   # 156
                
#                 denum= self.awgn_var
#                 for i1 in range(self.K):                    
#                     if i1 != i:                        
#                         s2 = F[:,i].conj().T    #(1, 4)   
#                         for i2 in range(self.L):                            
#                             s2 += np.matmul(np.matmul((x[i2]*h1[:,i,i2].conj().T),phi[:,:,i2]),G[:,:,i2]) #phi[:,i2]//(4,)
#                             sr[:,i]=s2
#                         s_4 = np.matmul(sr[:,i],W1[:,i1])     # (1,4),(4,)== (1,)           # before==np.matmul(s2,W[:,i1]) 
#                         denum+= np.power(np.abs(s_4),2)  # 22.00
#                 sinr1[i] = num/(denum)  #int 0.823
#                 rate1[i] = np.log2(1+sinr1[i])           #denum += self.awgn_var 
               
# #//////////// //////////////for 2       
#        # s = np.zeros((self.M,self.J,self.K), dtype= np.complex) 
#         sinr2= np.zeros((self.K),dtype=np.float32)   #LK
#         rate2 = np.zeros((self.K),dtype=np.float32)    
#         st = np.zeros((self.M,self.K), dtype= np.complex)               
#         for j in range(self.K):                                           
#             for j0 in  range(self.L):                
#                 s2= np.matmul(np.matmul(x[j0]*h2[:,j,j0].conj().T,phi[:,:,j0]),G[:,:,j0])     #phi[:,i0] /// (4,)          
#                 st[:,j] = s2   #(8,)  
#                 num_ = np.matmul(st[:,j],W2[:,j])    #(8,) (8,)                                                     #before==np.matmul(s1,W[:,i]).....Ichange == 16 jan np.matmul(s[:,i],W[:,i])
#                 num = np.power(np.abs(num_),2)   # 156       
                
#                 denum= self.awgn_var
#                 for j1 in range(self.K):                    
#                     if j1 != i:                        
#                         for j2 in range(self.L):                            
#                             s2 = np.matmul(np.matmul((x[j2]*h2[:,j,j2].conj().T),phi[:,:,j2]),G[:,:,j2]) #phi[:,i2]//(4,)
#                             st[:,j]=s2
#                         s_4 = np.matmul(st[:,j],W2[:,j1])     # (1,4),(4,)== (1,)           # before==np.matmul(s2,W[:,i1]) 
#                         denum+= np.power(np.abs(s_4),2)  # 22.00
#                         #denum += self.awgn_var
                        
#                 sinr2[j] = num/(denum)  #int 0.823
#                 rate2[j] = np.log2(1+sinr2[j])       #0.86  sinr[i]...maine i remove ki 
        
#         s = np.add(sr,st)    # state 
#         rate= np.sum(rate1+rate2)                       
#                          ## P[i] = np.linalg.norm(W)                 
# # power:      
    
#         P_1=0
#         for K0 in range(self.K):
#             P_11=np.abs(np.matmul(W1[:,K0], (np.transpose(np.conjugate(W1[:,K0]))))) *self.u   # power of bs     //power_bs  /////power_BS*self.u            
#         for J0 in range(self.K):
#             P_12=np.abs(np.matmul(W2[:,K0], (np.transpose(np.conjugate(W2[:,K0]))))) *self.u   # power of bs     //power_bs  /////power_BS*self.u    
#         P_1+= P_11+P_12
        
#         P_3=0         
#         for i in range(self.K):
#             P_31 = self.P_K 
#         for j in range(self.K):
#             P_32 = self.P_J
#         P_3+= P_31+ P_32
                        
#         P_4 = 0     
#         for l in range (self.L):                                                          
#             P_4+= (x[l]*self.N)*self.P_R    # 40   x[L0]=0    
                     
#         P_t = np.sum( P_1+self.P_B + P_3 + P_4)  # 54
        
#        # Power_BS[i] = np.linalg.norm(W[:,0,:,i])                    
#         reward =rate/ P_t       #pen if W_h.W<= P_max  //(np.sum(power_BS)*self.u+ self.P_B+self.P_K+P_4 )                                                       #  before==sum(rate)/P_T /// after==  16 jansum(rate[i])/P_T                                                   # np.sum(rate)/(P_t) 
#         s1_reshape = s.reshape(-1,1)   #(4, 1)
#         new_state = np.concatenate((np.real(s1_reshape),np.imag(s1_reshape)),axis=0)     #(8, 1)                                                            #np.sum(rate) /(P_t )             
#         return reward, new_state[:,0]          #(32,)   Power_BS
    
   
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
   
    
    
   
    # def get_bound_action(self):     ##  tahn h network me final layer pr rkhenge 
    #     max_action = np.zeros(((2*self.L)+(self.K*self.L)+self.N),dtype=np.float32)
    #     min_action = np.zeros_like(max_action,dtype=np.float32)
    #     for i in range(len(max_action)):
    #         if i < 2*self.L:
    #             max_action[i] = 1
    #             min_action[i] = -1
    #         else:
    #             max_action[i] = 1
    #             min_action[i] = 0        
    #     return min_action, max_action
        

















# def get_bound_action(self):

#         max_action = np.zeros((self.action_dim),dtype=np.float32)
#         min_action = np.zeros_like(max_action,dtype=np.float32)
#         for i in range(len(max_action)):
#             max_action[i] = 1
#             min_action[i] = 0
        
#         return min_action, max_action


































# class UEs_and_utilities():         location
    
#     def __init__(self,M,N,K,L):
#         self.K = K
#         self.L = L
#         self.bs_loc = np.array([[0],[0]])
#         self.generate_irs_loc()
#         self.generate_user_loc()
      
#     def generate_irs_loc(self):
#         irs_loc = np.zeros((3,L),dtype = float)
#         for i in range(self.L):
#             irs_loc[:,i] = np.array([[np.cos(2*i*np.pi/self.L)*100],[np.sin(2*i*np.pi/self.L)*100],[50]])
#         self.irs_loc = irs_loc
        
#     def generete_user_loc(self):
#         user_loc = np.zeros((2,self.K))
#         for i in range(self.K):
#             user_loc[:,i] = np.array([[np.random.uniform(-100,100)],[np.random.uniform(-100,100)]])
#         self.user_loc = user_loc






















         
            

       
            
       
        
       
        
            


            
            
                

                                        
                    
        #  return self.state, reward, done,                                                                               
          
        #    theta1 = np.random.rand(self.N)*2*np.pi                  
            #  w1 = np.zeros(self.M, dtype = np.complex)  
          
          # x1 = np.zeros(self.L)
          # for i in range(self.L):    
          #     x1[i]=  np.random.rand(0,2)#@@@@@
                                  
          # for a0 in range (len(action)):
          #     if a0< self.N:                  
          #         theta1[a0]= action[i]*np.pi*2             #np.exp(1j*(action[i]*self.phase_normalize))                
          #     elif a0< self.M:
          #           w1[a0- self.M]= action [i]*self.M
          #     else:
          #        x1[a0]=  action [i]                                                                              # ///////////////////////
                                       
          # theta = np.random.rand(self.N)*2*np.pi         
          # w = np.zeros(self.M ,dtype = np.complex)    
          # x = np.random.rand(0,2)  
          # cntr = 0                                                                                                                                              
                                 
'''def _compute_reward(self):
        
        
         
         reward= 0
         s1 = np.zeros((self.M,self.K,self.L), dtype= complex)           # state 
         sinr = np.zeros((self.K,self.L),dtype=np.float32)
         Rate = np.zeros((self.K,self.L),dtype=np.float32)                            
        
         theta = np.random.rand(self.N,self.L)*2*np.pi         #initiate theta
         phi = np.zeros(self.N,self.N,self.L,dtype=complex)
         for i in range(self.L):
             phi[:,:,i] = np.diag(np.exp(1j*theta[:,i]))         #@@@@@@@                       
                                                  
         x = np.zeros(self.L)
         for i in range(self.L):     #initiate xl
             x[i]= np.random.rand(0,2)         #@@@@@@@@
          
         numerator=0    
         for i00 in range(self.K): 
            s2= self.g[:,:,i00].conj().T
            w_k = self.w[i00]
            for i0 in  range(self.L):
              s2 += np.matmul(np.matmul((x[i0]*self.h[:,i00,i0].conj().T),phi[:,:,i0]),self.G[:,:,i0])
              numerator += np.power(np.abs(np.matmul(s1,w_k)),2) #@@@@@@@@
             
              denumrator = self.awgn_var
              w=  np.zeros(self.M,  dtype= complex)
              for i in range(self.K):                                          #  user k=1,2,.....K                  
                  if i != i00:                    
                     w2 = w[i]
                     for k0 in range(self.K):         # gk_H,, hk_H
                        s_1= self.g[:,:,k0].conj().T                                             
                        for l0 in range(self.L):                                                   
                         s_1 += np.matmul(np.matmul((x[l0]*self.h[:,k0,l0].conj().T),phi[:,:,l0]),self.G[:,:,l0]) 
                         denumrator +=  np.power(np.abs(np.matmul(s_1,w2)),2)  
                         
             
         sinr[i0,i00]= numerator/denumrator         
         Rate[i0,i00] =   np.log2(1+sinr[i0,i00])                             # np.log(1+sinr)/np.log(2)         
                         
        #power: 
         P_t=0 
         
         for K00 in range(self.K):
             w_k= self.w[K00]
             w_kh= self.w[K00].conj().T
             P_1 = np.matmul (np.matmul(w_k, w_kh),self.u)   #Pmax= w.w_H
                      
         x = np.zeros(self.L)      
         for L0 in range (self.L) :                     
           x[L0] = np.random.rand(0,2)
          # N[L0]= np.zeros(self.L)===================           
           P_4 = np.matmul(np.matmul(x[L0],self.N) , self.P_R)    # Nl
                     
           P_t += P_1 + self.P_B + self.P_K + P_4
           
         EE = np.sum (Rate) / (P_t )         
         reward +=  EE  
         s1_reshape = s1.reshape(-1,1)
         new_state = np.concatenate((np.real(s1_reshape),np.imag(s1_reshape)),axis=0)
                 
         return reward , new_state[:,0], Rate'''
     
        
     
     
        
#     #power: ////////////////======================   trial    =================================
#         P_t=0 
         
#         P_1 = 0
#         for K00 in range(self.K):                    
#             w_k= self.w[:,K00]
#             w_kh= self.w[:,K00].conj().T
#             P_1 += np.matmul(w_k, w_kh) *self.u   #Pmax= w.w_H    #same problem likle phi //////////////////
                      
#         P_4 = 0     
#         for L0 in range (self.L) :                                      
#             P_4 += np.matmul(np.matmul(x[L0],self.N) , self.P_R)    # Nl   #x[L0]/////////////////////////
                     
#             P_t += P_1 + self.P_B + self.P_K + P_4
# EE= np.sum(rate)/(P_t)                                                                     # np.sum(rate)/(P_t) 
# reward += EE 
# s1_reshape = s1.reshape(-1,1)
# new_state = np.concatenate((np.real(s1_reshape),np.imag(s1_reshape)),axis=0)                                                               #np.sum(rate) /(P_t ) 
    
# return reward, new_state[:,0], rate          #, new_state[:,0], Rate
    
    
  
        
     
        
     
        
        
     
 # v = asanyarray(v)
 # s = v.shape
 # if len(s) == 1:
 #     n = s[0]+abs(k)
 #     res = zeros((n, n), v.dtype)
 #     if k >= 0:
 #         i = k
 #     else:
 #         i = (-k) * n
 #     res[:n-k].flat[i::n+1] = v
 #     return res
 # elif len(s) == 2:
 #     return diagonal(v, k)
 # else:
 #     raise ValueError("Input must be 1- or 2-d.")
       
     
        
     
           
       
             
             
  
             
             
             
             
             
                     
                 
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
              
             
             
             
'''def _compute_reward(self):              # EE /rate/=======  SE  ================= 
     
     reward= 0
     s1 = np.zeros((self.M,self.K,self.L), dtype= complex)           # state 
     sinr = np.zeros((self.K,self.L),dtype=np.float32)
     Rate = np.zeros((self.K,self.L),dtype=np.float32)                            
    
     theta = np.random.rand(self.N,self.L)*2*np.pi         #initiate theta
     phi = np.zeros(self.N,self.N,self.L,dtype=complex)
     for i in range(self.L):
         phi[:,:,i] = np.diag(np.exp(1j*theta[:,i]))         #@@@@@@@                       
                                              
     x = np.zeros(self.L)
     for i in range(self.L):     #initiate xl
         x[i]= np.random.rand(0,2)         #@@@@@@@@
      
     numerator=0    
     for i00 in range(self.K): 
        s1= self.g[:,:,i00].conj().T
        w_k = self.w[i00]
        for i0 in  range(self.L):
          s1 += np.matmul(np.matmul((x[i0]*self.h[:,i00,i0].conj().T),phi[:,:,i0]),self.G[:,:,i0])
          numerator += np.power(np.abs(np.matmul(s1,w_k)),2) #@@@@@@@@
         
          denumrator = self.awgn_var
          w=  np.zeros(self.M,  dtype= complex)
          for i in range(self.K):                                          #  user k=1,2,.....K                  
              if i != i00:                    
                 w2 = w[i]
                 for k0 in range(self.K):         # gk_H,, hk_H
                    s_1= self.g[:,:,k0].conj().T                                             
                    for l0 in range(self.L):                                                   
                     s_1 += np.matmul(np.matmul((x[l0]*self.h[:,k0,l0].conj().T),phi[:,:,l0]),self.G[:,:,l0]) 
                     denumrator +=  np.power(np.abs(np.matmul(s_1,w2)),2)          
         
     sinr[i0,i00]= numerator/denumrator         
     Rate[i0,i00] =   np.log2(1+sinr[i0,i00])                             # np.log(1+sinr)/np.log(2)         
                     
    #power: 
     P_t=0 
     
     for K00 in range(self.K):
         w_k= self.w[K00]
         w_kh= self.w[K00].conj().T
         P_1 = np.matmul (np.matmul(w_k, w_kh),self.u)   #Pmax= w.w_H
                  
     x = np.zeros(self.L)      
     for L0 in range (self.L) :                     
       x[L0] = np.random.rand(0,2)
      # N[L0]= np.zeros(self.L)===================           
       P_4 = np.matmul(np.matmul(x[L0],self.N) , self.P_R)    # Nl
                 
       P_t += P_1 + self.P_B + self.P_K + P_4
       
     EE = np.sum (Rate) / (P_t )         
     reward +=  EE  
     s1_reshape = s1.reshape(-1,1)
     new_state = np.concatenate((np.real(s1_reshape),np.imag(s1_reshape)),axis=0)
             
     return reward , new_state[:,0], Rate
                        
                     
                        
                     
                        
                     
                        
                 
                 S1 += np.matmul(np.matmul((x[l]*self.h[:,:,l].conj().T),phi[:,:,l]),self.G[:,:,l])
             S[:,k] = S1
             numerator += np.power(np.abs(np.matmul(S1,w1)),2) 
             
             denumrator = self.awgn_var
             for k0 in range(self.K):
                 if k0 != k:
                     a1= self.g[:,:,k0].conj().T 
                     w2 = w[k0]                      
                     for l0 in range(self.L):                                                   
                       s_1= np.matmul(np.matmul((x[l0]*self.h[:,l,l0].conj().T),phi[:,k0,l0]),self.G[:,k0,l0]) 
                       denumrator +=  np.power(np.abs(np.matmul(s_1,w2)),2)          
             
         sinr[i]= numerator/denumrator         
         Rate[i] = np.log(1+sinr[i])/np.log(2)
                 
        
        
         P_4=0                     
         for K0 in range(self.K):
             w_k=w[k0]
             w_kh= self.w[k0].conj().T
             P_1= np.matmul (np.matmul(w_k, w_kh),u)
             
         reward=0   
         x = np.zeros(self.L)      
         for L0 in range (self.L) :                     
           x[L0] = np.random.rand(0,2)
           
           P_4 += np.matmul(np.matmul(x[L0],self.N) , P_R)
                     
           P_t= P_1+ self.P_B + self.P_k + P_4
           
           EE = self.Rate/P_t            
         reward +=   EE           
   return reward   

       
   for i in range(self.K): 
             s1= self.g[:,:,i].conj().T
             for i0 in  range(self.L):
                 s1 += np.matmul(np.matmul((x[i0]*self.h[:,i,i0].conj().T),phi[:,:,i0]),self.G[:,:,i0])
             s[:,i] = s1    '''   
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
        
       
       
        
        
             
            
             
             
             
             
             
             
             
             
             
             
             
             
             
         

       
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
         
                 
       
                   
                   
                   
               
           

           
             
           
            
           
            
           
            
           
            
           
            
           
            
           
         
         
        
''' for i in range(self.L): 
             h2=h[:,i]
              
             S2[i,:]= np.matmul(np.matmul(np.matmul(self.x[i],self.h2.conj().T ),phi),self.G[:,i]) 
             S3 [i,:]= np.add(S1+S2) 
             S4= np.matmul(S3,self.w)                                #/                   ////////////////////////////////////
             
        #     numerator + = np.power(np.abs(S4),2)
             numerator = np.power(np.abs(S4),2)
                        
                
             denumrator = self.awgn_var   ## interference + Noise             
             interference=0
             for k in range(self.K):
                 if n != k:
                     h4= h[:,k]
                     
                     S4= self.g1.conj().T 
                     S5= np.matmul(np.matmul(np.matmul(self.x[i],self.h4.conj().T ),phi),self.G[:,k]) 
                     S6= np.add(S4+S5)
                     S7= np.matmul(S6,self.w)
                     denumrator  =  np.power(np.abs(S7),2)          
             
         sinr[i]= numerator/denumrator         
         Rate[i] = np.log(1+sinr[i])/np.log(2)
                 
        
         p1= 0 
         P_4=0
         
         for i in range(self.K):
             w1=w[:,i]
             if (self.w.conj().T.self.w. <= self.P_max): 
                 
                 p1 = self.u*(self.w, self.w.conj().T)
                 P_B= P_B
                 P_k= P_k
                 P_R= P_R  
                 
                 for i0 in range (self.L)                                 
                 P_4 + = self.x,self.N,P_R
                 
            P_t= P1+ self.P_B + self.P_k + P_4
                 
                 EE = self.Rate/P_t 
                 
                 reward + =   EE
                 
           return reward '''    
                 















# =============================================================================
# K=4
# M=4
# state = np.zeros((K,M), dtype= np.complex)
# for in in range (K):
#     w=state[i,:]
#     u=state[:,i]
#     t=state[i]
#     u=state[:,0]
#     p=state[0,:]
# =============================================================================
# ==============================//     single dimention me change krne k liye '///    =======================
# K=4
# M=4
# L=4
# state = np.zeros((K,M), dtype= np.complex)   
# 
# for i in range(K):
#         q=state[i,:] 
#         print(q)
# =============================================================================
'''def reset_state(self):           # g_hermetian + x / h_hermitian./phi /.G  ============ by ali==============

    theta = np.random.rand(self.N,self.L)*2*np.pi #initiate theta
    phi = np.zeros(self.N,self.N,self.L,dtype=complex)
    for i in range(self.L):
        phi[:,:,i] = np.diag(np.exp(1j*theta[:,i]))
        
    self.G= np.random.rand(self.N, self.M) +(1j * (np.random.rand(self.N, self.M)))                            
    self.h= np.random.rand(1, self.N) +(1j * (np.random.rand(1, self.N)))      

    s = np.zeros((self.M,self.K), dtype= np.complex)    ##2 dec///////////////
                                    ##  it will update at every episode 
    x = np.zeros(self.L)
    for i in range(self.L):     #initiate xl
        x[i]= np.random.rand(0,2)
        
    for i in range(self.K): 
        s1= self.g[:,:,i].conj().T
        for i0 in  range(self.L):
            s1 += np.matmul(np.matmul((x[i0]*self.h[:,i,i0].conj().T),phi[:,:,i0]),self.G[:,:,i0])
        s[:,i] = s1          
    s_reshape = s.reshape(-1,1)                 #  only column me 
    state = np.concatenate((np.real(s_reshape),np.imag(s_reshape)), axis=0)   
    return state[:,0]

def step(self, action):             #================ by ali ==========================================      
  
      x = np.zeros(self.L)
      theta_w = np.zeros((self.M*self.K))
      theta_irs = np.zeros(self.N*self.L)
      
      
      for i in range(len(action)):
          if i < self.L:
              x[i] = action[i]
          elif i < (self.M*self.K)+self.L:
             theta_w[i-self.L] = action[i]
          else:
              theta_irs[i-(self.M*self.K) -self.L] = action[i]
      
      theta_irs.reshape((self.N,self.L))
      
      phi = np.zeros(self.N,self.N,self.L,dtype=complex)
      for i in range(self.L):
              phi[:,:,i] = np.diag(np.exp(1j*theta_irs[:,i])). #### maine diag ko htaya hai 11 dec ko ..maine irs[:,i]...ko bss irs[i ] me kiya hai
             
      W = np.zeros((self.M,self.K),dtype=complex)
      
'''

#++++++++++++++++++++++++++++++++++++++++   first full trial




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
               
        reward = sum(rate[i])/P_t'''
        
#============================================= channel model /////vincent  M109...ddpg vli file me    4/26/2022 ///////////////

# def Rayleigh_channel(self,M,BS,user):
#         distance=math.sqrt(((BS[0]-user[0])**2)+((BS[1]-user[1])**2)+(self.BS_height)**2)
#         gamma=(10**-3)*(distance)**-2 #large scale path loss
#         channel=(math.sqrt(gamma)*np.matrix(np.random.randn(M)+1j*np.random.randn(M)))
#         return channel
#     def Rician_channel2user(self,N,UAVIRS,user):
#         angle=np.random.uniform(0,np.pi*2)
#         eplison=10
#         distance=math.sqrt((UAVIRS[0]-user[0])**2+(UAVIRS[1]-user[1])**2+(self.UAVIRS_height)**2)
#         ak=(10**-3)*(distance)**-2
#         NLoS=np.array(np.random.randn(N)+1j*np.random.randn(N))
#         d=1/2 #antenna spacing
#         LoS=np.array([1,np.exp(1j*2*np.pi*d*math.sin(angle)),np.exp(1j*2*np.pi*d*2*math.sin(angle)),np.exp(1j*2*np.pi*d*3*math.sin(angle))])
#         channel=np.matrix((math.sqrt(ak/(eplison+1)))*(math.sqrt(eplison)*LoS+NLoS))
#         return channel
#     def Rician_channel2UAVIRS(self,M,N,UAVIRS,BS):
#         h=self.UAVIRS_height-self.BS_height
#         angle1=np.random.uniform(0,np.pi*2)
#         angle2=np.random.uniform(0,np.pi*2)
#         distance=math.sqrt((UAVIRS[0]-BS[0])**2+(UAVIRS[1]-BS[1])**2+(h)**2)
#         beta=(10**-3)*(distance**-2)
#         delta=1
#         d=1/2 #antenna spacing
#         NLoS=np.array(np.random.randn(M,N)+1j*np.random.randn(M,N))
#         LoS1=np.array([1,np.exp(1j*2*np.pi*d*math.sin(angle1)),np.exp(1j*2*np.pi*d*2*math.sin(angle1)),np.exp(1j*2*np.pi*d*3*math.sin(angle1))])
#         LoS2=np.array([1,np.exp(1j*2*np.pi*d*math.sin(angle2)),np.exp(1j*2*np.pi*d*2*math.sin(angle2)),np.exp(1j*2*np.pi*d*3*math.sin(angle2))])
#         LoS=np.matrix(np.matrix(LoS2).H*np.matrix(LoS1)).T
#         channel=math.sqrt(beta/(delta+1))*(math.sqrt(delta)*LoS+np.matrix(NLoS)).H
#         return channel