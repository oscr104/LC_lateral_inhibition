''' LC unit class file w functions & differential equations from Alvarez & al 2002 paper- formulated for Euler-Murayama method
 3/4/2020 edits for single unit simulations, recieving pulse train of 'optostimulated' spikes, to model Ray Perrins data.

 NO GAP JUNCTION FUNCTIONALITY (V or [Ca])
 '''


import numpy as np



class LC_unit(object):
    def __init__(self, param_list):
        self.g_na = param_list['g_na']  # Specifies parameters to be stored in self
        self.g_k = param_list['g_k']
        self.g_l = param_list['g_l']
        self.g_ld = param_list['g_ld']
        self.g_p = param_list['g_p']
        self.g_ahp = param_list['g_ahp']
        self.g_ca = param_list['g_ca']
        self.g_d =  param_list['g_d']
        self.g_ca_gap = param_list['g']
        self.g = param_list['g']
        self.c_s = param_list['c_s']
        self.c_d = param_list['c_d']
        self.v_s = param_list['v_l']
        self.v_d = param_list['v_ld']
        self.v_na = param_list['v_na']
        self.v_k = param_list['v_k']
        self.v_ca = param_list['v_ca']
        self.v_l = param_list['v_l']
        self.v_ld = param_list['v_ld']
        self.v_i = param_list['v_i']
        self.m_inf = param_list['m']
        self.h = param_list['h']
        self.n = param_list['n']
        self.p = param_list['p']
        self.m_ca = param_list['m_ca']
        self.y = param_list['y']
        self.ca_conc = param_list['ca_conc']
        self.t = 1
        self.t_i = 1
        self.g_i = 0
        self.l_i = 0
        self.li_w = param_list['li_w']
        self.li_tau = param_list['li_tau']
        self.scount = 0
        self.l_i_count = 0
        self.tvec = np.array([])
        self.l_i_tvec = np.array([])
        self.refrac = [False, 0]
        self.alpha_trace = self.alpha_func( self.li_tau)



    def update(self, i, dt, ttx_on, carb_on, l_i_spk):
        # Track spikes, initiate refractory period
        spike = False
        if self.refrac[1] < 0.001:
            self.refrac[0] = False
        if self.v_s > 20 and self.refrac[0] == False:  #if Vs > 20mV, and NOT in refractory period
            spike = True
            self.tvec = np.append(self.tvec, 0)   #start timer for this spike, goes up in 1s (representing 1x dt)
            self.scount += 1    # add one to total spike count
            #print('spike count', self.scount)
            self.refrac[0] = True  # True = in refractory period
            self.refrac[1] = 3   # start countdown for refractory period in ms


        if self.refrac[0]:
            self.refrac[1] += (-dt)   #if IN refractory period, subtract time step from refrac countdown



        # auto inhibition
        self.g_i = 0
        if np.prod(self.tvec.shape)!= 0:  # if spikes have occured (.: tvec array will have entries)
            for spk in range(len(self.tvec)):    # for every spike logged in tvec
                index = int(round(self.tvec[spk])/1000)
                #self.g_i += self.alpha_func[index]    # cumulative add value from alpha function indexed at relevant time point for each spike


        # lateral inhibition
        li_num = np.sum(l_i_spk)
        self.l_i = 0
        if li_num > 0:    # number of spikes in opposing module l_i = lateral inhibition
            for spk in range(li_num):
                self.l_i_tvec = np.append(self.l_i_tvec,0) # start timer for effect of cross modular inhibition for this spike
                self.l_i_count += 1
        for spk in range(len(self.l_i_tvec)):
            index = int(round(self.l_i_tvec[spk]))
            if index < 80000:
                self.l_i +=  self.li_w * self.alpha_trace[index]  #0.3 good for spontaneous & evoked flip flopping
        self.g_i += self.l_i
        self.tvec += 1
        self.l_i_tvec += 1

        #Function to update values per each step in time
        # currents
        i_leak = self.g_l*(self.v_s-self.v_l)
        i_k = self.g_k*(self.n**4)*(self.v_s -self.v_k)
        if ttx_on == True:
            i_na =  0.005 *(self.m_inf**3)*self.h*(self.v_s - self.v_na) #5 = 10% sodium conductance 0.5% = 1% sodium conductance
            i_p = 0.00008*(self.p**2)*(self.v_s-self.v_na)   # Slow sodium current
        else:
            i_na =  self.g_na *(self.m_inf**3)*self.h*(self.v_s - self.v_na)
            i_p = self.g_p*(self.p**2)*(self.v_s-self.v_na)   # Slow sodium current
        i_stan = i_leak + i_na # Leak current + Potassium current + Sodium current
        #print(self.ca_conc)
        i_ahp = self.g_ahp*(self.ca_conc/(self.ca_conc+1)) * (self.v_s - self.v_k)  # afterhyperpolarisation current
        i_ca = self.g_ca*(self.m_ca)*self.y*(self.v_s-self.v_ca)
        i_a2 = self.g_i*(self.v_s - self.v_i)
        #i_glia = 0.008 * (self.v_s-(-80))  # constant hyperpolarising influence from glia, as per alvarez-maubecin et al, 2000 - conductance = 1 gap junction as per Alvarex et al 2002
        i_mem = i_stan + i_p  +  i_ahp + i_ca + i_a2 #+ i_glia   # Sum membrane currents

        # Formulae for m & variables (Na current)
        alpha_m_v = 0.1*(self.v_s+25.0)/(1-np.exp(-0.1*(self.v_s+25.0)))
        beta_m_v = 4.0*np.exp(-(self.v_s+50.0)/18.0)
        self.m_inf = alpha_m_v/(alpha_m_v + beta_m_v)
        #Formulae for n (K current) & h (Na current) variables
        alpha_n_v = 0.01*(self.v_s+34.0)/(1-np.exp(-.2*(self.v_s+34.0)))
        beta_n_v = 0.125*np.exp(-(self.v_s+40.0)/80.0)
        alpha_h_v = 0.07*np.exp(-(self.v_s+40.0)/20.0)
        beta_h_v = 1.0/(1.0+np.exp(-0.1*(self.v_s+14.0)))
        # Formulae for p variables (slow Na current)
        alpha_p_v = 0.0001*(self.v_s+30.0)/(1-np.exp(-0.05*(self.v_s+40.0)))
        beta_p_v = 0.004*np.exp(-(self.v_s+60.0)/18.0)
        #Formulae for Ca current variables
        m_ca = 1/(1+np.exp(-(self.v_s+ 55)/9))
        y_inf_v = 1/(1+np.exp((self.v_s+ 77)/5))
        tau_y_v = 20+100/(1+np.exp((self.v_s+76)/3))

        # Differentials
        wnoise = np.random.normal(loc = 0.0, scale = 1)
        #wnoise = 0
        self.v_s += dt*(i - i_mem - self.g_d*(self.v_s - self.v_d) + wnoise)/self.c_s
        self.v_d += dt*(-self.g_ld * (self.v_d - self.v_ld) - self.g_d*(self.v_d - self.v_s) / self.c_d)
        self.n += dt*(3 * (alpha_n_v * (1-self.n) - beta_n_v*self.n))   # K current
        self.h += dt*(3 * (alpha_h_v * (1-self.h) - beta_h_v*self.h))  # #Sodium current
        self.p +=  dt*(alpha_p_v * (1-self.p) - beta_p_v*self.p)   # Slow sodium
        #self.q += dt*(alpha_q_v * (1-self.q) - beta_q_v*self.q) # Slow potassium curr
        self.ca_conc += dt*(-0.002*2*(self.v_s - self.v_ca)/(1+np.exp(-(self.v_s+25)/2.5)) - self.ca_conc/80)
        self.y += dt*(0.75 * (y_inf_v - self.y)/tau_y_v)
        return  self.v_s, self.v_d, self.ca_conc, i_a2 , 0




    def alpha_func(self,li_tau):

        t = 1
        t_i = 1
        t1 = self.li_tau*10
        dt = 0.05
        tvec = np.linspace(1, t1, int(t1/dt))
        results=np.zeros(len(tvec))
        idx=0
        for t in tvec:
                # exponential decay function
            g_i = ((t-t_i)/(li_tau)) * np.exp(-(t-t_i)/(li_tau))
            results[idx]=g_i
            #print(g_i)
            idx += 1
        return results
