# LC differential equations for LC_unit from Alvarez & al 2002 paper- formulated for Euler-Murayama method
# no sK or Nav currents
# for modular simulations (+ lateral inhibition)
# Edit 23/3/20 for multiprocessing w map & pool functions.


import numpy as np



class LC_unit(object):
    def __init__(self, g_na, g_k, g_l, g_ld, g_p, g_ahp, g_ca, g_d, g, c_s, c_d, v_na, v_k, v_ca, v_l, v_ld, v_i, m,  h, n, p, q, r, m_ca, y, ca_init, li_w, rectifying_gap_junctions, autoInhibition):
        self.g_na = g_na   # Specifies parameters to be stored in self
        self.g_k = g_k
        self.g_d = g_d
        self.g_l = g_l
        self.g_ld = g_ld
        self.g_ahp = g_ahp
        self.g_ca = g_ca
        self.g_ca_gap = g
        self.g_p = g_p
        self.g = g
        self.li_w = li_w
        self.c_s = c_s
        self.c_d = c_d
        self.v_s = v_l
        self.v_d = v_l
        self.v_na = v_na
        self.v_k = v_k
        self.v_ca = v_ca
        self.v_l = v_l
        self.v_ld = v_ld
        self.v_i = v_i
        self.mv = 0.001
        self.ms = 0.001
        self.m_inf_v = m
        self.h = h
        self.n = n
        self.p = p
        self.q = q
        self.r = r
        self.m_ca_inf_v = m_ca
        self.y = y
        self.ca_conc = ca_init
        self.t = 1
        self.t_i = 1
        self.g_i = 0
        self.l_i = 0
        self.scount = 0
        self.l_i_count = 0
        self.tvec = np.array([])
        self.l_i_tvec = np.array([])
        self.refrac = [False, 0]
        self.i_e = np.random.uniform(low=-0.1, high = 0.1)
        self.rectifying_gap_junctions = rectifying_gap_junctions
        self.autoInhibition = autoInhibition

    def update(self, i, gap_voltages,  dt, ttx_on, carb_on, l_i_spk):
        # Track spikes, initiate refractory period
        spike = False
        if self.refrac[1] < 0.001:
            self.refrac[0] = False
        if self.v_s > 20 and self.refrac[0] == False:  #if Vs > 20mV, and NOT in refractory period
            #print('SPIKE')
            spike = True
            self.tvec = np.append(self.tvec, 0)   #start timer for this spike, goes up in 1s (representing 1x dt)
            #print(self.tvec)
            self.scount += 1    # add one to total spike count
            #print('spike count', self.scount)
            self.refrac[0] = True  # True = in refractory period
            self.refrac[1] = 3   # start countdown for refractory period in ms
        if self.refrac[0]:
            self.refrac[1] += (-dt)   #if IN refractory period, subtract time step from refrac countdown
        # auto inhibition
        self.g_i = 0
        i_aTwo=0
        if self.autoInhibition:
            # auto inhibition
            if np.prod(self.tvec.shape)!= 0:  # if spikes have occured (.: tvec array will have entries)
                for spk in range(len(self.tvec)):    # for every spike logged in tvec
                    t = self.tvec[spk]
                    self.g_i +=  self.alphafunc(t)    # cumulative add value from alpha function indexed at relevant time point for each spike
            i_aTwo = self.g_i * (self.v_s - self.v_k)

        # lateral inhibition
        li_num = np.sum(l_i_spk)
        self.l_i = 0
        if li_num > 0:    # number of spikes in opposing module l_i = lateral inhibition
            for spk in range(li_num):
                self.l_i_tvec = np.append(self.l_i_tvec,0) # start timer for effect of cross modular inhibition for this spike
                self.l_i_count += 1
        for spk in range(len(self.l_i_tvec)):
            t = self.tvec[spk]
            self.l_i += self.li_w*self.alpha_func(t)  #0.3 good for spontaneous & evoked flip flopping

        self.g_i += self.l_i
        self.g_i = 0
        self.tvec += 1
        self.l_i_tvec += 1

        #Function to update values per each step in time
        # currents
        i_leak = self.g_l*(self.v_s-self.v_l)
        i_k = self.g_k*(self.n**4)*(self.v_s -self.v_k)
        if ttx_on == True:
            i_na =  0.005 *(self.m_inf_v**3)*self.h*(self.v_s - self.v_na) #5 = 10% sodium conductance 0.5% = 1% sodium conductance
            i_p = 0.00008*self.p**2*(self.v_s-self.v_na)   # Slow sodium current
        else:
            i_na =  self.g_na *(self.m_inf_v**3)*self.h*(self.v_s - self.v_na)
            i_p = self.g_p*self.p**2*(self.v_s-self.v_na)   # Slow sodium current
        i_stan = i_leak + i_na + i_k # Leak current + Potassium current + Sodium current
        i_ahp = self.g_ahp*(self.ca_conc/(self.ca_conc+1)) * (self.v_s-self.v_k)  # afterhyperpolarisation current
        i_ca = self.g_ca*self.m_ca_inf_v**2*self.y*(self.v_s-self.v_ca)
        #i_glia = 0.008 * (self.v_s-(-80))  # constant hyperpolarising influence from glia, as per alvarez-maubecin et al, 2000 - conductance = 1 gap junction as per Alvarex et al 2002
        i_mem = self.i_e + i_stan + i_p  +  i_ahp + i_ca #+ i_glia   # Sum membrane currents

        # Formulae for m & variables (Na current)
        alpha_m_v = 0.1*(self.v_s+25.0)/(1-np.exp(-0.1*(self.v_s+25.0)))
        beta_m_v = 4.0*np.exp(-(self.v_s+50.0)/18.0)
        self.m_inf_v = alpha_m_v/(alpha_m_v + beta_m_v)
        #Formulae for n (K current) & h (Na current) variables
        alpha_n_v = 0.01*(self.v_s+34.0)/(1-np.exp(-.2*(self.v_s+34.0)))
        beta_n_v = 0.125*np.exp(-(self.v_s+40.0)/80.0)
        alpha_h_v = 0.07*np.exp(-(self.v_s+40.0)/20.0)
        beta_h_v = 1.0/(1.0+np.exp(-0.1*(self.v_s+14.0)))
        # Formulae for p variables (slow Na current)
        alpha_p_v = 0.0001*(self.v_s+30.0)/(1-np.exp(-0.05*(self.v_s+40.0)))
        beta_p_v = 0.004*np.exp(-(self.v_s+60.0)/18.0)
        #Formulae for Ca current variables
        m_ca_inf_v = 1/(1+np.exp(-(self.v_s+ 55)/9))
        y_inf_v = 1/(1+np.exp((self.v_s+ 77)/5))
        tau_y_v = 20+100/(1+np.exp((self.v_s+76)/3))

        # Differentials
        wnoise = self.dW(dt)

        self.v_s += dt*(i - i_mem - self.g_d*(self.v_s - self.v_d) - self.g_i*(self.v_s - self.v_i) + wnoise)/self.c_s
        gap_volt = self.gap_func(gap_voltages)   # calls separate function to calculate sum voltage differences over all connected units
        self.v_d += dt*(-self.g_ld * (self.v_d - self.v_ld) - self.g_d*(self.v_d - self.v_s) - gap_volt/ self.c_d)
        self.n += dt*(3 * (alpha_n_v * (1-self.n) - beta_n_v*self.n))   # K current
        self.h += dt*(3 * (alpha_h_v * (1-self.h) - beta_h_v*self.h))  # #Sodium current
        self.p +=  dt*(alpha_p_v * (1-self.p) - beta_p_v*self.p)   # Slow sodium


        self.ca_conc += dt*(-0.002*2*(self.v_s - self.v_ca)/(1+np.exp(-(self.v_s+25)/2.5)) - self.ca_conc/80 )
        self.y += dt*(0.75 * (y_inf_v - self.y)/tau_y_v)
        return  self.v_s, self.v_d, i_leak, i_k, i_na, i_p, i_ahp, i_ca, i_mem,  self.ca_conc, self.l_i, spike




    def gap_func(self, gap_voltages):   # function to calculate sum voltage difference via gap junctions
        self.vd_diffsum = 0   # voltage difference over gap
        for q in range(len(gap_voltages)):
            if self.rectifying_gap_junctions:
                if gap_voltages[q] > self.v_d:
                    self.vd_diffsum += self.g*(self.v_d - gap_voltages[q])
            else:
                self.vd_diffsum += self.g*(self.v_d - gap_voltages[q])
        return self.vd_diffsum






    def dW(self,delta_t):   # noise function for Euler Murayama
        return np.random.normal(loc = 0.0, scale = 1)





    def alphafunc(self, t):
        t_i = 1
        # exponential decay function
        g_i =  self.aTwoScalar *((t-0)/(self.aTwoTau)) * np.exp(-(t-0)/(self.aTwoTau))
        return g_i
