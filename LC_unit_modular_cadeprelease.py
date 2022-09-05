# LC differential equations for LC_unit from Alvarez & al 2002 paper- formulated for Euler-Murayama method
# no sK or Nav currents
# for modular simulations (+ lateral inhibition)
# Edit 23/3/20 for multiprocessing w map & pool functions.


import numpy as np



class LC_unit(object):
    def __init__(self, g_na, g_k, g_l, g_ld, g_p, g_ahp, g_ca, g_d, g, c_s, c_d, v_na, v_k, v_ca, v_l, v_ld, v_i, m,  h, n, p, q, r, m_ca, y, ca_init, chR2,caBindingFactor):
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
        self.alpha_func = alpha_func()
        self.scount = 0
        self.l_i_count = 0
        self.tvec = np.array([])

        self.refrac = [False, 0]
        self.i_e = np.random.uniform(low=-0.1, high = 0.1)
        self.rel_alpha = 0.01
        self.rel_beta = 0.1
        self.activeRelProtein = 0
        self.release_delay = -1
        self.a2_tau = 400
        self.releaseTimers = []
        self.releaseWeights = []
        self.chR2 = chR2
        self.caBindingFactor = caBindingFactor

    def update(self, i, gap_voltages,  dt, ttx_on, carb_on, led, g_a2_in):
        if self.caBindingFactor > 0:
            self.activeRelProtein += dt * self.rel_alpha * self.ca_conc**self.caBindingFactor * (1-self.activeRelProtein) - self.rel_beta*self.activeRelProtein
            releaseScalar = 1 * self.activeRelProtein
        else:
            releaseScalar = 0.3
        # Track spikes, initiate refractory period
        spike = False
        if self.refrac[1] < 0.001:
            self.refrac[0] = False
        if self.v_s > 20 and self.refrac[0] == False:  #if Vs > 20mV, and NOT in refractory period
            spike = True
            self.tvec = np.append(self.tvec, 0)   #start timer for this spike, goes up in 1s (representing 1x dt)
            self.scount += 1    # add one to total spike count
            self.refrac[0] = True  # True = in refractory period
            self.refrac[1] = 3   # start countdown for refractory period in ms
            self.release_delay = int(5/dt)  # noradrenaline release timer in ms
        if self.refrac[0]:
            self.refrac[1] += (-dt)   #if IN refractory period, subtract time step from refrac countdown
        self.tvec += 1

        # calculate g_a2
        self.g_a2 = 0
        self.release_delay += (-1)
        if self.release_delay == 0:
            print('NA release')
            self.releaseTimers.append(0)
            #print(self.releaseTimers)
            self.releaseWeights.append(releaseScalar)
            print(f'release weight= {releaseScalar}')
        idx=0
        for t in self.releaseTimers:
            w = self.releaseWeights[idx]
            self.g_a2 += self.releaseEvent(t, w)
            self.releaseTimers[idx] += dt
            idx+=1


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
        i_led = 0
        if led and self.chR2:
            i_led = 200
        #i_glia = 0.008 * (self.v_s-(-80))  # constant hyperpolarising influence from glia, as per alvarez-maubecin et al, 2000 - conductance = 1 gap junction as per Alvarex et al 2002
        i_mem = self.i_e + i_stan + i_p  +  i_ahp + i_ca  # Sum membrane currents

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
        g_a2_sum = np.sum(g_a2_in)
        i_a2 = g_a2_sum*(self.v_s - self.v_i)
        self.v_s += dt*((i+i_led) - i_mem - self.g_d*(self.v_s - self.v_d) - i_a2 + wnoise)/self.c_s
        gap_volt = self.gap_func(gap_voltages)   # calls separate function to calculate sum voltage differences over all connected units
        self.v_d += dt*(-self.g_ld * (self.v_d - self.v_ld) - self.g_d*(self.v_d - self.v_s) - gap_volt/ self.c_d)
        self.n += dt*(3 * (alpha_n_v * (1-self.n) - beta_n_v*self.n))   # K current
        self.h += dt*(3 * (alpha_h_v * (1-self.h) - beta_h_v*self.h))  # #Sodium current
        self.p +=  dt*(alpha_p_v * (1-self.p) - beta_p_v*self.p)   # Slow sodium


        self.ca_conc += dt*(-0.002*2*(self.v_s - self.v_ca)/(1+np.exp(-(self.v_s+25)/2.5)) - self.ca_conc/80 )
        self.y += dt*(0.75 * (y_inf_v - self.y)/tau_y_v)
        return  self.v_s, self.v_d, i_leak, i_k, i_na, i_p, i_ahp, i_ca, i_mem,  self.ca_conc, i_a2, self.g_a2, spike




    def gap_func(self, gap_voltages):   # function to calculate sum voltage difference via gap junctions
        self.vd_diffsum = 0   # voltage difference over gap
        for q in range(len(gap_voltages)):
            self.vd_diffsum += self.g*(self.v_d - gap_voltages[q])
        return self.vd_diffsum






    def dW(self,delta_t):   # noise function for Euler Murayama
        return np.random.normal(loc = 0.0, scale = 1)


    def releaseEvent(self,t, w):
        g_a2 = 10 * w * ((t-1)/self.a2_tau) * np.exp(-(t-1)/(self.a2_tau))
        return g_a2




def alpha_func():
    results = [0]
    t = 1
    t_i = 1
    while len(results) <= 1000000:
            # exponential decay function
        g_i =   ((t-t_i)/200) * np.exp(-(t-t_i)/(200))
        results.append(g_i)
        #print(g_i)
        t += 1
    return results
