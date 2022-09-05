''' LC unit class file w functions & differential equations from Alvarez & al 2002 paper- formulated for Euler-Murayama method
1/12/2021 edits for PSAM transduction: when PSEM == True, triggers transient alpha-function describing PSAM channel activation by PSEM bolus

 NO GAP JUNCTION FUNCTIONALITY (V or [Ca])

 '''


import numpy as np



class LC_unit(object):
    def __init__(self, param_list):
        self.i_e = param_list['i_e']
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
        self.v_s = param_list['v_l'] + np.random.uniform(-2,2)
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
        self.release_delay = -1
        self.ca_tau = param_list['ca_tau']

        self.dt = param_list['dt']
        self.trans = param_list['trans']
        self.hyp = param_list['hyp']
        self.releaseTaus = []
        self.g_psam = 0
        self.auto_g_i = 0
        self.psem_clock = 0
        self.rel_alpha = 0.01
        self.rel_beta = 0.1
        self.releaseTimers = []
        self.releaseWeights = []
        self.caBindingFactor = 5
        self.activeRelProtein = 0
        self.releaseTaus=[]
        self.gcamp = 0
        self.gcampTimers=[]



    def update(self,  dt, ttx_on,  gap_vs, lat_g_i, psem, g_syn):

        i_e = self.i_e
        if self.hyp == True:
            i_e += -0.5
        # Track spikes, initiate refractory period
        spike = False
        na_rel = False
        if self.refrac[1] < 0.001:
            self.refrac[0] = False
        if self.v_s > 20 and self.refrac[0] == False:  #if Vs > 20mV, and NOT in refractory period
            spike = True
            self.tvec = np.append(self.tvec, 0)   #start timer for this spike, goes up in 1s (representing 1x dt)
            self.scount += 1    # add one to total spike count
            #print('spike count', self.scount)
            self.refrac[0] = True  # True = in refractory period
            self.refrac[1] = 3   # start countdown for refractory period in ms
            self.release_delay = int(10 / dt)  # delayed release timer in ms
            self.gcampTimers.append(0)
        self.gcamp_func()

        self.release_delay += (-1)
        if self.release_delay == 0:

            na_rel = True

        if psem and self.trans:
            self.psem_alpha_func(dt)

        self.activeRelProtein += dt * self.rel_alpha * self.ca_conc**self.caBindingFactor * (1-self.activeRelProtein) - self.rel_beta*self.activeRelProtein
        releaseScalar = 1 * self.activeRelProtein

        if na_rel:
            self.releaseTimers.append(0)
            self.releaseWeights.append(releaseScalar)
            norm_g_i = np.interp(self.auto_g_i, (0.00000002, 0.1), (-10, 10))  # normalise g_i to work with sigmoid function
            tau_d= 200+ (4000 / (1 + np.exp(-norm_g_i)))
            self.releaseTaus.append(tau_d)
            print(f'g_i = {self.auto_g_i}, decay tau: {tau_d}, [Ca] = {self.ca_conc}')

        self.auto_i_func_simple()

        if self.refrac[0]:
            self.refrac[1] += (-dt)   #if IN refractory period, subtract time step from refrac countdown


        self.tvec += 1
        self.l_i_tvec += 1

        #Function to update values per each step in time
        # currents
        i_leak = self.g_l*(self.v_s-self.v_l)
        i_k = self.g_k*(self.n**4)*(self.v_s -self.v_k)
        if ttx_on == True:
            i_na =  0.005 *(self.m_inf**3)*self.h*(self.v_s - self.v_na) #5 = 10% sodium conductance 0.5% = 1% sodium conductance
            #i_p = 0.00008*(self.p**2)*(self.v_s-self.v_na)   # Slow sodium current
        else:
            i_na =  self.g_na *(self.m_inf**3)*self.h*(self.v_s - self.v_na)
            #i_p = self.g_p*(self.p**2)*(self.v_s-self.v_na)   # Slow sodium current
        i_p = self.g_p*(self.p**2)*(self.v_s-self.v_na)   # Slow sodium current
        i_stan = i_leak + i_na # Leak current + Potassium current + Sodium current

        i_ahp = self.g_ahp*(self.ca_conc/(self.ca_conc+1)) * (self.v_s - self.v_k)  # afterhyperpolarisation current
        i_ca = self.g_ca*(self.m_ca)*self.y*(self.v_s-self.v_ca)
        g_i = self.auto_g_i + lat_g_i
        i_a2 =  g_i*(self.v_s - self.v_i)
        i_psam = self.g_psam * (self.v_s - self.v_na)

        i_syn = g_syn * (self.v_s - self.v_na)

        i_mem = i_stan + i_p  +  i_ahp + i_ca + i_a2  +i_syn  # Sum membrane currents

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
        wnoise = np.random.normal(loc = 0.0, scale = 0.15)
        self.v_s += dt*(i_e - i_psam - i_mem - self.g_d*(self.v_s - self.v_d) + wnoise)/self.c_s
        gap_volt = self.gap_func(gap_vs)   # calls separate function to calculate sum voltage differences over all connected unit
        self.v_d += dt*((-self.g_ld * (self.v_d - self.v_ld) - self.g_d*(self.v_d - self.v_s)) - gap_volt)/self.c_d
        self.n += dt*(3 * (alpha_n_v * (1-self.n) - beta_n_v*self.n))   # K current
        self.h += dt*(3 * (alpha_h_v * (1-self.h) - beta_h_v*self.h))  # #Sodium current
        self.p +=  dt*(alpha_p_v * (1-self.p) - beta_p_v*self.p)   # Slow sodium

        self.ca_conc += dt*(-0.004*(-self.v_s-self.v_ca)/(1+np.exp(-(self.v_s+25)/2.5)) - self.ca_conc/self.ca_tau)
        self.y += dt*(0.75 * (y_inf_v - self.y)/tau_y_v)
        return  self.v_s, self.v_d, self.ca_conc, i_a2, self.auto_g_i, self.releaseTimers, self.releaseWeights, self.releaseTaus, spike, i_ahp, self.g_psam, self.gcamp



    def auto_i_func(self):

        tau_r = 100
        rel_count = len(self.releaseTimers)
        self.auto_g_i = 0
        for n in np.arange(rel_count):
            t = self.releaseTimers[n]
            tau_d = self.releaseTaus[n]
            auto_i_single = 0.7 * self.releaseWeights[n] * (np.exp(-((t-0)/tau_r))   - np.exp(-((t-0)/tau_d)))/ (( (tau_d/tau_r)**(tau_d/(tau_r-tau_d))) -   ((tau_d/tau_r)**(tau_r/(tau_r-tau_d))))
            #auto_i_single = self.releaseWeights[n] *((t-0)/(3000)) * np.exp(-(t-0)/(3000))
            self.auto_g_i += auto_i_single
            self.releaseTimers[n] += self.dt
        return

    def auto_i_func_simple(self):

        rel_count = len(self.releaseTimers)
        self.auto_g_i = 0
        for n in np.arange(rel_count):
            t = self.releaseTimers[n]

            auto_i_single = self.releaseWeights[n] *((t-0)/(400)) * np.exp(-(t-0)/(400))
            self.auto_g_i += auto_i_single
            self.releaseTimers[n] += self.dt
        return





    def gap_func(self, gap_voltages):   # function to calculate sum voltage difference via gap junctions
        vd_diffsum = 0   # voltage difference over gap
        for q in gap_voltages:
            vd_diffsum += self.g*(self.v_d - q)
        return vd_diffsum

    def psem_alpha_func(self,dt):

        t_i = 1

        # exponential decay function
        self.g_psam =   0.2*((self.psem_clock-t_i)/2000) * np.exp(-(self.psem_clock-t_i)/(2000))
        self.psem_clock+=dt
            #print(g_i)

        return



    def gcamp_func(self):

        tau_r = 200
        tau_d = 600
        spike_count = len(self.gcampTimers)
        self.gcamp = 0
        for n in np.arange(spike_count):
            t = self.gcampTimers[n]
            gcamp_single =  (np.exp(-((t-0)/tau_r))   - np.exp(-((t-0)/tau_d)))/ (( (tau_d/tau_r)**(tau_d/(tau_r-tau_d))) -   ((tau_d/tau_r)**(tau_r/(tau_r-tau_d))))

            self.gcamp += gcamp_single
            self.gcampTimers[n] += self.dt
        return
