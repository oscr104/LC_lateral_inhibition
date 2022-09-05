# Plot F_I curve using LC unit from Alvarez et al 2002, with Euler method
# O Davy 2/2/18

import LC_unit_euler_master as LC_U
import LC_vals_alt as LC
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def run(lc_neu, tvec, i_input):



   spk_count= 0

   k = 0
   dt = tvec[1]
   v_dj = 0  # dendrite voltage of other neuron = 0 ... no other neuron
   for t in tvec:
       spike, i_na, i_k, i_l, i_p, i_ahp, i_ca, i_mem, v_s, v_d, ca_conc= lc_neu.update(i_input[k], False, False,  dt, False, False, False)    # 0 for gap junction voltage   i, gap_voltages, ca_concs, dt, ttx_on, carb_on, l_i_spk
       if spike:
          spk_count+=1
       k +=1

   return spk_count


# Time values in ms
t0 = int(0)
t1 = int(5000)
dt = 0.05  # paper = 0.05 but artifacts on spike, 0.025 gives smooth AP
tvec = np.linspace(t0, t1, int(t1/dt), dtype=float)


# Import LC Parameters
#Conductances
g_na = LC.g_na  # Sodium
g_k = LC.g_k   # Potassium
g_l = LC.g_l  # Leak current
g_ld = LC.g_ld  # dendritic leak current
g_p = LC.g_p   # Slow persistent sodium current
g_ahp = LC.g_ahp  # afterhyperpolarisation current
g_ca = LC.g_ca  # Calcium current
g_d = LC.g_d    # Dendrite-soma conductance
g = LC.g_weak  # choose weak or strong gap conductance
#Capacitances
c_s = LC.c_s   # Somatic C
c_d = LC.c_d  # Dendritic C
#Voltages
v_na = LC.v_na # sodium equilibrium voltage
v_k = LC.v_k #Potassium equilibrium voltage
v_ca = LC.v_ca #Calcium resting voltage
v_l = LC.v_l   # resting Vm
v_ld = LC.v_ld # dendritic resting Vm
v_i = LC.v_i
# Activation values
m = LC.m
h = LC.h
n = LC.n
p = LC.p
m_ca = LC.m_ca
y = LC.y
autoInhibition=True
rectifyingGapJunctions=False

#Calcium concentration
ca_init = LC.ca_conc

#i_vec = [-0.4,  -0.2,  0., 0.2,  0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6]
i_vec = np.linspace(-0.4, 10, 10)
aTwoTauVec = np.linspace(20,2000, 10)
res_vec = np.zeros((len(i_vec), 5))

sim=False
plot=True

if sim:
   for aTwoTau_r in aTwoTauVec:
      aTwoTau= int(aTwoTau_r)
      loop_count=0
      for i in i_vec:
         i_input = [i] * len(tvec)
         for idx in np.arange(5):
            lc_neu = LC_U.LC_unit(g_na, g_k, g_l, g_ld, g_p, g_ahp, g_ca, g_d, g, c_s, c_d,  v_na, v_k, v_ca, v_l, v_ld, v_i, m, h, n, p, m_ca, y, ca_init, rectifyingGapJunctions,autoInhibition,aTwoTau)
            spk_count= run(lc_neu, tvec, i_input )
            res_vec[loop_count,idx] = spk_count/5
         print(f'{i}uA/cm2:  {spk_count/5}Hz')
      loop_count+=1
      df = pd.DataFrame(data=res_vec, index=i_vec, columns=np.arange(5))
      df.to_pickle(f'/home/nd17878/lc_code_git/results/thesis/autoInhibition/FI_data_{aTwoTau}aTwoTau')

if plot:
   fig, ax = plt.subplots()
   df= pd.read_pickle(f'/home/nd17878/lc_code_git/results/thesis/autoInhibition/fig_1_fi_data_full')   # <2 i_e
   df.reset_index(inplace=True)
   print(df.head())
   df = df.rename(columns= {'index': 'I_in'})
   mdf=df.melt(id_vars = 'I_in', value_vars = [0, 1, 2, 3, 4], value_name='FR')
   sns.lineplot(ax=ax, x='I_in', y='FR', data=mdf,estimator='mean', ci=95, lw=2, alpha=0.8, color='cornflowerblue')
   ax.set_ylabel('Firing rate (Hz)')
   ax.set_xlabel(r'$I_{input} (uA/cm^2)$')



   for aTwoTau in aTwoTauVec:
      aTwoTau = int(aTwoTau)
      df= pd.read_pickle(f'/home/nd17878/lc_code_git/results/thesis/autoInhibition/FI_data_{aTwoTau}aTwoTau')   # <2 i_e
      print(df.head())
      df.reset_index(inplace=True)
      df = df.rename(columns= {'index': 'I_in'})
      mdf=df.melt(id_vars = 'I_in', value_vars = [0, 1, 2, 3, 4], value_name='FR')
      sns.lineplot(ax=ax, x='I_in', y='FR', data=mdf,estimator='mean', ci=95, lw=2, alpha=0.8, label=f'a2 Tau: {aTwoTau}ms')

   plt.legend()
   sns.despine();
   sns.set_context("paper")
   plt.show()
