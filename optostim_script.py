''' Test script for single neuron receiving lateral inhibition from neighbour, for comparison to Ray Perrins data.
 using lcmp_funcs functions, lc_unit_euler. stimulated neurons are not explicitly simulated, 'reciever' neurons recieve input vector of optogenetically stimulated spikes (buzz at ?Hz for 1 second).
 Output variables Vs, Vd, [Ca], and i_a2, with i_a2 being primary variable of interest.
 initially to run in 'CC'. Then work out how to simulate VC
 started 3/1/2020 '''

import LC_unit_opto_single as LC_U
import LC_vals_alt as LC
import lcmp_funcs as lcmpf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


t0 = 0
t1 =15 * 1000
dt = 0.05  #ms   # 0.0001 standard
tvec = np.linspace(t0, t1, int(t1/dt), dtype=float)
#tvec = np.arange(t0,t1,dt, dtype = float)
print(tvec)
print(f"Simulating from {t0} to {t1}ms, increment = {dt}")

ie_vec = np.full((len(tvec),1), 0.05, dtype=float)

gnum = True  # use to make lcmpf choose all-to-all connection function in connection_motifs.py
gw = 0.25
neu_num = 3

freq = int(20)

spike_train = np.zeros(len(tvec), dtype=bool) #  Rays stimulus = 30ms light pulses at 20Hz for 1 seconds duration.
spike_train[int(3000/dt)] = True
spike_train[int(8000/dt):int(8500/dt):int((1000/freq)/dt)] = True
print(f'modelling response to {np.sum(spike_train)} spikes at {freq} Hz')

params = {'tvec':tvec, 'ie_vec':ie_vec, 'gnum':gnum, 'gw':gw}
li_w = 0.1
li_tau = 400
param_list = {'g_na': LC.g_na, 'g_k': LC.g_k, 'g_l': LC.g_l, 'g_ld': LC.g_ld, 'g_p': LC.g_p, 'g_ahp': LC.g_ahp, 'g_ca': LC.g_ca, 'g_d': LC.g_d, 'g': LC.g_strong*gw,
              'c_s': LC.c_s, 'c_d': LC.c_d, 'v_na': LC.v_na, 'v_k': LC.v_k, 'v_ca': LC.v_ca, 'v_l': LC.v_l, 'v_ld': LC.v_ld, 'v_i': LC.v_i, 'm': LC.m, 'h': LC.h,
              'n': LC.n, 'p':LC.p, 'm_ca': LC.m_ca, 'y': LC.y, 'ca_conc': LC.ca_conc, 'li_w': li_w, 'li_tau': li_tau}


neuron = LC_U.LC_unit(param_list)

init_state = [LC.v_l, LC.v_ld, LC.ca_conc, 0, (1 / (1 + np.exp(-LC.ca_conc)))]

results_df = lcmpf.simulate_opto_s(neuron, init_state, params, spike_train)
print(results_df.head())
out_vars = results_df.columns.values.tolist()
print(out_vars)


# Output variables stored as a dataframe

results_df.to_pickle(f'/home/nd17878/lc_code_git/results/thesis/receiverneuron_{freq}Hz_single_n_{t1}ms_{dt}dt')

results_df = pd.read_pickle(f'/home/nd17878/lc_code_git/results/thesis/receiverneuron_{freq}Hz_single_n_{t1}ms_{dt}dt')
fig,ax = plt.subplots(3,1, sharex=True)
ax[0].plot(tvec/1000, spike_train, lw=0.5, alpha=0.8, color='firebrick')
ax[1].plot(tvec[:-1]/1000,results_df['V_s'], lw=2, alpha=0.8, color='forestgreen')
ax[1].set_ylabel(r'$V_s$')
ax[1].axvline(3, linestyle='--', lw=1, color='gray')
ax[1].axvline(8, linestyle='--', lw=1, color='gray')
ax[2].plot(tvec[:-1]/1000,results_df['I_a2'], lw=2, alpha=0.8, color='cornflowerblue')
ax[2].set_ylabel(r'$I_{α2} (μA/cm^2)$')
ax[2].set_xlabel('Time (s)')
ax[2].axvline(3, linestyle='--', lw=1, color='gray')
ax[2].axvline(8, linestyle='--', lw=1, color='gray')
sns.despine();
sns.set_context("paper")
plt.show()
