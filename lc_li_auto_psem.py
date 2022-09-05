''' LC modular sim with auto & lateral inhibition using LC_unit_psem with weighted function to do NA release & gi calculations
dec 2021

Reset filepath names etc as necessary to save outputs

'''

import LC_unit_psam as LC_U_p
import LC_vals_alt as LC
import lcmp_funcs as lcmpf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from neo.core import SpikeTrain
from quantities import s, ms
from elephant.statistics import instantaneous_rate
from elephant import kernels as ek
import warnings as wa



def lat_i_complex(releaseTimers, releaseWeights, releaseTaus,  dt):
    auto_i_sum = 0
    tau_r = 100
    rel_count = len(releaseTimers)
    g_i = 0
    for n in np.arange(rel_count):
        t = releaseTimers[n]
        tau_d = releaseTaus[n]
        g_i_single = 0.005 * (np.exp(-((t-0)/tau_r))   - np.exp(-((t-0)/tau_d)))/ (( (tau_d/tau_r)**(tau_d/(tau_r-tau_d))) -   ((tau_d/tau_r)**(tau_r/(tau_r-tau_d))))
        g_i += g_i_single

    return g_i

def lat_i_simple(releaseTimers, releaseWeights, dt):
    auto_i_sum = 0
    tau_r = 100
    rel_count = len(releaseTimers)
    g_i = 0
    for n in np.arange(rel_count):
        t = releaseTimers[n]
        w = releaseWeights[n]

        g_i_single = releaseWeights[n] *((t-0)/(300)) * np.exp(-(t-0)/(300))
        g_i += g_i_single

    return g_i




def runPairedSim(tvec, row_idx, ds_fac, i_e, trans_prop, neu_number, ttx, psem, psem_t, lateralInhibition):
    print(f"Simulating from {t0} to {t1}ms, increment = {dt}")
    out_vars = ['Vs_1', 'Vd_1', 'I_a2_1', 'g_i_1', 'dF_1', 'g_psam_1', 'Vs_2', 'Vd_2', 'I_a2_2', 'g_i_2', 'dF_2', 'g_psam_2']


    li_mat = np.array([[0,1],
                           [1, 0]], dtype=bool)


    a=0
    neurons=[]
    init_state = [LC.v_l, LC.v_ld, 0, 0, LC.ca_conc,  0]*2
    for x in range(neu_number):
        if x == 0:
            trans=True   # PSAM transduction
        else:
            trans = False

        param_list =  {'i_e':i_e, 'g_na': LC.g_na, 'g_k': LC.g_k, 'g_l': LC.g_l, 'g_ld': LC.g_ld, 'g_p': LC.g_p, 'g_ahp': LC.g_ahp, 'g_ca': LC.g_ca, 'g_d': LC.g_d, 'g': 0,
                      'c_s': LC.c_s, 'c_d': LC.c_d, 'v_na': LC.v_na, 'v_k': LC.v_k, 'v_ca': LC.v_ca, 'v_l': LC.v_l, 'v_ld': LC.v_ld, 'v_i': LC.v_i, 'm': LC.m, 'h': LC.h,
                      'n': LC.n, 'p':LC.p, 'm_ca': LC.m_ca, 'y': LC.y, 'ca_conc': LC.ca_conc, 'ca_tau': LC.ca_tau, 'li_w': 0.5, 'li_tau': 200, 'dt':dt, 'trans':trans, 'v_z':LC.v_l, 'hyp':False}


        neuron = LC_U_p.LC_unit(param_list)
        neurons.append(neuron)




    rows = len(row_idx)
    cols = len(out_vars)
    res_arr = np.zeros((rows,cols), dtype=float)

    res_arr[0, :] = init_state
    spike_vec = np.zeros((len(tvec),neu_number))

    r=0 #results table index (if ds_fac = 0 will go up in line with K)
    k = 0 #time index
    releaseTimers = []
    releaseWeights = []
    g_i = 0
    v_d = LC.v_ld
    l_i_v = [0] * neu_number

    for t in tvec:
        n = 0 # neuron index

        for neuron in neurons:
            if t < 900 and n==0:   #additive current at start of simulation to push simulation into balanced firing
                add_curr = -0.5
            else:
                add_curr = 0
            gap_vs = [v_d]
            lat_g_i = 0
            li_idx = np.nonzero(li_mat[n,:])
            for q in li_idx[0]:
                lat_g_i += l_i_v[q]
            if t>= psem_t:
                psem=True

            v_s, v_d, ca_conc, i_a2, g_i, releaseTimers, releaseWeights, releaseTaus, spike, i_ahp, g_psam, gcamp  = neuron.update( dt, ttx, gap_vs, lat_g_i, psem, add_curr)

            spike_vec[k,n] = spike
            for prop in releaseWeights:
                if prop > 1:
                    print(f'Lat I: weight: {n} at {t} ms')
            if lateralInhibition == 'simple':
                l_i_v[n] = lat_i_simple(releaseTimers, releaseWeights, dt)
            if lateralInhibition == 'complex':
                l_i_v[n] = lat_i_complex(releaseTimers, releaseWeights, releaseTaus,  dt)
            if k % ds_fac == 0 and t != row_idx[-1]:
                offset = int(len(out_vars)/neu_number)*n
                res_arr[r,offset+0] = v_s
                res_arr[r,offset+1] = v_d
                res_arr[r,offset+2] = i_a2
                res_arr[r,offset+3] = g_i
                res_arr[r,offset+4] = gcamp
                res_arr[r, offset+5] = g_psam
                if n+1 == neu_number:
                    r+=1
            n += 1
        k+= 1


    results_df = pd.DataFrame(res_arr,index=row_idx, columns=out_vars, dtype=float)
    spike_df = pd.DataFrame(spike_vec, index=tvec, columns=['n1 spike', 'n2 spike'], dtype=float)

    return results_df, spike_df
#
#
#
#
#
#
#
#


#Time parameters
t0 = 0
t1= 50 * 1000
dt = 0.05  #ms
tvec = np.linspace(t0, t1, int(t1/dt), dtype=float)
ds_fac =10 # downsample factor for saving results
row_idx = tvec[:-1:ds_fac]


#neuron parameters
i_e = 0.1
neu_number = 2
trans_prop = 0.5
ttx = False
psem = False
psem_t = 10*1000

lateralInhibition = 'simple'  # for non-clearing model (static tau_decay)
#lateralInhibition = 'complex' # for clearing model (tau_decay depends on g_i at time of spike)

run = False
plot = True

if run:
    print('TEST')
    results_df, spike_df = runPairedSim(tvec, row_idx, ds_fac, i_e, trans_prop, neu_number, ttx, psem, psem_t, lateralInhibition)
    key = np.random.randint(1000, 9999)
    results_df.to_pickle(f'/home/nd17878/lc_code_git/results/thesis/psam_lateral_inhibition/{neu_number}_unit_psem_{lateralInhibition}releaseModel_{key}')
    print(results_df)
    spike_df.to_pickle(f'/home/nd17878/lc_code_git/results/thesis/psam_lateral_inhibition/{neu_number}_unit_psem_{lateralInhibition}releaseModel_spikes_{key}')

if plot:
    key = int(1283)
    results_df = pd.read_pickle(f'/home/nd17878/lc_code_git/results/thesis/psam_lateral_inhibition/{neu_number}_unit_psem_{lateralInhibition}releaseModel_{key}')
    spike_df = pd.read_pickle(f'/home/nd17878/lc_code_git/results/thesis/psam_lateral_inhibition/{neu_number}_unit_psem_{lateralInhibition}releaseModel_spikes_{key}')


plot_colors = ['firebrick', 'forestgreen']

fig, axs = plt.subplots(4, sharex=True)
fig.suptitle(f'Paired simulation PSEM308 response')
axs[0].set_ylabel('g_psam')
axs[1].set_ylabel('Vm')
axs[2].set_ylabel('Vm')

axs[3].set_ylabel('Spikes/sec')


time_plot = tvec[::ds_fac]/1000


i_freq_window = 3  #in seconds
for n in np.arange(neu_number):

    nkey = int(n+1)
    print(f'neuron {nkey}')
    color=plot_colors[n]
    if n==0:
        axs[0].plot(time_plot, results_df[f'g_psam_{nkey}'], color='black', alpha=0.7, lw=2)
        axs[0].set_ylabel(r'$g_{PSAM}$')

    axs[n+1].plot(time_plot,  results_df[f'Vs_{nkey}'], color=color,alpha=0.7,  lw=2)
    spikes = spike_df[f'n{nkey} spike'].values
    spike_df2 = pd.DataFrame( columns=['Time(s)', 'Spike'])
    spike_df2['Time(s)'] = tvec/1000
    spike_df2['spike'] = spikes

    spike_ts = spike_df2.loc[spike_df2['spike'] > 0, ['Time(s)']]

    spike_train = SpikeTrain(spike_ts['Time(s)'].values,units=s, t_stop = int(t1/1000)*s)
    print(spike_ts)
    kernel = ek.GaussianKernel(sigma=1 * s)
    inst_rate = instantaneous_rate(spike_train, kernel=kernel, sampling_period=0.05*ms)
    axs[3].plot(tvec/1000, inst_rate, lw=2, alpha=0.7, color=color)



axs[0].axvline(psem_t/1000, color='gray', lw=2, linestyle='--')
axs[1].axvline(psem_t/1000, color='gray', lw=2, linestyle='--')
axs[2].axvline(psem_t/1000, color='gray', lw=2, linestyle='--')
axs[3].axvline(psem_t/1000, color='gray', lw=2, linestyle='--')
axs[3].set_xlabel('Time (s)')

sns.set_context('paper')
sns.despine(offset=10, trim=False);
plt.savefig(f'/home/nd17878/Documents/LC/LC_Data/Model_results/psem_test/single_unit_psem_initial_test_summary_plot_{key}.png', format='png')
plt.show()
