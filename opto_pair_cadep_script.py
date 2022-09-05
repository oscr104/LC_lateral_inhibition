''' Test script for homogenous module receiving lateral inhibition from stimulated Ns, for comparison to Ray Perrins data.
 using lcmp_funcs functions, lc_unit_euler. stimulated neurons are not explicitly simulated, 'reciever' neurons recieve input vector of optogenetically stimulated spikes (buzz at ?Hz for 1 second).
 Output variables Vs, Vd, [Ca], and i_a2, with i_a2 being primary variable of interest.
 initially to run in 'CC'. Then work out how to simulate VC
 started 3/1/2020 '''

import LC_unit_modular_cadeprelease as LC_U
import LC_vals_alt as LC
import lcmp_funcs as lcmpf
import connection_motifs as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def runSimulation(tvec, dt, gnum, gw, neu_num, trans, ie, isi_vec, ca_bf_vec, pulse_num, first_led):
        for isi in isi_vec:
                for caBindingFactor in ca_bf_vec:
                        led_train = np.zeros(len(tvec), dtype=bool) #  Rays stimulus = 30ms light pulses at 20Hz for 1 seconds duration.
                        for p in np.arange(pulse_num):
                                on = int((first_led+p*isi)/dt)
                                off = int(on + (5/dt))
                                led_train[on:off] = True

                        mod_params = {'tvec':tvec, 'ie':ie, 'gnum':gnum, 'gw':gw}

                        li_tau = 400
                        n1_list = {'g_na': LC.g_na, 'g_k': LC.g_k, 'g_l': LC.g_l, 'g_ld': LC.g_ld, 'g_p': LC.g_p, 'g_ahp': LC.g_ahp, 'g_ca': LC.g_ca, 'g_d': LC.g_d, 'g': LC.g_strong * gw,
                                      'c_s': LC.c_s, 'c_d': LC.c_d, 'v_na': LC.v_na, 'v_k': LC.v_k, 'v_ca': LC.v_ca, 'v_l': LC.v_l, 'v_ld': LC.v_ld, 'v_i': LC.v_i, 'm': LC.m, 'h': LC.h,
                                      'n': LC.n, 'p':LC.p, 'q':LC.q, 'r':LC.r, 'm_ca': LC.m_ca, 'y': LC.y, 'ca_init': LC.ca_conc,  'chR2': False, 'caBindingFactor':caBindingFactor}

                        n2_list = {'g_na': LC.g_na, 'g_k': LC.g_k, 'g_l': LC.g_l, 'g_ld': LC.g_ld, 'g_p': LC.g_p, 'g_ahp': LC.g_ahp, 'g_ca': LC.g_ca, 'g_d': LC.g_d, 'g': LC.g_strong * gw,
                                      'c_s': LC.c_s, 'c_d': LC.c_d, 'v_na': LC.v_na, 'v_k': LC.v_k, 'v_ca': LC.v_ca, 'v_l': LC.v_l, 'v_ld': LC.v_ld, 'v_i': LC.v_i, 'm': LC.m, 'h': LC.h,
                                      'n': LC.n, 'p':LC.p,'q':LC.q, 'r':LC.r, 'm_ca': LC.m_ca, 'y': LC.y, 'ca_init': LC.ca_conc,  'chR2': True, 'caBindingFactor':caBindingFactor}
                        population=[]

                        for params in [n1_list, n2_list]:
                                population.append(LC_U.LC_unit(**params))

                        wiring = np.ones((2,2))
                        np.fill_diagonal(wiring, int(0))
                        print(population)
                        print(len(population))

                        init_state = [LC.v_l, LC.v_ld, LC.ca_conc, 0, 0]

                        results_df = lcmpf.simulate_opto_m(population, wiring, init_state, mod_params, led_train)
                        print(results_df.head())

                        out_vars = results_df.columns.get_level_values('variable')
                        print(out_vars)

                        results_df.to_pickle(f'/home/nd17878/lc_code_git/results/thesis/calciumDependentRelease/paired_oneway_optostim_{isi}ISI_{caBindingFactor}CalciumExponent_currentout')
        return



t0 = 0
t1 = 20 * 1000
dt = 0.05  #ms   # 0.0001 standard
tvec = np.linspace(t0, t1, int(t1/dt), dtype=float)
#tvec = np.arange(t0,t1,dt, dtype = float)
print(tvec)
print(f"Simulating from {t0} to {t1}ms, increment = {dt}")

gnum = True  # use to make lcmpf choose all-to-all connection function in connection_motifs.py
gw = 0.0
neu_num = 2
trans = [True, False]
ie = 0.05
isi_vec = [20, 30, 40, 50, 60, 70, 80, 90,100]
ca_bf_vec = [0, 1, 5]
pulse_num = 4
first_led = int(3 * 1000)


simulate =False
plot = True
if simulate:
        runSimulation(tvec, dt, gnum, gw, neu_num, trans, ie, isi_vec, ca_bf_vec, pulse_num, first_led)

if plot:
        fig, ax = plt.subplots()
        exp_results = pd.read_csv('/home/nd17878/Documents/LC/LC_Data/Ray Paper/analysis/charge transfer/ct_graph_stats.csv')
        print(exp_results.head())
        ex_means = [100, 74.8, 57.85, 46.5, 36.17, 28.62, 26.75, 23.74, 22.06]
        ex_stdev = [0, 9.09, 5.58, 6.57, 3.21, 8.24, 11.14, 11.12, 12.13]
        m_df = exp_results.melt( id_vars = ['Cell_ID'], var_name = 'ISI (ms)',  value_name='Charge transfer (%)')
        m_df['ISI (ms)'] = m_df['ISI (ms)'].apply(pd.to_numeric)
        sns.lineplot(ax=ax, data=m_df,  x='ISI (ms)', ci=95, y='Charge transfer (%)', lw=3, marker='o',markersize=10, color='black', label='experimental results')
        #ax.plot(isi_vec, ex_means, lw=2, label= 'experimental results', color='black')
        #ax.errorbar(isi_vec, ex_means, yerr=ex_stdev, color='black');
        #print(exp_results.head())

        isi_results = np.zeros((len(isi_vec), len(ca_bf_vec)))
        isi_idx = 0
        for isi in isi_vec:
                ca_idx = 0
                for caBindingFactor in ca_bf_vec:

                        df = pd.read_pickle(f'/home/nd17878/lc_code_git/results/thesis/calciumDependentRelease/paired_oneway_optostim_{isi}ISI_{caBindingFactor}CalciumExponent_currentout')

                        clip_df = df.loc[int(first_led):int(first_led+4000)]
                        tvec = clip_df.index

                        max_ga2 = np.max(clip_df[(1,'g_a2')])
                        print(f'Max ga2 at {isi} ISI: {max_ga2}')
                        isi_results[isi_idx, ca_idx] = max_ga2

                        ca_idx+=1
                isi_idx+=1

        max_responses = np.max(isi_results,axis=0)


        count=0
        for max_r in max_responses:
                if ca_bf_vec[count] == 0:
                        label= 'no Ca dependence'
                elif ca_bf_vec[count] == 1:
                        label = f'Ca dependent release (1)'
                else:
                        label = f'Ca dependent release (5)'
                plot_vector = (isi_results[:,count] / max_r) *100
                df = pd.DataFrame(columns=isi_vec)
                df.loc[0] = plot_vector

                m_df = pd.melt(df, var_name='ISI (ms)')
                print(m_df.head())
                sns.lineplot(ax=ax, data=m_df, x='ISI (ms)', y='value', lw=2.5, marker='X',markersize=10, alpha=0.8,  label= label)
                count+=1

        ax.invert_xaxis()
        ax.legend(loc='lower right')
        ax.set_xlabel('ISI (ms)')
        ax.set_ylabel('Charge transfer (%)')
        sns.set_context('paper')
        sns.despine(offset=10, trim=True);

        plt.show()
