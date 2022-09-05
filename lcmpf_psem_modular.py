''' LC modular sim with auto & lateral inhibition using LC_unit_psem with novel function to do NA release & gi calculations
dec 2021'''

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
from scipy.signal import savgol_filter, detrend
from scipy.optimize import curve_fit
from scipy.special import expit
from uncertainties import ufloat


def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

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

def lat_i_simple(releaseTimers, releaseWeights, dt,scalar):
    auto_i_sum = 0
    tau = 200
    rel_count = len(releaseTimers)
    g_i = 0
    for n in np.arange(rel_count):
        t = releaseTimers[n]
        w = releaseWeights[n]

        g_i_single = scalar *  ((t-0)/tau) * np.exp(-(t-0)/(tau))
        g_i += g_i_single

    return g_i


def syn_simple(afferent_Timers,  dt):

    tau = 10
    rel_count = len(afferent_Timers)
    g_syn = 0
    for n in np.arange(rel_count):
        t = afferent_Timers[n]

        g_syn_single = 0.1 * ((t-0)/(tau)) * np.exp(-(t-0)/(tau))
        g_syn += g_syn_single

    return g_syn



def runPairedSim(tvec, row_idx, ds_fac, i_e, trans_prop, neu_number, ttx, psem, psem_t, lateralInhibition, afferent_spikes, li_scalar):
    print(f"Simulating from {t0} to {t1}ms, increment = {dt}")
    out_vars = ['Vs1', 'Vd1', 'I_a21', 'g_i1', 'dF1', 'g_psam1', 'Vs2', 'Vd2', 'I_a22', 'g_i2', 'dF2', 'g_psam2', 'Vs3', 'Vd3', 'I_a23', 'g_i3', 'dF3', 'g_psam3', 'Vs4', 'Vd4', 'I_a24', 'g_i4', 'dF4', 'g_psam4', 'Vs5', 'Vd5', 'I_a25', 'g_i5', 'dF5', 'g_psam5', 'Vs6', 'Vd6', 'I_a26', 'g_i6', 'dF6', 'g_psam6']



    li_mat = np.array([[0 ,0, 0, 1, 1, 1],
                       [0, 0, 0, 1, 1, 1],
                       [0, 0, 0, 1, 1, 1],
                       [1, 1, 1, 0, 0, 0],
                       [1, 1, 1, 0, 0, 0],
                       [1, 1, 1, 0, 0, 0]], dtype=bool)


    a=0
    neurons=[]
    init_state = [LC.v_l, LC.v_ld, 0, 0, LC.ca_conc,  0]*neu_number
    for x in range(neu_number):


        param_list =  {'i_e':i_e, 'g_na': LC.g_na, 'g_k': LC.g_k, 'g_l': LC.g_l, 'g_ld': LC.g_ld, 'g_p': LC.g_p, 'g_ahp': LC.g_ahp, 'g_ca': LC.g_ca, 'g_d': LC.g_d, 'g': 0,
                      'c_s': LC.c_s, 'c_d': LC.c_d, 'v_na': LC.v_na, 'v_k': LC.v_k, 'v_ca': LC.v_ca, 'v_l': LC.v_l, 'v_ld': LC.v_ld, 'v_i': LC.v_i, 'm': LC.m, 'h': LC.h,
                      'n': LC.n, 'p':LC.p, 'm_ca': LC.m_ca, 'y': LC.y, 'ca_conc': LC.ca_conc, 'ca_tau': LC.ca_tau, 'li_w': 0.5, 'li_tau': 200, 'dt':dt, 'trans':False,'v_z':LC.v_l, 'hyp':False}


        neuron = LC_U_p.LC_unit(param_list)
        neurons.append(neuron)




    row_num = len(row_idx)
    col_num = len(out_vars)
    res_arr = np.zeros((row_num,col_num), dtype=float)
    spike_vec = np.zeros((len(tvec), neu_number), dtype=bool)

    res_arr[0, :] = init_state


    r=0 #results table index (if ds_fac = 0 will go up in line with K)
    k = 0 #time index
    releaseTimers = []
    releaseWeights = []
    afferent_timers = []
    g_i = 0
    v_d = LC.v_ld
    l_i_v = [0] * neu_number

    for t in tvec:
        n = 0 # neuron index
        spike_in = afferent_spikes[k]
        g_syn = 0
        if spike_in:
            afferent_timers.append(0)
        g_syn = syn_simple(afferent_timers, t)
        afferent_spikenum = len(afferent_timers)
        for aff_count in np.arange(afferent_spikenum):
            afferent_timers[aff_count]+=dt



        for neuron in neurons:

            gap_vs = [v_d]
            lat_g_i = 0
            li_idx = np.nonzero(li_mat[n,:])
            for q in li_idx[0]:
                lat_g_i += l_i_v[q]
            if t>= psem_t:
                psem=True

            if n > 2:
                g_syn = 0



            add_curr=0

            v_s, v_d, ca_conc, i_a2, g_i, releaseTimers, releaseWeights, releaseTaus, spike, i_ahp, g_psam, gcamp  = neuron.update( dt, ttx, gap_vs, lat_g_i, psem, g_syn)

            spike_vec[k,n] = spike
            for prop in releaseWeights:
                if prop > 1:
                    print(f'Lat I: weight: {n} at {t} ms')
            if lateralInhibition == 'simple':
                l_i_v[n] = lat_i_simple(releaseTimers, releaseWeights, dt, li_scalar)
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
    spike_df = pd.DataFrame(spike_vec, index=tvec, columns=['n1 spike', 'n2 spike', 'n3 spike', 'n4 spike', 'n5 spike', 'n6 spike'], dtype=float)

    return results_df, spike_df

def simulateModules():
    for li_scalar in li_scalar_vec:
        for rep in np.arange(10):
            results_df, spike_df = runPairedSim(tvec, row_idx, ds_fac, i_e, trans_prop, neu_number, ttx, psem, psem_t, lateralInhibition, afferent_train, li_scalar)
            #key = np.random.randint(1000, 9999)


            results_df.to_pickle(f'/home/nd17878/lc_code_git/results/thesis/phasic_lateral_inhibition/MODULAR_{neu_number}_unit_psem_{lateralInhibition}releaseModel_{li_scalar}LI_weight_{rep}')
            print(results_df)
            spike_df.to_pickle(f'/home/nd17878/lc_code_git/results/thesis/phasic_lateral_inhibition/MODULAR_{neu_number}_unit_psem_{lateralInhibition}releaseModel_{li_scalar}LI_weight_spikes_{rep}')
    return


def plotAndQuantify(plot_individual, li_scalar_vec, neu_number, ds_fac):
    #savgol filter to remove HF noise
    window = 2501
    polyn = 3

    ex_in_results = np.zeros((10*neu_number,len(li_scalar_vec*2)+1))
    isi_results = np.zeros((10*neu_number, len(li_scalar_vec*2)+1))
    print(ex_in_results)
    li_s_idx = 0
    for li_scalar in li_scalar_vec:
        for rep in np.arange(10):
            #results_df = pd.read_pickle(f'/home/nd17878/lc_code_git/results/thesis/phasic_lateral_inhibition/MODULAR_{neu_number}_unit_psem_{lateralInhibition}releaseModel_{li_scalar}LI_weight_{rep}')
            spike_df = pd.read_pickle(f'/home/nd17878/lc_code_git/results/thesis/phasic_lateral_inhibition/MODULAR_{neu_number}_unit_psem_{lateralInhibition}releaseModel_{li_scalar}LI_weight_spikes_{rep}')

            '''
            fig, axs = plt.subplots(5, sharex=True)
            fig.suptitle(f'Paired simulation PSEM308 response')
            axs[0].set_ylabel('Input spikes')
            axs[1].set_ylabel('Vm M1)')
            axs[2].set_ylabel('Vm M2')
            axs[3].set_ylabel('Spikes/sec')
            axs[4].set_ylabel('dF')

            time_plot = tvec[::ds_fac]/1000
            inst_rate_m1 = np.zeros((len(tvec), neu_number))
            inst_rate_m2 = np.zeros((len(tvec), neu_number))

            axs[0].plot(tvec/1000, afferent_train, color='black', alpha=0.7, lw=2)
            axs[0].set_ylabel('Spikes')


            m1_basal_rates = np.zeros(3)
            m2_basal_rates = np.zeros(3)
            m1_epoch_rates = np.zeros(3)
            m2_epoch_rates = np.zeros(3)
            '''
            i_freq_window = 3  #in seconds
            m1_idx = 0
            m2_idx = 0
            for n in np.arange(neu_number):
                print(f'neuron {n+1}')
                nkey = int(n+1)
                spikes = spike_df[f'n{nkey} spike'].values
                spike_df2 = pd.DataFrame( columns=['Time', 'spike'])
                spike_df2['Time'] = tvec/1000
                spike_df2['spike'] = spikes
                '''
                if n < 3:
                    axs[1].plot(time_plot,  results_df[f'Vs{nkey}'], color='firebrick',alpha=0.7,  lw=2)
                    axs[4].plot(time_plot, savgol_filter(results_df[f'dF{nkey}'], window, polyn), color='firebrick', alpha=0.7, lw=2)
                    #m1_basal_spikes = np.sum(spike_df)
                    m1_basal_spikes = spike_df2.loc[spike_df2['Time']<10, 'spike']

                    m1_basal_rates[m1_idx] = m1_basal_spikes.sum() /10
                    m1_epoch_spikes =spike_df2.query('10 < Time <13')['spike']
                    m1_epoch_rates[m1_idx] = m1_epoch_spikes.sum()/3

                    m1_idx += 1
                else:
                    axs[2].plot(time_plot,  results_df[f'Vs{nkey}'], color='forestgreen',alpha=0.7,  lw=2)
                    axs[4].plot(time_plot, savgol_filter(results_df[f'dF{nkey}'], window, polyn), color='forestgreen', alpha=0.7, lw=2)
                    m2_basal_spikes = spike_df2.loc[spike_df2['Time']<10, 'spike']
                    m2_basal_rates[m2_idx] = m2_basal_spikes.sum() /10
                    m2_epoch_spikes =spike_df2.query('10 < Time <13')['spike']
                    m2_epoch_rates[m2_idx] = m2_epoch_spikes.sum()/3
                    m2_idx += 1

                '''

                spike_ts = spike_df2.loc[spike_df2['spike'] > 0, ['Time']]
                sts = spike_ts['Time'].values
                print('spike times')
                basal_isis = []
                epoch_isis = []
                inhibited_isi_check = 0 # variable codes whether inhibited isi has been calculated
                for st1, st2 in zip(sts, sts[1:]):
                    print(f'st1 {st1}')
                    print(f'st2 {st2}')
                    if st1 < 10 and st2 < 10:   # get basal ISIs

                        print('basal ISI')
                        basal_isis.append(st2-st1)
                    elif st1 > 10 and st1 < 13 and st2 > 10 and st2 < 13 and (n+1)< 4:  # get ISIs during stimulus period for module 1
                        print('m1 stim period')
                        epoch_isis.append(st2-st1)

                    elif (n+1) > 3 and st1 < 10 and st2 > 10 and st2 < 13:
                        print('m2 inhibitory period ISI')
                        epoch_isis.append(st2-st1)
                    elif (n+1) > 3 and st1 < 13 and st2 > 13:
                        print('m2 stim final ISI')
                        epoch_isis.append(st2-st1)


                print(basal_isis)
                print(epoch_isis)
                row_offset = int(rep*neu_number)+n
                column_offset = int(li_s_idx*2)+1
                basal_isi_mean = np.nanmean(basal_isis)
                epoch_isi_mean = np.nanmean(epoch_isis)
                isi_results[row_offset, column_offset] = basal_isi_mean
                isi_results[row_offset, column_offset+1] = epoch_isi_mean






                '''
                spike_train = SpikeTrain(spike_ts['Time'].values,units=s, t_stop = int(t1/1000)*s)
                kernel = ek.GaussianKernel(sigma= 0.5 * s)
                inst_rate = instantaneous_rate(spike_train, kernel=kernel, sampling_period=0.05*ms)
                if n<3:
                    inst_rate_m1[:,n] = inst_rate.ravel()
                else:
                    inst_rate_m2[:,n] = inst_rate.ravel()


                if n < 3:
                    ex_in_results[row_offset, 0] = '0'
                else:
                    ex_in_results[row_offset, 0] = '1'
                basal_spikes = spike_df2.loc[spike_df2['Time']<10, 'spike']
                ex_in_results[row_offset, column_offset] = basal_spikes.sum() / 10
                epoch_spikes = spike_df2.query('10 < Time <13')['spike']
                ex_in_results[row_offset, column_offset+1] = epoch_spikes.sum() / 3



            avg_inst_rate_m1 = np.mean(inst_rate_m1, axis=1)
            ci = 95 * np.mean(inst_rate_m1, axis=1)/np.sqrt(len(tvec))
            axs[3].plot(tvec/1000, avg_inst_rate_m1, lw=2, alpha=0.7, color='firebrick')
            axs[3].fill_between(tvec/1000, (avg_inst_rate_m1-ci), (avg_inst_rate_m1+ci), color='firebrick', alpha=.5)
            avg_inst_rate_m2 = np.mean(inst_rate_m2, axis=1)
            ci = 95 * np.mean(inst_rate_m2, axis=1)/np.sqrt(len(tvec))
            axs[3].plot(tvec/1000, avg_inst_rate_m2, lw=2, alpha=0.7, color='forestgreen')
            axs[3].fill_between(tvec/1000, (avg_inst_rate_m2-ci), (avg_inst_rate_m2+ci), color='forestgreen', alpha=.5)


            ex_peak = np.max(avg_inst_rate_m1)
            print(f'peak: {ex_peak}')


            ex_peak_t = np.argmax(avg_inst_rate_m1)
            inhib_at_peakt = avg_inst_rate_m2[ex_peak_t]




            axs[0].axvline(phasic_on/1000, color='gray', lw=2, linestyle='--')
            axs[1].axvline(phasic_on/1000, color='gray', lw=2, linestyle='--')
            axs[2].axvline(phasic_on/1000, color='gray', lw=2, linestyle='--')
            axs[3].axvline(phasic_on/1000, color='gray', lw=2, linestyle='--')
            axs[4].axvline(phasic_on/1000, color='gray', lw=2, linestyle='--')
            #axs[3].axvline((ex_peak_t*dt)/1000, color='gray', lw=1, linestyle='--')
            axs[4].set_xlabel('Time (s)')

            sns.set_context('paper')
            sns.despine(offset=10, trim=False);
            plt.savefig(f'/home/nd17878/lc_code_git/results/thesis/phasic_lateral_inhibition/plots/MODULAR_{neu_number}_unit_psem_{lateralInhibition}releaseModel_{li_scalar}LI_weight_{rep}', format='png')
            '''
        li_s_idx += 1
        plt.close('all')


    columns = ['module']
    for li in li_scalar_vec:
        columns.append(f'{li}_basal')
        columns.append(f'{li}_epoch')
    #ex_in_df = pd.DataFrame(data=ex_in_results, columns=columns)
    ex_in_df = 0
    isi_df = pd.DataFrame(data=isi_results, columns = columns)

    return ex_in_df, isi_df



def aggregatePlot(ex_in_df):


    ratio_results = np.zeros(( len(li_scalar_vec), 10+1))
    for rep in np.arange(10):
        df = ex_in_df.query('Rep == @rep')
        li_idx = 0
        for li in li_scalar_vec:
            ratio_results[li_idx, 0] = li
            bkey = f'{li}_basal'
            ekey = f'{li}_epoch'
            m1_basal_avg = df.query('module == 0')[bkey].mean(skipna=True)
            m1_epoch_avg = df.query('module == 0')[ekey].mean(skipna=True)
            m2_basal_avg = df.query('module == 1')[bkey].mean(skipna=True)
            m2_epoch_avg = df.query('module == 1')[ekey].mean(skipna=True)
            ratio_results[li_idx, rep+1] =  ((m1_epoch_avg - m1_basal_avg) - (m2_epoch_avg -m2_basal_avg)) / ((m1_basal_avg + m2_basal_avg) / 2)
            li_idx += 1

    columns = ['li_scalar', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    df = pd.DataFrame(data=ratio_results, columns = columns)
    print(df.head())

    fig, ax = plt.subplots()
    plot_df =df.melt(id_vars = 'li_scalar', value_vars=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    print(plot_df.head())
    sns.stripplot(ax=ax, data=plot_df, x='li_scalar', y='value', marker='o')
    ax.set_ylim([0, 6.5])
    plt.savefig(f'/home/nd17878/lc_code_git/results/thesis/phasic_lateral_inhibition/plots/aggregate_stripplot.png', format='png', transparent=True)
    plt.close()

    y_columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    xdata=li_scalar_vec
    ydata = np.nanmean(df[y_columns], axis=1)
    errbar = np.nanstd(df[y_columns], axis=1)



    sigma = np.ones(len(xdata))
    popt, pcov = curve_fit(sigmoid, xdata, ydata, p0=[1, 1, 1, 1], method='lm')
    print(pcov)
    perr = np.sqrt(np.diag(pcov))  # S.D error of the fit
    a = ufloat(popt[0], perr[0])
    b = ufloat(popt[1], sigma[1])
    text_res = f'Best fit parameters:\na = {a}\nb= {b}'

    print(popt)
    x=np.linspace(0,0.0055,1000)
    yfit = sigmoid(x, *popt)
    bound_upper = sigmoid(x, *(popt + perr))
    bound_lower = sigmoid(x, *(popt - perr))

    df.to_csv(f'/home/nd17878/lc_code_git/results/thesis/phasic_lateral_inhibition/sigmoid_data_exp.csv')
    fig, ax = plt.subplots()
    ax.plot(xdata, ydata, 'o', markersize=5, color ='b', alpha=0.8)
    ax.errorbar(xdata, ydata, yerr=errbar, fmt='none', alpha = 0.4)
    ax.plot(x ,yfit, color= 'black', lw =2, label='Sigmoid +/- S.E. of fit')
    #plt.plot(x, perr, color='gray', lw=1, alpha=0.2)
    #ax.fill_between(x, bound_lower, bound_upper, color='black', alpha=0.15)

    plt.legend( loc='upper left', frameon=False, fontsize=10)
    plt.set_ylabel('Contrast enhancement', fontsize=10)
    plt.set_xlabel('Lat. Inhib. strength', fontsize=10)
    sns.set('paper')
    sns.despine(offset=10)

    plt.savefig(f'/home/nd17878/lc_code_git/results/thesis/phasic_lateral_inhibition/plots/aggregate_sigmoidpplot.png', format='png', transparent=True)

    return

def aggregateISIPlot(isi_df):

    ratio_results = np.zeros(( len(li_scalar_vec), 10+1))
    for rep in np.arange(10):
        df = isi_df.query('Rep == @rep')
        print(df.head())
        li_idx = 0
        for li in li_scalar_vec:
            ratio_results[li_idx, 0] = li
            bkey = f'{li}_basal'
            print(bkey)
            ekey = f'{li}_epoch'
            m1_basal_avg = df.query('module == 0')[bkey].mean(skipna=True)
            print(m1_basal_avg)
            m1_epoch_avg = df.query('module == 0')[ekey].mean(skipna=True)
            print(m1_epoch_avg)
            m2_basal_avg = df.query('module == 1')[bkey].mean(skipna=True)
            print(m2_basal_avg)
            m2_epoch_avg = df.query('module == 1')[ekey].mean(skipna=True)
            print(m2_epoch_avg)

            # ISI module 2 / ISI module 1]
            change_ratio = (m2_epoch_avg/m1_epoch_avg) / (m2_basal_avg / m1_basal_avg)
            print(change_ratio)
            ratio_results[li_idx, rep+1] = change_ratio
            li_idx += 1

    columns = ['li_scalar', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    df = pd.DataFrame(data=ratio_results, columns = columns)
    print(df.head())

    fig, ax = plt.subplots()
    plot_df =df.melt(id_vars = 'li_scalar', value_vars=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    print(plot_df.head())
    sns.stripplot(ax=ax, data=plot_df, x='li_scalar', y='value', marker='o')
    #ax.set_ylim([0, 6.5])
    plt.savefig(f'/home/nd17878/lc_code_git/results/thesis/phasic_lateral_inhibition/plots/aggregate_ISI_stripplot.png', format='png', transparent=True)
    plt.close()

    y_columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    xdata=li_scalar_vec
    ydata = np.nanmean(df[y_columns], axis=1)
    errbar = np.nanstd(df[y_columns], axis=1)



    sigma = np.ones(len(xdata))
    popt, pcov = curve_fit(sigmoid, xdata, ydata, p0=[1, 1, 1, 1], method='lm')
    print(popt)
    perr = np.sqrt(np.diag(pcov))  # S.D error of the fit
    a = ufloat(popt[0], perr[0])
    b = ufloat(popt[1], sigma[1])
    text_res = f'Best fit parameters:\na = {a}\nb= {b}'

    print(popt)
    x=np.linspace(0,0.0055,1000)
    yfit = sigmoid(x, *popt)
    bound_upper = sigmoid(x, *(popt + perr))
    bound_lower = sigmoid(x, *(popt - perr))

    df.to_csv(f'/home/nd17878/lc_code_git/results/thesis/phasic_lateral_inhibition/sigmoid_data_ISI.csv')
    fig, ax = plt.subplots()
    ax.plot(xdata, ydata, 'o', markersize=5, color ='b', alpha=0.8, label='mean +/- S.D')
    ax.errorbar(xdata, ydata, yerr=errbar, fmt='none', alpha = 0.4)
    ax.plot(x ,yfit, color= 'black', lw =2, label=f'Sigmoid fit')
    #ax.plot(x, perr, color='gray', lw=1, alpha=0.2)
    #ax.fill_between(x, bound_lower, bound_upper, color='black', alpha=0.15)

    plt.legend( loc='upper left', frameon=False, fontsize=10)
    ax.set_ylabel('Contrast enhancement', fontsize=10)
    ax.set_xlabel('Lat. Inhib. strength', fontsize=10)
    sns.set('paper')
    sns.despine(offset=10)

    #plt.savefig(f'/home/nd17878/lc_code_git/results/thesis/phasic_lateral_inhibition/plots/aggregate_sigmoidpplot_ISI.png', format='png')

    return











#Time parameters
t0 = 0
t1= 20 * 1000
dt = 0.05  #ms
tvec = np.linspace(t0, t1, int(t1/dt), dtype=float)
ds_fac =10 # downsample factor for saving results
row_idx = tvec[:-1:ds_fac]


#neuron parameters
i_e = 0.2
neu_number = 6
trans_prop = 0.5
ttx = False
psem = False
psem_t = 10
li_scalar = 0.3



#synaptic stimulation paramters
duration= 2 * 1000 #seconds
frequency = 15 # Hz
isi = 67
phasic_on = 10 * 1000
afferent_train = np.zeros(len(tvec), dtype=bool) #  Rays stimulus = 30ms light pulses at 20Hz for 1 seconds duration.

for spike in np.arange(45):
    afferent_train[int((phasic_on/dt)+((spike*isi)/dt))] = True



lateralInhibition = 'simple'  # for non-clearing model (static tau_decay)
#lateralInhibition = 'complex' # for clearing model (tau_decay depends on g_i at time of spike)
li_scalar_vec = [0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004,  0.0045, 0.005]


simulate = False
quantify = False
plot_individual = False
plotAggregate =True
if simulate:
    plot_individual = False
    simulateModules( li_scalar_vec, neu_number, ds_fac)
if quantify:
    ex_in_df, isi_df = plotAndQuantify(plot_individual, li_scalar_vec, neu_number, ds_fac)
    #ex_in_df.to_csv(f'/home/nd17878/lc_code_git/results/thesis/phasic_lateral_inhibition/MODULAR_simulation_tenruns_lateral_inhib_vs_exin_difference_ratio_intermediate.csv')
    isi_df.to_csv(f'/home/nd17878/lc_code_git/results/thesis/phasic_lateral_inhibition/MODULAR_simulation_tenruns_lateral_inhib_vs_exin_difference_ratio_ISI.csv')
if plotAggregate:
    #ex_in_df = pd.read_csv(f'/home/nd17878/lc_code_git/results/thesis/phasic_lateral_inhibition/MODULAR_simulation_tenruns_lateral_inhib_vs_exin_difference_ratio_master.csv')
    isi_df = pd.read_csv(f'/home/nd17878/lc_code_git/results/thesis/phasic_lateral_inhibition/MODULAR_simulation_tenruns_lateral_inhib_vs_exin_difference_ratio_ISI.csv')

    aggregateISIPlot(isi_df)
