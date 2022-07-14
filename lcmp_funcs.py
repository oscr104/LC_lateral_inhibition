
''' Functions for multiprocessing parallelised LC simulations '''

import LC_unit_modular as LC_Um
import LC_unit_psam as LC_Up
import LC_vals_alt as LC
import lcmp_funcs as lcmpf
import connection_motifs as c_m
import numpy as np
import pandas as pd



def lat_i_func(na_timers, na_ws, na_prop, dt):
    li_sum = 0
    rel_count = len(na_timers)
    for n in np.arange(rel_count):
        t = na_timers[n]
        li_single = na_ws[n] * (((t - 0)/400) * np.exp(-(t-0)/400))
        #li_single = 0.01  * (((t - 0)/400) * np.exp(-(t-0)/400)) # for SIMPLE no calcium dependent release
        li_sum += li_single
        na_timers[n] += dt
    return li_sum


def make_module(neu_number, gnum, gw):
    # Parameter initial state arrays
    #Conductances
    g_na = [LC.g_na] * neu_number  # Sodium or TTX
    g_k = [LC.g_k]  * neu_number   # Potassium
    g_l = [LC.g_l]  * neu_number  # Leak current
    g_ld = [LC.g_ld]  * neu_number # dendritic leak current
    g_p = [LC.g_p]  * neu_number  # Slow persistent sodium current
    g_ahp = [LC.g_ahp]  * neu_number  # afterhyperpolarisation current
    g_ca = [LC.g_ca]  * neu_number # Calcium current
    g_d = [LC.g_d]  * neu_number   # Dendrite-soma conductance
    g =  [gw * LC.g_strong]  * neu_number # choose single, weak or strong gap conductance  NORMALISED SO THAT P1= 1.0  -> strong gap j conducatance as per Alvarez (10 single gap channels)
    #Capacitances
    c_s = [LC.c_s]  * neu_number  # Somatic C
    c_d = [LC.c_d]  * neu_number  # Dendritic C
    #Voltages
    v_na = [LC.v_na]  * neu_number # sodium equilibrium voltage
    v_k = [LC.v_k]  * neu_number # Potassium equilibrium voltage
    v_ca = [LC.v_ca]  * neu_number #Calcium resting voltage
    v_l = [LC.v_l]   * neu_number # resting Vm
    v_ld = [LC.v_ld]  * neu_number # dendritic resting Vm
    v_i = [LC.v_i]  * neu_number  # rev potential for LC synaptic inhibition - Patel & Joshi 2015 / Egan 1983
    # Activation values
    m = [LC.m]  * neu_number
    h = [LC.h]  * neu_number
    n = [LC.n]  * neu_number
    p = [LC.p]  * neu_number
    q = [LC.q]  * neu_number
    r = [LC.r]  * neu_number
    m_ca = [LC.m_ca]  * neu_number
    y = [LC.y]  * neu_number
    #Calcium concentration
    ca_init = [LC.ca_conc]  * neu_number
    li_w = [1]  * neu_number

    # Array of lc units
    neurons = []
    for x in range(neu_number):
        unit = LC_Um.LC_unit(g_na[x], g_k[x], g_l[x], g_ld[x], g_p[x], g_ahp[x], g_ca[x], g_d[x], g[x], c_s[x], c_d[x], v_na[x], v_k[x], v_ca[x], v_l[x], v_ld[x], v_i[x], m[x], h[x], n[x], p[x], q[x],r[x], m_ca[x], y[x], ca_init[x], li_w[x])
        neurons.append(unit)
    # wiring matrices
    if gnum == True:
        wiring = c_m.all_to_all_ex(neu_number)
    else:
        wiring = c_m.sparse(neu_number, gnum) #gap junction connection all to all
    if wiring.any == False:
        print('Problem with wiring matrix, check parameters.')
    print(wiring)

    return neurons, wiring


def make_module(neu_number, gnum, gw):
    # Parameter initial state arrays
    #Conductances
    g_na = [LC.g_na] * neu_number  # Sodium or TTX
    g_k = [LC.g_k]  * neu_number   # Potassium
    g_l = [LC.g_l]  * neu_number  # Leak current
    g_ld = [LC.g_ld]  * neu_number # dendritic leak current
    g_p = [LC.g_p]  * neu_number  # Slow persistent sodium current
    g_ahp = [LC.g_ahp]  * neu_number  # afterhyperpolarisation current
    g_ca = [LC.g_ca]  * neu_number # Calcium current
    g_d = [LC.g_d]  * neu_number   # Dendrite-soma conductance
    g =  [gw * LC.g_strong]  * neu_number # choose single, weak or strong gap conductance  NORMALISED SO THAT P1= 1.0  -> strong gap j conducatance as per Alvarez (10 single gap channels)
    #Capacitances
    c_s = [LC.c_s]  * neu_number  # Somatic C
    c_d = [LC.c_d]  * neu_number  # Dendritic C
    #Voltages
    v_na = [LC.v_na]  * neu_number # sodium equilibrium voltage
    v_k = [LC.v_k]  * neu_number # Potassium equilibrium voltage
    v_ca = [LC.v_ca]  * neu_number #Calcium resting voltage
    v_l = [LC.v_l]   * neu_number # resting Vm
    v_ld = [LC.v_ld]  * neu_number # dendritic resting Vm
    v_i = [LC.v_i]  * neu_number  # rev potential for LC synaptic inhibition - Patel & Joshi 2015 / Egan 1983
    # Activation values
    m = [LC.m]  * neu_number
    h = [LC.h]  * neu_number
    n = [LC.n]  * neu_number
    p = [LC.p]  * neu_number
    q = [LC.q]  * neu_number
    r = [LC.r]  * neu_number
    m_ca = [LC.m_ca]  * neu_number
    y = [LC.y]  * neu_number
    #Calcium concentration
    ca_init = [LC.ca_conc]  * neu_number
    li_w = [1]  * neu_number

    # Array of lc units
    neurons = []
    for x in range(neu_number):
        unit = LC_p.LC_unit(g_na[x], g_k[x], g_l[x], g_ld[x], g_p[x], g_ahp[x], g_ca[x], g_d[x], g[x], c_s[x], c_d[x], v_na[x], v_k[x], v_ca[x], v_l[x], v_ld[x], v_i[x], m[x], h[x], n[x], p[x], q[x],r[x], m_ca[x], y[x], ca_init[x], li_w[x])
        neurons.append(unit)
    # wiring matrices
    if gnum == True:
        wiring = c_m.all_to_all_ex(neu_number)
    else:
        wiring = c_m.sparse(neu_number, gnum) #gap junction connection all to all
    if wiring.any == False:
        print('Problem with wiring matrix, check parameters.')
    print(wiring)

    return neurons, wiring

def make_psam_module(neu_number, gnum, gw, params):


    # Array of lc units
    neurons = []
    for x in range(neu_number):
        unit = LC_Up.LC_unit(params)
        neurons.append(unit)
    # wiring matrices
    if gnum == True:
        wiring = c_m.all_to_all_ex(neu_number)
    else:
        wiring = c_m.sparse(neu_number, gnum) #gap junction connection all to all
    if wiring.any == False:
        print('Problem with wiring matrix, check parameters.')
    print(wiring)

    return neurons, wiring


def singlestep(neuron, input_current, spk_vec, v_d_vec, ca_vec, dt):
    spk, v_s, v_d, ca_conc = neuron.update(input_current, v_d_vec, ca_vec, dt, False, False, spk_vec)
    return spk, v_s, v_d, ca_conc

def simulate(neu_number, tvec, ie_vec):
    module, wiring = make_module(neu_number)
    dt = tvec[1]
    results = np.zeros([len(tvec), len(module), 4])
    k = 0 #time index
    for t in tvec:
        p = 0 # neuron index
        for n in module:
            i_e = ie_vec[k, p]
            spk_vec =  results[k-1, :, 0]
            v_d_vec = results[k-1, :, 2]
            ca_vec = results[k-1, :, 3]
            spike, v_s, v_d, ca_conc = singlestep(n, 1., spk_vec, v_d_vec, ca_vec, dt)
            results[k, p, :] = spike, v_s, v_d, ca_conc
            p += 1
        k+= 1
    export_res = results[0:-1:10,:,:]
    return export_res


def simulate_p(params):
    neu_num = params[0]
    tvec = params[1]
    ie = params[2]
    gnum = params[3]
    gw = params[4]
    print(f'simulating {neu_num} units for {tvec[-1]}s, i.e. = {ie}, gap num = {gnum}, gap weight = {gw}')
    module, wiring = make_module(neu_num, gnum, gw)
    dt = tvec[1]
    results = np.zeros([len(tvec), len(module), 4])
    k = 0 #time index
    for t in tvec:
        p = 0 # neuron index
        for n in module:
            spk_vec =  results[k-1, :, 0]
            v_d_vec = results[k-1, :, 2]
            ca_vec = results[k-1, :, 3]
            spike, v_s, v_d, ca_conc = singlestep(n, ie, spk_vec, v_d_vec, ca_vec, dt)
            results[k, p, :] = spike, v_s, v_d, ca_conc
            p += 1
        k+= 1
    export_res = results[0:-1:10,:,:]
    return export_res

def simulate_m(modules, wiring, params):
    ''' runs simulation for modular population, taking arguments of initiated modules & wiring schemes, parameters, and initial state results data frame.
    calling functions singlestep_m to implement individual dts, and appends output variables to results dataframe
    '''
    print(modules)
    pd.options.mode.chained_assignment = None  # default='warn'  I THINK THIS IS SAFE - pandas gets anxious about altering values in a copy of a slice from the original results_df, but thats exactly what I want here, so this shuts the warning up

    neu_num = params['neu_num']
    mod_num = params['mod_num']
    tvec = params['tvec']
    dt = tvec[1]
    row_idx = tvec[::10]
    ie = params['ie']
    gnum = params['gnum']
    gw = params['gw']
    mod_idx = np.arange(mod_num)

    out_vars=['Spike', 'V_s' 'V_d', '[Ca]', 'g_a2']

    mdx = pd.MultiIndex.from_product((np.arange(mod_num), np.arange(neu_num), out_vars), names=['module', 'neuron', 'variable'])

    results_df = pd.DataFrame(index=row_idx, columns=mdx)
    results_df.loc[tvec[0]] = [False, LC.v_l, LC.v_ld, LC.ca_conc, 0]*len(modules)
    step_res_df = results_df.loc[tvec[0]]
    print(f'step res df: {step_res_df}')


    print(f'simulating {mod_num} modules w/ {neu_num} units each, for {tvec[-1]}ms, i.e. = {ie}, gap num = {gnum}, gap weight = {gw}')

    k = 0 #time index
    r = 1 #results array index, increases 1 in 10 k increases
    for t in tvec[:-1]:

        p = 0 # module counter for indexing
        for m in modules:
            idx = pd.IndexSlice
            spk_vec = step_res_df.loc[idx[mod_idx[p-1], :, 'Spike']]
            v_d_vec = step_res_df.loc[idx[mod_idx[p], :, 'V_d']] # v_d from same module
            ca_vec = step_res_df.loc[idx[mod_idx[p], :, '[Ca]']]# ca_vec
            res_vec = singlestep_m(m, wiring[p], ie, spk_vec, v_d_vec, ca_vec, dt)
            ######print(f'spike into : {step_res_df.loc[idx[mod_idx[p], :, 'Spike']]}')
            step_res_df.loc[idx[mod_idx[p], :, 'Spike']] = res_vec[0::5]
            step_res_df.loc[idx[mod_idx[p], :, 'V_s']] = res_vec[1::5]
            step_res_df.loc[idx[mod_idx[p], :, 'V_d']] = res_vec[2::5]
            step_res_df.loc[idx[mod_idx[p], :, '[Ca]']] = res_vec[3::5]
            step_res_df.loc[idx[mod_idx[p], :, 'g_a2']] = res_vec[4::5]
            p += 1
        if k % 10 == 0:
            results_df.loc[row_idx[r]] = step_res_df
            r += 1
        k+= 1

    return results_df



def simulate_opto_s(neuron, init_state, params, spike_vec):
    ''' runs simulation for optostim reciever neuron (non transduced cells in Ray Perrins data), taking arguments of initiated modules & wiring schemes, parameters,
     and initial state results data frame, as well as T/F time series representing optostim spikes..
    calling functions singlestep_m to implement individual dts, and appends output variables to results dataframe
    '''

    tvec = params['tvec']
    dt = tvec[1]
    ds_fac = 1 # downsample factor for saving results
    row_idx = tvec[:-1:ds_fac]
    ie_vec = params['ie_vec']
    gnum = params['gnum']
    gw = params['gw']
    ttx = False
    carb = False

    #out_vars = ['Spike', 'V_s', 'V_d', '[Ca]', 'I_a2']
    rows = len(row_idx)
    cols = len(init_state)
    res_arr = np.zeros((rows,cols), dtype=float)
    res_arr[0,:] = init_state

    print(f'simulating one unit for {tvec[-1]}ms, gap num = {gnum}, gap weight = {gw}')

    k = 0 #time index
    r = 1 # results_df index counter, advances once for every 10 time steps
    for t in tvec[:-1]:
        spk_in = spike_vec[k]
        ie = ie_vec[k]
        step_res = neuron.update(ie, dt, ttx, carb, spk_in)
        if k % ds_fac == 0 and t != row_idx[-1]:
            res_arr[r] = step_res[:]
            r += 1
        k+= 1

    out_vars = ['V_s', 'V_d', '[Ca]', 'I_a2', 'NA']
    results_df = pd.DataFrame(res_arr,index=row_idx, columns=out_vars, dtype=float)

    return results_df


def simulate_opto_m(module, wiring, init_state, mod_params, led_vec):
    ''' runs simulation for optostim reciever module (non transduced cells in Ray Perrins data), taking arguments of initiated module & wiring scheme, parameters,
     and initial state results data frame, as well as T/F time series representing optostim spikes..
    calling functions singlestep_m to implement individual dts, and appends output variables to results dataframe

    args:
        module = list of neurons
        wiring = binary connection matrix - 1=gap connection, 0 = no connections
        init_state = [v_l, v_ld, ca_conc, spike]
        mod_params =  {'tvec':tvec, 'ie':ie, 'gnum':gnum, 'gw':gw}
        spike_train = boolean vector, one entry per time step
    '''

    tvec = mod_params['tvec']
    dt = tvec[1]
    ds_fac = 10 # downsample factor for saving results
    row_idx = tvec[::ds_fac]
    ie = mod_params['ie']
    gnum = mod_params['gnum']
    gw = mod_params['gw']
    gap_junctions = False
    ca_concs = False
    ttx = False
    carb = False

    rows = len(row_idx)
    cols = len(init_state) * len(module)

    res_arr = np.zeros((rows,cols), dtype=float)
    step_res = np.tile(init_state, len(module))
    res_arr[0,:] = step_res



    print(f'simulating module of {len(module)} units for {tvec[-1]}ms by {dt}ms steps, i.e. = {ie}, gap num = {gnum}, gap weight = {gw}')

    k = 0 #time index
    r = 1 # results_df index counter, advances once for every 10 time steps

    for t in tvec[:-1]:
        led_on = led_vec[k]
        v_d_vec = step_res[1::5]
        ca_vec = step_res[2::5]
        a2_vec = step_res[3::5]
        step_res = singlestep_m(module, wiring, ie, led_on, v_d_vec, ca_vec, a2_vec, dt)

        if k % ds_fac == 0 and t != row_idx[-1]:
            res_arr[r] = step_res[:]
            r += 1
        k+= 1

    out_vars = ['V_s', 'V_d', '[Ca]', 'g_a2', 'i_a2']
    neu_list = [n for n in np.arange(len(module))]
    col_idx = pd.MultiIndex.from_product([neu_list,out_vars], names=('neuron', 'variable'))
    results_df = pd.DataFrame(res_arr,index=row_idx, columns=col_idx, dtype=float)

    return results_df



def singlestep_m(module, wiring, ie, led_on, v_d_vec, ca_vec, a2_vec, dt):
    res_vec = [np.nan, np.nan, np.nan, np.nan, np.nan] * len(module)
    for unit in range(len(module)):   # for each neuron
        ie_n = np.random.normal(loc=ie, scale=0.05)
        connections = np.flatnonzero(wiring[unit,:])  # find indexs of gap Js involving this N]
        if sum(connections) > 0:
            gap_voltages =[]
            gap_ca = []
            g_a2_in = []
            for conn in connections:
                gap_voltages.append(v_d_vec[conn]) # the relevant dendritic voltage is the value of v_d_vec at the index specified by connections matrix
                gap_ca.append(ca_vec[conn])    # ditto for Ca differnces
                g_a2_in.append(a2_vec[conn])
        else:
            gap_voltages = []
            gap_ca = []
        v_s, v_d, i_leak, i_k, i_na, i_p, i_ahp, i_ca, i_mem,  ca_conc, i_a2, g_a2, spike = module[unit].update(ie_n, gap_voltages, dt, False, False, led_on, g_a2_in)

        res_vec[unit*5:unit*5+5] = [spike, v_s, v_d, ca_conc, i_a2]

    return res_vec
