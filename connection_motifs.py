### 12/11/17- File of connection motifs for neuron_population code
### # row = pre sy neuron, column = post sy neuron
###  -1 = no connection, 1 = excitatory synapse, 2 = inhibitory synapse
import numpy as np

# creates connection matrix of all neurons to all others with excitatory synapses (no self connection)
def all_to_all_ex(neu_number):
    all_all = np.ones((neu_number, neu_number), dtype=np.int8)
    np.fill_diagonal(all_all, int(0))
    return(all_all)

# all to all motif as above, with all inhibitory synapses
def all_to_all_inhib(neu_number):
    all_all = np.full((neu_number, neu_number), 2, dtype=np.int8)
    np.fill_diagonal(all_all, int(0))
    return(all_all)

def sparse(neu_number, sparsity):   # Matrix of specified sparsity (given as number of connections each unit makes)
    if sparsity > neu_number:
        return [False]   # If there a less neurons than the given proportion of connections
    if neu_number == 0:
        sparse_mat = [0]
    else:
        sparse_mat = np.zeros((neu_number, neu_number), dtype=np.int8)   # Blank array
        for n in range(neu_number):     # For each neuron
            while sum(sparse_mat[n,:]) < sparsity:   # For given proportion of connections
                index = n-np.random.randint(1, neu_number)   # Generate random index of connection
                if index != n and sparse_mat[n,index] == 0:    # Must not be on self, i.e. diagonal must be zeros, must not already be filled
                    sparse_mat[n, index] = 1

    return sparse_mat

def modular_gaps():

    modular_gaps = np.array([[0, -1, 1, 0],
                             [-1, 0, 0, 1]
                             [0, 0, 0 ,-1]
                             [0, 0, -1, 0]])
    return modular_gaps

# Two mutual inhibiting neurons, each w one excitatory connection to 3rd (3rd has no outputs)
def two_mutual_one_receiver():
    mutual_in_one_ex = np.array([[-1, -1, -1],
                    [1, -1, 2],
                    [1, 2, -1]])
    return(mutual_in_one_ex)

# Two layers, one excitatory layer (no connections) to layer with mutual inhib of neighbours MUST BE EVEN No UNITS
def ex_to_MI(neu_number):
    n_per_layer = int(neu_number/2)
    ex_layer = np.zeros((n_per_layer, n_per_layer), dtype=np.int8)

    ex_to_i = np.full((n_per_layer, n_per_layer), 0, dtype=np.int8)
    np.fill_diagonal(ex_to_i, int(1))

    i_to_ex = np.zeros((n_per_layer, n_per_layer), dtype= np.int8)

    in_layer = np.zeros((n_per_layer, n_per_layer), dtype= np.int8)
    for n in range(n_per_layer):
        in_layer[n, n-(n_per_layer-1)] = int(2)
        in_layer[n, (n-1)] = int(2)

    ex_wiring = np.concatenate((ex_layer, ex_to_i), axis=1)    # constructs top half of matrix (excitatory intra, excitatory outputs)
    in_wiring = np.concatenate((i_to_ex, in_layer), axis=1)    # constructs bottom half of matrix (inhibitory intra, inhibitory outputs)
    wiring = np.concatenate((ex_wiring, in_wiring), axis=0)

    return(wiring)

def MI_two_layers(neu_number):  # must be even, one layer w MI neighbours, excitatory afferent to another MI layer
    n_per_layer = int(neu_number/2)

    layer_one = np.zeros((n_per_layer, n_per_layer), dtype=np.int8)

    for n in range(n_per_layer):
        layer_one[n, n-(n_per_layer-1)] = int(2)
        layer_one[n, (n-1)] = int(2)

    layer_two = layer_one

    one_to_two = np.zeros((n_per_layer, n_per_layer), dtype=np.int8)
    np.fill_diagonal(one_to_two, int(1))

    two_to_one = np.zeros((n_per_layer, n_per_layer), dtype=np.int8)

    wiring_one = np.concatenate((layer_one, one_to_two), axis=1)    # constructs top half of matrix (excitatory intra, excitatory outputs)
    wiring_two = np.concatenate((two_to_one, layer_two), axis=1)    # constructs bottom half of matrix (inhibitory intra, inhibitory outputs)
    wiring = np.concatenate((wiring_one, wiring_two), axis=0)

    return(wiring)



# Two layers, one excitatory layer (w. skip neighbour connections) to layer with mutual inhib of neighbours MUST BE EVEN No UNITS
def ex_SN_to_MI(neu_number):
    n_per_layer = int(neu_number/2)
    ex_layer = np.zeros((n_per_layer, n_per_layer), dtype=np.int8)

    for n in range(n_per_layer):
        ex_layer[n, n-(n_per_layer - 2)] = 1           # connects every Ex neuron (L1) to its neighbour at n+2
        ex_layer[n, (n-2)] = 1

    ex_to_i = np.zeros((n_per_layer, n_per_layer), dtype=np.int8)
    np.fill_diagonal(ex_to_i, int(1))

    i_to_ex = np.zeros((n_per_layer, n_per_layer), dtype= np.int8)

    in_layer = np.zeros((n_per_layer, n_per_layer), dtype= np.int8)
    for n in range(n_per_layer):
        in_layer[n, n-(n_per_layer-1)] = int(2)
        in_layer[n, (n-1)] = int(2)

    ex_wiring = np.concatenate((ex_layer, ex_to_i), axis=1)    # constructs top half of matrix (excitatory intra, excitatory outputs)
    in_wiring = np.concatenate((i_to_ex, in_layer), axis=1)    # constructs bottom half of matrix (inhibitory intra, inhibitory outputs)
    wiring = np.concatenate((ex_wiring, in_wiring), axis=0)

    return(wiring)


# Two layers, one excitatory layer, to layer with mutual inhib of neighbours AND ex. inputs (MUST BE EVEN No UNITS)
def ex_to_MI_and_return_in(neu_number):
    n_per_layer = int(neu_number/2)
    ex_layer = np.zeros((n_per_layer, n_per_layer), dtype=np.int8)

    ex_to_i = np.full((n_per_layer, n_per_layer), 0, dtype=np.int8)
    np.fill_diagonal(ex_to_i, int(1))

    i_to_ex = np.zeros((n_per_layer, n_per_layer), dtype= np.int8)
    np.fill_diagonal(i_to_ex, int(2))

    in_layer = np.zeros((n_per_layer, n_per_layer), dtype= np.int8)
    for n in range(n_per_layer):
        in_layer[n, n-(n_per_layer-1)] = int(2)
        in_layer[n, (n-1)] = int(2)

    ex_wiring = np.concatenate((ex_layer, ex_to_i), axis=1)    # constructs top half of matrix (excitatory intra, excitatory outputs)
    in_wiring = np.concatenate((i_to_ex, in_layer), axis=1)    # constructs bottom half of matrix (inhibitory intra, inhibitory outputs)
    wiring = np.concatenate((ex_wiring, in_wiring), axis=0)

    return(wiring)
