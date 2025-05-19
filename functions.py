import numpy as np
import scipy.linalg as la
import datetime, timeit
import random as rd
import numpy.random as nprd
import shutil
terminal_width, _ = shutil.get_terminal_size()
from functools import reduce
import pickle
from scipy.stats import unitary_group
from tqdm import tqdm
from scipy.linalg import fractional_matrix_power as fmp


from numba import njit, prange#, config,

# set the threading layer before any parallel target compilation
# config.THREADING_LAYER = 'threadsafe'

def print_matrix(matr, precision=4):
    s = [[str(e) if abs(e) > 1e-15 else '.' for e in row] for row in np.round(matr,precision)]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens if x != 0) or '.'
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))

##########################################################################
# Timekeeping functions
##########################################################################

def begin(verbosity=1):
    if verbosity==2:
        print('\n')
        print('*' * terminal_width)
        print(' '*4, 'Started:', datetime.datetime.now())
    return None

def finish(verbosity=1):
    if verbosity==2:
        print(' '*4, 'Finished:', datetime.datetime.now())
        print('*' * terminal_width)
        print('\n')
    return None

def tic(task_description_string, verbosity=1):
    if verbosity==2:
        print('-' * terminal_width)
        print('---> ' + task_description_string)
    return timeit.default_timer()

def toc(tic_time, verbosity=1):
    toc_time = timeit.default_timer()
    if verbosity==2:
        print(' '*4, round(toc_time - tic_time, 6), 'seconds')
    return toc_time

#########################################################################
I = np.diag([1, 
               1])
X = np.array([[0,1],
              [1,0]])
Y = np.array([[0,-1j],
              [1j,0]])
Z = np.array([[1, 0],
              [0,-1]])
M = np.array([[0,1],
              [0,0]])
P = np.array([[0,0],
              [1,0]])
UP = np.array([1,0])
DOWN = np.array([0,1])

II = np.kron(I,I)
IX = np.kron(I,X)
XI = np.kron(X,I)
IZ = np.kron(I,Z)
ZI = np.kron(Z,I)
XX = np.kron(X,X)
YY = np.kron(Y,Y)
ZZ = np.kron(Z,Z)
PM = np.kron(P,M)
MP = np.kron(M,P)

#########################################################################

def get_masks(N, first_qubit, K=2):
    """
    Return an array of shape (2**K,) of index-arrays (masks), each selecting those
    computational-basis states (0..2**N-1) whose bits at positions
    first_qubit, first_qubit+1, ..., first_qubit+K-1 (mod N) equal one of the
    2**k possible patterns.

    Qubits are numbered from 0 (LSB) to N-1 (MSB).  A mask is the sorted list of
    integers whose binary representation has the specified K-bit pattern in those
    positions.
    
    K is the locality of the mask, i.e. the number of qubits in the window.
    """
    comp_basis = np.arange(2**N)
    masks = []

    # generate all k‐bit patterns 0..2**k-1
    for pattern in range(2**K):
        idx = comp_basis
        # for each qubit in the window, filter idx by whether that bit matches
        for offset in range(K):
            q = (first_qubit + offset) % N # qubit index
            want = (pattern >> offset) & 1 # get the bit of pattern at offset
            idx = idx[(idx // 2**q) % 2 == want]
        masks.append(idx)

    return np.array(masks, dtype=object)


def get_masks_typed(N, first_qubit, K):
    comp = np.arange(2**N, dtype=np.int64)
    D = 2**K
    M = (2**N) // D
    masks = np.empty((D, M), dtype=np.int64)

    for pat in prange(D):
        idx = comp
        for offset in range(K):
            q   = (first_qubit + offset) % N
            bit = (pat >> offset) & 1
            idx = idx[(idx // (1 << q)) % 2 == bit]
        masks[pat, :] = idx

    return masks

def apply_gate(state, gate, masks):
    '''
    Apply a gate to the state

    Parameters:
    - state: state vector on full Hilbert space
    - gate: 4x4 matrix corresponding to a 2-qubit gate
    - masks: list of 4 elements
        . first element contains the indices to treat as |00>
        . second element contains the indices to treat as |01>
        . third element contains the indices to treat as |10>
        . fourth element contains the indices to treat as |11>
    '''
    state_fin = np.zeros_like(state, dtype=complex)

    # Split the state in its four components
    state_split = state[masks]

    # Apply gate to state
    state_fin[masks] =  np.matmul(gate, state_split)[:,]

    return state_fin
    

@njit(parallel=True, fastmath=True, cache=True)
def apply_gate(state, gate, masks):
    '''
    Apply a gate to the state

    Parameters:
    - state: state vector on full Hilbert space
    - gate: 4x4 matrix corresponding to a 2-qubit gate
    - masks: list of 4 elements
        . first element contains the indices to treat as |00>
        . second element contains the indices to treat as |01>
        . third element contains the indices to treat as |10>
        . fourth element contains the indices to treat as |11>
    '''
    state_fin = np.zeros_like(state, dtype=np.complex128)
    num_elements = len(masks[0]) # 2^N/4
    for idx in prange(num_elements):
        i0, i1, i2, i3 = masks[0][idx], masks[1][idx], masks[2][idx], masks[3][idx]
        s0, s1, s2, s3 = state[i0], state[i1], state[i2], state[i3]
        t0 = gate[0,0]*s0 + gate[0,1]*s1 + gate[0,2]*s2 + gate[0,3]*s3
        t1 = gate[1,0]*s0 + gate[1,1]*s1 + gate[1,2]*s2 + gate[1,3]*s3
        t2 = gate[2,0]*s0 + gate[2,1]*s1 + gate[2,2]*s2 + gate[2,3]*s3
        t3 = gate[3,0]*s0 + gate[3,1]*s1 + gate[3,2]*s2 + gate[3,3]*s3
        state_fin[i0], state_fin[i1], state_fin[i2], state_fin[i3] = t0, t1, t2, t3
    return state_fin

@njit(parallel=True, fastmath=True, cache=True)
def apply_gate_k(state, gate, masks):
    """
    Apply a K-local gate to an N-qubit state vector.

    Parameters
    ----------
    state : complex128[2**N]
        The input state vector.
    gate : complex128[2**K, 2**K]
        The K-qubit gate to apply.
    masks : int64[2**K, M], M = 2**N / 2**K
        masks[c] is the array of all basis-state indices whose
        local K-bit pattern equals the integer c (0 ≤ c < 2**K).

    Returns
    -------
    state_out : 1D complex128 array, length = 2**N
        The output state, with the K-qubit gate applied in place.
    """
    D, M = masks.shape         # D = 2**K,  M = 2**N / 2**K
    out = np.zeros_like(state)

    for b in prange(M):
        # gather
        amp = np.empty(D, dtype=np.complex128)
        for c in range(D):
            amp[c] = state[masks[c, b]]

        # apply
        res = gate.dot(amp)

        # scatter
        for c in range(D):
            out[masks[c, b]] = res[c]

    return out

def apply_U(state, gates, gate_ordering_idx_list, masks_dict, K=None):
    '''
    Apply the Floquet operator to the state psi 2-qubit gate at a time

    Parameters:
    - state: state vector on full Hilbert space
    - gates: list of matrices. each is a 2-qubit gate
    - gate_ordering_idx_list: list of indeces correspoding to the
                              order the gates wil be applied:
                              eg. i -> gate_{i,i+1}
    - masks_dict: dictionary containing N masks defining how a gate on
                  2 consecutive sites needs to be applied
    '''
    for gate_idx, order_idx in enumerate(gate_ordering_idx_list):
        if K is not None:
            try:
                state = apply_gate_k(state, gates[gate_idx], masks_dict[order_idx])
            except:
                print('Error applying gate')
                print('len gates:', len(gates))
                print('gate:', gate_idx)
                print('list:', gate_ordering_idx_list)
                print('order:', order_idx)
                print('masks:', masks_dict[order_idx])
                raise
        else:
            state = apply_gate(state, gates[gate_idx], masks_dict[order_idx])
    return state

#########################################################################

def get_magn(state):
    '''
    Calculate the magnetization of the state
    '''
    N = int(np.log2(len(state)))
    M = 0
    for n in range(N):
        paulis = [I]*N
        paulis[n] = Z
        s_z_n = reduce(np.kron, paulis)
        M += (state.conj().T @ s_z_n @ state).real
    return M

def r_y(theta):
    return np.cos(theta/2)*I - 1j*np.sin(theta/2)*Y

def initial_entangled_state(N, theta, state_phases):
    """
    Generate the state |\psi> = \bigotimes_{i=0}^{N/2-1} |\Psi^+>_{(N/2 - i), (N/2 + i)}
    for an N-qubit system, where N is even.

    Parameters:
        N (int): Total number of qubits, must be even.

    Returns:
        numpy.ndarray: State vector of the entangled system.
    """
    def generate_initial_order(N):
        """Generate the initial qubit order [0, N-1, 1, N-2, ...]."""
        order = []
        for i in range(N // 2):
            order.append(i)
            order.append(N - 1 - i)
        if N % 2 == 1:
            order.append(N // 2)
        return order

    def resort_state_vector_general(state_vector, N):
        """Resort a state vector with N qubits from [0, N-1, 1, N-2, ...] to canonical [0, 1, 2, ...]."""
        num_states = len(state_vector)
        num_qubits = int(np.log2(num_states))
        
        assert num_qubits == N, "State vector size does not match qubit count."
        assert 2**N == num_states, "State vector size is not a power of 2."

        # Generate initial qubit order and its inverse mapping
        initial_order = generate_initial_order(N)

        # New state vector
        new_state_vector = np.zeros_like(state_vector, dtype=complex)

        # Permute indices
        for i in range(num_states):
            # Convert index to binary, rearrange bits, then convert back to integer
            binary = f"{i:0{N}b}"  # Binary representation with padding
            rearranged_binary = "".join(binary[initial_order.index(j)] for j in range(N))
            new_index = int(rearranged_binary, 2)
            new_state_vector[new_index] = state_vector[i]

        return new_state_vector

    states = []
    for n in range(N//2):
        if state_phases == 'homogenous':
            phase = 1
        elif state_phases == 'staggered':
            phase = (-1)**n
        st = np.sin(theta * phase)
        ct = np.cos(theta * phase)
        st_1 = np.sin(theta * phase/2)
        st_2 = np.sin(theta * phase/2)
        ct_2 = np.cos(theta * phase/2)
        # if n != 4:
        #     state = np.array([-st, ct, ct, st])/np.sqrt(2) # ry(theta)ry(theta) |Psi+>
        # else:
        state = np.array([ct, st, st, -ct])/np.sqrt(2) # ry(theta)ry(theta) |Phi->, not preserving M
            # state = np.array([ct_2**2, st_1/2, st_1/2, st_2**2]) # ry(theta)ry(theta) |00>

        states.append(state)

    unsorted_state = reduce(np.kron, states)
    return resort_state_vector_general(unsorted_state, N)

def initial_state(N, qkeep, theta, state_phases):
    '''
    Create the initial state as (exp(-i theta/2 sigma_y) |0>)^\otimes N
    '''

    states_n = []
    for n in range(N):
        if n not in qkeep:
            theta_ = .0 * np.pi #np.pi/2
        else:
            theta_ = theta
        if state_phases == 'homogenous':
            state_n = r_y(theta_) @ np.array([1, 0])
        elif state_phases == 'staggered':
            state_n = r_y(theta_) @ (np.array([1, 0]) if n%2==0 else np.array([0, 1]))
        
        states_n.append(state_n)

    state = reduce(np.kron, states_n)

    return state

def ptrace(rho, qkeep):
    N = int(np.log2(rho.shape[0]))
    rd = [2,] * N
    qkeep = list(np.sort(qkeep))
    dkeep = list(np.array(rd)[qkeep])
    qtrace = list(set(np.arange(N))-set(qkeep))
    dtrace = list(np.array(rd)[qtrace])
    if len(rho.shape) == 1: # if rho is ket
        temp = (rho
                .reshape(rd) # convert it to 2x2x2x...x2
                .transpose(qkeep+qtrace) # leave sites to trace as last
                .reshape([np.prod(dkeep),np.prod(dtrace)])) # dkeep x dtrace 
        partial_rho = temp.dot(temp.conj().T) 
    else : # if rho is density matrix
        partial_rho = np.trace(rho
                      .reshape(rd+rd)
                      .transpose(qtrace+[N+q for q in qtrace]+qkeep+[N+q for q in qkeep])
                      .reshape([np.prod(dtrace),np.prod(dtrace),
                                np.prod(dkeep),np.prod(dkeep)]))
    return partial_rho

def gen_u1(params=None):
    if params is not None:
        if len(params) == 2: params += [0, 0, 0]
        return gate_xxz_disordered(*params)
    gate = np.zeros((4,4), dtype=complex)
    gate[0,0] = np.exp(1j*np.random.rand()*2*np.pi)
    gate[3,3] = np.exp(1j*np.random.rand()*2*np.pi)
    gate[1:3,1:3] = unitary_group.rvs(2)
    return gate

def gen_su2(J=None):
    if J is None:
        J = np.random.rand()*np.pi
    swap = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]])
    
    gate = np.eye(4) * np.cos(J/2) - 1j * np.sin(J/2) * swap
    return gate

def gen_gates_order(N, geometry='random', boundary_conditions='PBC', eo_first='True'):
    # Generate the order the gates will be applied
    if geometry == 'random':
        if boundary_conditions == 'PBC':
            return rd.sample([n for n in range(N)],N)
        elif boundary_conditions == 'OBC':
            return rd.sample([n for n in range(N-1)],N-1)
    if geometry != 'brickwork':
        raise ValueError('Only random and brickwork geometries are supported')
    gate_ordering_idx_list = []
    if eo_first:
        for n in range(N):
            if n % 2 == 0:
                if n == N-1 and boundary_conditions == 'OBC':
                    continue
                else:
                    gate_ordering_idx_list.append(n)
    for n in range(N):
        if n % 2 == 1:
            if n == N-1 and boundary_conditions == 'OBC':
                continue
            else:
                gate_ordering_idx_list.append(n)
    if not eo_first:
        for n in range(N):
            if n % 2 == 0:
                if n == N-1 and boundary_conditions == 'OBC':
                    continue
                else:
                    gate_ordering_idx_list.append(n)

    return np.array(gate_ordering_idx_list, dtype=int)

def vNE(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-10]


    return -np.sum(eigvals*np.log2(eigvals))


def gen_Ls(N_A, circuit_type):
    if circuit_type == 'u1':
        L = np.zeros((2**N_A, 2**N_A), dtype=complex)
        for n in range(N_A):
            paulis = [I]*N_A
            paulis[n] = Z
            L += reduce(np.kron, paulis)
        return [L]
    elif circuit_type == 'su2':
        Ls = []
        for PAULI in [X, Y, Z]:
            for n in range(N_A):
                L = np.zeros((2**N_A, 2**N_A), dtype=complex)
                paulis = [I]*N_A
                paulis[n] = PAULI
                L += reduce(np.kron, paulis)
            Ls.append(L)
        return Ls

def WY(rho_A, Ls):
    # compute Wigner-Yanase skew information
    rho_A_sqrt = la.sqrtm(rho_A)
    WYs = []
    for L in Ls:
        WYs.append(
            np.trace(L @ rho_A @ L) - np.trace(rho_A_sqrt @ L @ rho_A_sqrt @ L))
    if len(WYs) == 1:
        return WYs[0].real
    return np.array(WYs).real

def gen_QFI(rho_A, Ls, ss):
    # compute QFI information
    # ss is a list of parameters s, where s=0 for SLD and s=1 for RLD and s=.5 for WY
    rho_A_sqrt = la.sqrtm(rho_A)
    QFIs = np.zeros((len(Ls), len(ss)))
    for Lidx, L in enumerate(Ls):
        LrhoL = np.trace(L @ rho_A @ L)
        for sidx, s in enumerate(ss):
            rhoLrhoL = np.trace(fmp(rho_A, s) @ L @ fmp(rho_A, 1-s) @ L)
            QFIs[Lidx, sidx] = (LrhoL - rhoLrhoL).real
    return QFIs

def load_mask_memory(N, K=2):
    '''
    Load the mask memory for a given N and K
    '''
    mask_dict = {}
    for n in range(N):
        mask_dict[n] = get_masks_typed(N, n, K)
    return mask_dict

def gen_Q(N, N_A=None):
    '''
    Generate the Q matrix composed by the projectors on the sectors of different
    magnetization values
    '''
    # if f'N{N}.pkl' in os.listdir('mask_memory'):
    #     DB = pickle.load(open(f'mask_memory/N{N}.pkl', 'rb'))
    #     mask_dict = DB['mask_dict']
    #     states_per_sector = DB['states_per_sector']
    #     qs = DB['qs']
    #     if N_A is None:
    #         return mask_dict
    #     Q = DB['Q']
    #     return mask_dict, qs, states_per_sector, Q
    
    if N_A is None:
        print('N_A must be specified if the mask memory is not available yet')
        mask_dict = {}
        for n in range(N):
            mask_dict[n] = get_masks(N, n)
        return mask_dict
    
    mask_dict = {}
    for n in range(N):
        mask_dict[n] = get_masks(N, n)
        
    computational_basis = np.arange(2**N_A)
    basis = np.array([bin(i).count('1') for i in computational_basis], dtype=int)

    states_per_sector = {}
    Q = np.zeros((2**N_A, 2**N_A), dtype=complex)    
    qs = []
    for M_A in range(N_A+1):
        temp_comp_states = computational_basis[basis == M_A]
        states_per_sector[M_A] = temp_comp_states
        vector = np.zeros(2**N_A)
        vector[temp_comp_states] = 1
        qs.append(np.outer(vector, np.conj(vector)))

    Q = reduce(np.add, qs)

    data_to_save = {'mask_dict': mask_dict,
                    'qs': qs,
                    'states_per_sector': states_per_sector,
                    'Q': Q}   

    with open(f'mask_memory/N{N}.pkl', 'wb') as file:
        pickle.dump(data_to_save, file)

    return mask_dict, qs, states_per_sector, Q

@njit(parallel=True, fastmath=True, cache=True)
def apply_u1(state, gate, masks):
    '''
    Apply a gate to the state

    Parameters:
    - state: state vector on full Hilbert space
    - gate: 4x4(1x1,2x2,1x1) matrix corresponding to a 2-qubit gate
    - masks: list of 4 elements
        . first element contains the indices to treat as |00>
        . second element contains the indices to treat as |01>
        . third element contains the indices to treat as |10>
        . fourth element contains the indices to treat as |11>
    '''
    state_fin = np.zeros_like(state, dtype=np.complex128)
    num_elements = len(masks[0]) # 2^N/4
    for idx in prange(num_elements):
        i0, i1, i2, i3 = masks[0][idx], masks[1][idx], masks[2][idx], masks[3][idx]
        s0, s1, s2, s3 = state[i0], state[i1], state[i2], state[i3]
        state_fin[i0] = gate[0,0]*s0
        state_fin[i1] = gate[1,1]*s1 + gate[1,2]*s2
        state_fin[i2] = gate[2,1]*s1 + gate[2,2]*s2
        state_fin[i3] = gate[3,3]*s3
    return state_fin

@njit(parallel=True, fastmath=True, cache=True)
def apply_su2(state, gate, masks):
    '''
    Apply a gate to the state

    Parameters:
    - state: state vector on full Hilbert space
    - gate: 4x4(1x1,1x1,1x1,1x1) matrix corresponding to a 2-qubit gate
    - masks: list of 4 elements
        . first element contains the indices to treat as |00>
        . second element contains the indices to treat as |01>
        . third element contains the indices to treat as |10>
        . fourth element contains the indices to treat as |11>
    '''
    state_fin = np.zeros_like(state, dtype=np.complex128)
    num_elements = len(masks[0]) # 2^N/4
    for idx in prange(num_elements):
        i0, i1, i2, i3 = masks[0][idx], masks[1][idx], masks[2][idx], masks[3][idx]
        state_fin[i0] = gate[0,0]*state[i0]
        state_fin[i1] = gate[1,2]*state[i1]
        state_fin[i2] = gate[2,1]*state[i2]
        state_fin[i3] = gate[3,3]*state[i3]
    return state_fin

def gate_xxz_disordered(J, Jz, h1, h2, h3, h4):
    """Return the unitary matrix for the disordered XXZ model."""
    U_H1 = np.diag(np.exp(-.5j*np.array([h1+h2, h1-h2, h2-h1, -h1-h2])))
    U_H2 = np.diag(np.exp(-.5j*np.array([h3+h4, h3-h4, h4-h3, -h3-h4])))
    U_XX = II * np.cos(J) - 1.0j * XX * np.sin(J)
    U_YY = II * np.cos(J) - 1.0j * YY * np.sin(J)
    U_ZZ = II * np.cos(Jz) - 1.0j * ZZ * np.sin(Jz)
    U_XXZ = U_XX @ U_YY @ U_ZZ
    return U_H1 @ U_XXZ @ U_H2

def gate_xxz_disordered(J, Jz, h1, h2, phi):
    ''' phase diagram [0,Pi] x [0,Pi]
    SWAP at J = pi
    '''
    U_H1 = np.diag(np.exp(-.5j*np.array([h1+h2, h1-h2, h2-h1, -h1-h2])))
    U_PM_MP = la.expm(-1j * J/2 * (PM * np.exp(1.0j * phi) + \
                                   MP * np.exp(-1.0j * phi)))
    U_ZZ = II * np.cos(Jz/4) - 1j * ZZ * np.sin(Jz/4) 
    U_XXZ = U_PM_MP @ U_ZZ
    return U_H1 @ U_XXZ

def gen_MagMask(masks_dict):
    '''generate mask for magnetization calculation'''
    N = len(masks_dict)
    magn_mask_is = np.zeros((N, 2**N))
    for i in range(N):
        masks = masks_dict[N - i - 1]
        qubit_up_mask = masks[0].tolist() + masks[1].tolist()
        magn_mask_i = np.zeros(2**N)
        magn_mask_i[qubit_up_mask] = 1
        magn_mask_is[i] = 2*magn_mask_i-1
    return magn_mask_is

@njit(parallel=True, fastmath=True, cache=True)
def compute_magn(psi_2, magn_mask_is):
    N = int(np.log2(len(psi_2)))
    magn_is = np.zeros(N)
    for i in prange(N):
        magn_is[i] = np.dot(psi_2, magn_mask_is[i]).real
    return magn_is

def compute_single_trajectory(gates, order, psi_0, T, masks_dict, magn_mask_is):
    spin_densities_list = []
    psi_2 = psi_0.conj() * psi_0    
    magnetization = compute_magn(psi_2, magn_mask_is)
    spin_densities_list.append(magnetization)
    N = len(order)

    for t in range(T):
        if t < N:

            for order_idx, gate_idx in enumerate(order):
                psi_0 = apply_gate(psi_0, gates[order_idx], masks_dict[gate_idx])
                psi_2 = psi_0.conj() * psi_0 
                magnetization = compute_magn(psi_2, magn_mask_is)   
                spin_densities_list.append(magnetization)
        else:
            psi_0 = apply_U(psi_0, gates, order, masks_dict)
            psi_2 = psi_0.conj() * psi_0    
            magnetization = compute_magn(psi_2, magn_mask_is)
            spin_densities_list.append(magnetization)

    return np.array(spin_densities_list)

def gen_t_list(N, T):
    '''generate time list
    First N steps computed gate by gate
    Then stroboscopic steps computed at each whole time step up to T
    '''
    additional_steps = (np.arange(N).reshape(N, 1) + 
                    (np.arange(1, N + 1) / N)
                    ).flatten()
    stroboscobic = np.arange(N,T)
    return np.concatenate(([0], additional_steps, stroboscobic))

def gen_initial_state(masks_dict, rnd_seed=None):
    '''generate initial state
    |0>^{\otimes N} with the first qubit up

    verify it with 
        psi_2 = psi_0.conj() * psi_0    
        fn.compute_magn(psi_2, magn_mask_is)
    '''
    N = len(masks_dict)
    if rnd_seed is not None:
        nprd.seed(rnd_seed)
    psi_0 = nprd.uniform(0,1,2**N) + 1.0j*nprd.uniform(0,1,2**N)
    qubit_0_down_mask = masks_dict[0][2].tolist() + masks_dict[0][3].tolist()
    psi_0[qubit_0_down_mask] = 0
    return psi_0 / np.sqrt(np.dot(psi_0.conj(), psi_0))

def apply_single_z(psi_0, masks_dict):
    psi = psi_0.copy()
    N = len(masks_dict)
    for i in range(0, N, 2):
        h1, h2 = nprd.uniform(-np.pi, np.pi, 2)
        ZZ_random = np.diag(np.exp(-.5j*np.array([h1+h2, h1-h2, h2-h1, -h1-h2])))
        psi = apply_gate(psi, ZZ_random, masks_dict[i])
    return psi

# def compute_single_trajectory_with_trick(params, order, psi_0, T, 
#                                          disorder_realizations, masks_dict, magn_mask_is):
#     N = len(order)
#     numb_steps = (T-N) + N**2 + 1
#     spin_evol = np.zeros((disorder_realizations, numb_steps, N))

#     for realization in range(disorder_realizations):
#         psi_r = apply_single_z(psi_0, masks_dict)
#         psi_2 = psi_r.conj() * psi_r    
#         magnetization = compute_magn(psi_2, magn_mask_is)
#         spin_evol[realization, 0, :] = magnetization

#     J, Jz = params

#     for t in tqdm(range(T)):
#         if t < N:
#             phis = nprd.uniform(-np.pi, np.pi, N)
#             hs = nprd.uniform(-np.pi, np.pi, 2*N)
#             gates = np.array([gate_xxz_disordered(J, Jz, hs[2*i], 
#                                                   hs[2*1+1], phis[i]) for i in range(N)])
            
#             for order_idx, gate_idx in enumerate(order):
#                 psi_0 = apply_gate(psi_0, gates[order_idx], masks_dict[gate_idx])

#                 for realization in range(disorder_realizations):
#                     psi_r = apply_single_z(psi_0, masks_dict)
#                     psi_2 = psi_r.conj() * psi_r    
#                     magnetization = compute_magn(psi_2, magn_mask_is)
#                     spin_evol[realization, t*N + order_idx, :] = magnetization
#         else:
#             for realization in range(disorder_realizations):
#                 psi_r = apply_single_z(psi_0, masks_dict)
#                 psi_2 = psi_r.conj() * psi_r    
#                 magnetization = compute_magn(psi_2, magn_mask_is)
#                 spin_evol[realization, (t-N) + N**2, :] = magnetization

#     return spin_evol

def gen_PS():
    '''Generate a random product state on one qubit'''
    state = np.random.rand(2) + 1j * np.random.rand(2)
    return state / np.linalg.norm(state)

def initial_mixed_state(N, theta, state_phases, p):
    '''Generate a random product state on N qubits'''
    # Generate id + p Product State
    paulis = []
    for n in range(N):
        if np.random.rand() < p:
            if state_phases == 'homogenous':
                phase = 1
            elif state_phases == 'staggered':
                phase = (-1)**n
            state = r_y(phase*theta) @ np.array([1, 0])
            paulis.append(state)
        else:
            if np.random.rand() < .5:
                paulis.append(UP)
            else:
                paulis.append(DOWN)
    return reduce(np.kron, paulis)

def is_bad_value(coeff):
    return coeff == 0 or np.isnan(coeff) or np.isinf(coeff) or not np.isfinite(coeff)

def generic_coherence_measure(rho, eigvals, eigvecs, Ls, f):
    '''From eq.1.44 of
    Irénée Frerot. A quantum statistical approach to quantum correlations in many-body systems. 
    Statistical Mechanics [cond-mat.stat-mech]. Université de Lyon, 2017. 
    
    f is a standard monotone function.
    the mixed state is assumed to be 
                ρ = p |ψ⟩⟨ψ| + (1 - p) 𝕀/d
    where |psi> is the tilted state and Id/d is the maximally mixed state:
            |ψ(θ)⟩ = e^(-iθ/2 ∑ₖ σₖʸ) |000...0⟩
            
    ρ_A = Tr_{B} ρ
    '''
    Coherent = np.zeros(len(Ls))
    # Op = np.zeros(len(Ls))
    
    # if f == f_SLD:
    #     sigma = L @ rho @ L
    #     rho_1 = la.inv(rho)
    #     Op = np.trace(rho @  L @ rho_1 @ L @ rho - rho @ sigma @ rho_1 - rho + L @ rho @ rho @ L @ rho_1)
    
    # if f == f_WY:
    #     Op = WY(rho, Ls)
        
    # if f == f_rel_ent:
    #     sigma = L @ rho @ L
    #     log_rho = la.logm(rho) # stable_logm(rho)
    #     log_sigma = la.logm(sigma)
    #     Op = np.trace(sigma @ log_sigma) - np.trace(sigma @ log_rho)
    
    # if f == f_q_info_var:
    # ### TO BE COMPLETED
    
    # if f == f_geo_mean:
    # ### TO BE COMPLETED
    
    # if f == f_harm_mean:
    #     rho_1 = la.inv(rho)
    #     Op = (np.trace(L @ rho @ rho @ L @ rho_1) - np.trace(L @ rho @ L))/2
        
    eigvals[eigvals < 1e-12] = 0
        
    indexes = np.argsort(eigvals)
    eigvals = eigvals[indexes]
    eigvecs = eigvecs.T[indexes]
            
    for i, eigval_i in enumerate(eigvals): # Apply L on the state
        Ls_eigvec_i = [L @ eigvecs[i] for L in Ls]
        
        for j, eigval_j in enumerate(eigvals):
            if i==j: continue
            try:
                coeff = ((eigval_i - eigval_j)**2/(eigval_i*f((eigval_j/(eigval_i))))) 
            except ZeroDivisionError:
                pass
            if is_bad_value(coeff): 
                try:
                    coeff = ((eigval_j - eigval_i)**2/(eigval_j*f((eigval_i/(eigval_j))))) 
                except ZeroDivisionError:
                    pass
                if is_bad_value(coeff): 
                    continue
        
            Coherent += [coeff * np.abs(eigvecs[j].conj().dot(Li))**2 for Li in Ls_eigvec_i]
                
    if not (f(0) == 0 or np.isnan(f(0))): Coherent *= f(0)/2 # Removed for the QFIs with f(0) = 0
    else: Coherent *= 1/2
            
    return Coherent

def compute_incoherent_fisher_info(p_t, p_t_minus_dt, dt):
    epsilon = 1e-12  # Small constant to avoid division by zero
    # Step 1: Compute dp/dt using backward difference
    dp_dt = (p_t - p_t_minus_dt) / dt
    # Step 2: Compute d/dt(log(p)) = dp/dt / p
    log_p_derivative = dp_dt / (p_t + epsilon)
    # Step 3: Compute the weighted sum for F_Q^IC
    F_Q_IC = np.sum(p_t * log_p_derivative**2)
    return F_Q_IC


def f_SLD(x): # Bures metric: f(x) = (x + 1) / 2
    return (x + 1) / 2 # y * f(x/y) = y * (x/y + 1) / 2 = (x + y) / 2

def f_Heinz(x, r): # Heinz family:: f(x) = (x^r + x^(1-r)) / 2
    return (x**r + x**(1 - r)) / 2

def f_ALPHA(x, alpha): # Alpha-divergency: f(x) = α(α - 1)(x - 1)^2 / ((x - x^α)(x^α - 1))
    numerator = alpha * (alpha - 1) * (x - 1)**2
    denominator = (x - x**alpha) * (x**alpha - 1)
    return numerator / denominator

def f_WY(x): # Wigner-Yanase metric: f(x) = (1/4) * (1 + sqrt(x))^2
    return (1 / 4) * (1 + np.sqrt(x))**2 # (pi-pj)/((1/4) * (pi+pj)^2/pi) = 4pi(pi-pj)/(pi+pj)^2 != sqrt(pi*pj)

def f_rel_ent(x): # Relative entropy: f(x) = (x - 1) / log(x)
    return (x - 1) / np.log(x)

def f_q_info_var(x): # Quantum information variance: f(x) = (2 * (x - 1)^2) / ((x + 1) * (log(x))^2)
    numerator = 2 * (x - 1)**2
    denominator = (x + 1) * (np.log(x)**2)
    return numerator / denominator

def f_geo_mean(x): # Geometric mean: f(x) = sqrt(x)
    return np.sqrt(x)

def f_harm_mean(x): # Harmonic mean: f(x) = (2 * x) / (x + 1)
    return (2 * x) / (x + 1)


def find_crossing_times(x_vals, y_vals1, y_vals2):
    """
    Find the crossing times between y_vals1 and y_vals2 by interpolation.
    
    Parameters:
    x_vals (np.ndarray): Array of x values.
    y_vals1 (np.ndarray): Array of y values for the first function.
    y_vals2 (np.ndarray): Array of y values for the second function.
    
    Returns:
    np.ndarray: Array of x values where the two functions cross.
    """
    # Compute the difference between the y values
    diff = y_vals1 - y_vals2
    
    # Find the indices where the sign of the difference changes
    crossing_indices = np.where(np.diff(np.sign(diff)))[0]
    
    # Interpolate to find the exact crossing points
    crossing_times = []
    if len(crossing_indices) == 0:
        return np.array(crossing_times)
    for idx in crossing_indices:
        x1, x2 = x_vals[idx], x_vals[idx + 1]
        y1, y2 = diff[idx], diff[idx + 1]
        crossing_time = x1 - y1 * (x2 - x1) / (y2 - y1)
        crossing_times.append(crossing_time)
    
    return np.array(crossing_times)

def compute_unitary(gates, order, masks_dict, N):
    '''
    Compute the unitary of the circuit
    '''
    # since the gates are applied on states apply the gates to the identity matrix, column by column sending them to the apply_U function
    U = np.eye(2**N, dtype=complex)
    for i in tqdm(range(2**N)):
        U[:, i] = apply_U(U[:, i], gates, order, masks_dict)
    return U

def compute_hamiltonian(gates, order, masks_dict, N):
    '''
    Compute the Hamiltonian of the circuit
    '''
    U = compute_unitary(gates, order, masks_dict, N)
    # diagonalize U
    eigvals, eigvecs = np.linalg.eig(U)
    # compute the log of the eigenvalues
    log_eigvals = np.log(eigvals)
    return eigvecs.T.conj() @ log_eigvals @ eigvecs


@njit(parallel=True, fastmath=True, cache=True)
def compute_projector(Ns, states):
    """
    Computes the projector onto the subspace spanned by the computational basis 
    states in the list 'states'. Each state is assumed to be an integer corresponding 
    to the basis index.
    """
    dim = 2**Ns
    P = np.zeros((dim, dim), dtype=np.complex128)
    for state_1 in states:
        v_1 = np.zeros(dim, dtype=np.complex128)
        v_1[state_1] = 1.0
        for state_2 in states:
            v_2 = np.zeros(dim, dtype=np.complex128)
            v_2[state_2] = 1.0
            # Add the projector for this state
            P += np.outer(v_1, v_2)
    # normalize
    P /= np.linalg.norm(P)
    return P

@njit(parallel=True, fastmath=True, cache=True)
def compute_projector(Ns, states):
    """
    Computes the projector onto span{|s⟩ : s in states}.
    states should be a 1D np.int64 array of basis indices.
    """
    dim = 1 << Ns           # 2**Ns
    P   = np.zeros((dim, dim), dtype=np.complex128)
    n   = states.shape[0]   # number of basis states

    # Fill P[s1,s2] = 1 for all s1,s2 in states
    for i in prange(n):
        s1 = states[i]
        for j in prange(n):
            s2 = states[j]
            P[s1, s2] = 1.0

    # The Frobenius norm of this matrix is n, so normalize by n
    return P / n



# blocks:

def get_block_sizes(N_A):
    """Compute the sizes of the diagonal blocks using binomial coefficients."""
    from quspin.basis import spin_basis_1d
    m_values = np.linspace(-.5, .5, N_A+1)  # Adjust range as needed
    return [int(spin_basis_1d(N_A, m=m).Ns) for m in m_values]

def split_block_diagonal(matrix, N_A):
    """Split a block-diagonal matrix into its individual blocks."""
    block_sizes = get_block_sizes(N_A)
    indices = np.cumsum([0] + block_sizes)  # Compute slicing indices

    blocks = []
    for i in range(len(block_sizes)):
        start, end = indices[i], indices[i+1]
        blocks.append(matrix[start:end, start:end])

    return blocks

def merge_block_diagonal(blocks):
    """Merge individual blocks into a single block-diagonal matrix."""
    total_size = sum(block.shape[0] for block in blocks)
    merged_matrix = np.zeros((total_size, total_size), dtype=np.complex128)

    start = 0
    for block in blocks:
        size = block.shape[0]
        merged_matrix[start:start+size, start:start+size] = block
        start += size

    return merged_matrix

def operation_per_block(rho, function, N_A):
    '''
    Compute function on each block of rho individually
    '''
    blocks = split_block_diagonal(rho, N_A)
    for idx, block in enumerate(blocks):
        # fn.print_matrix(block, 4)
        blocks[idx] = function(block)
    return merge_block_diagonal(blocks)

def manual_U1_tw(rho, projectors):
    '''
    Apply the twirling operation to the density matrix rho.
    The twirling operation is a sum over the projectors, weighted by the density matrix.
    If ordered is True, the projectors are applied on the reordered basis.
    '''
    P = np.array([Pj / np.max(Pj) for Pj in projectors.values()])  # Shape (N, d, d)
    
    return np.sum(P * rho, axis=0)


''' Checking the consistency of S(rho || G(rho)) == S(G(rho)) - S(rho) 
<\

import scipy.special  # For binomial coefficient

state = fn.initial_state(N, sites_to_keep, .2 * np.pi, state_phases)
state /= np.linalg.norm(state)
# apply a U
h_list = np.random.uniform(-np.pi, np.pi, 5*N).reshape(N, 5) /alpha
gates = [fn.gen_u1([*h]) for h in h_list]
order = fn.gen_gates_order(N, geometry=geometry)
state = fn.apply_U(state, gates, order, masks_dict)
pstate = fn.ptrace(state, sites_to_keep)
##############################################################################
pstateQ = fn.twirling(pstate, projectors)    
    
reordered_pstate = basis_reordering.T @ pstate @ basis_reordering
reordered_pstateQ = basis_reordering.T @ pstateQ @ basis_reordering
    
from scipy.linalg import logm, expm

def vNentropy(x): return - x @ logm(x)
def idfunction(x): return x 

A = - basis_reordering.T @ pstateQ @ logm(pstateQ) @ basis_reordering
B = fn.operation_per_block(reordered_pstate, vNentropy, N_A)
C = fn.twirling(- basis_reordering.T @ pstate @ logm(pstateQ) @ basis_reordering, reordered_projectors)
C = (- basis_reordering.T @ pstate @ logm(pstateQ) @ basis_reordering)
C = (- pstate @ logm(pstateQ))

fn.print_matrix(A, 2)
fn.print_matrix(B, 2)
fn.print_matrix(C, 2)
np.trace(A), np.trace(B), np.trace(C)

>
'''


##### Relative entropy

import numpy as np
from scipy.linalg import eigh, fractional_matrix_power, logm
import warnings
warnings.simplefilter("ignore", category=UserWarning)

def _safe_logm(mat: np.ndarray, epsilon: float) -> np.ndarray:
    """
    Compute log(mat) by eigen‑decomposition, clamping eigenvalues to [epsilon, ∞).
    """
    vals, vecs = eigh(mat)
    # clamp eigenvalues away from zero
    safe_vals = np.clip(vals, epsilon, None)
    log_vals  = np.log(safe_vals)
    return (vecs * log_vals) @ vecs.conj().T

def _safe_frac_power(mat: np.ndarray, power: float, epsilon: float) -> np.ndarray:
    """
    Compute mat**power by eigen‑decomposition, clamping eigenvalues to [epsilon, ∞).
    """
    vals, vecs = eigh(mat)
    safe_vals = np.clip(vals, epsilon, None)
    frac_vals = safe_vals**power
    return (vecs * frac_vals) @ vecs.conj().T

def renyi_divergence(
    rho: np.ndarray,
    sigma: np.ndarray,
    alpha: float = 1.0,
    epsilon: float = 1e-12,
) -> float:
    """
    Computes D_α(ρ || σ) with spectrum‑level regularization to avoid Infs/NaNs.

    Parameters
    ----------
    rho, sigma : np.ndarray
        Density matrices (Hermitian, trace 1).
    alpha : float
        Rényi parameter (α > 0, α ≠ 1 normally; α→1 gives KLD).
    epsilon : float
        Clamping floor for all eigenvalues.

    Returns
    -------
    float
        The Rényi divergence D_α(ρ || σ).
    """
    # basic checks
    if alpha <= 0:
        raise ValueError("α must be > 0.")
    t_rho = np.trace(rho)
    t_sig = np.trace(sigma)
    if not np.allclose(t_rho, 1, atol=1e-6) or not np.allclose(t_sig, 1, atol=1e-6):
        raise ValueError("Both ρ and σ must have trace 1: "
                         f"tr(ρ) = {t_rho}, tr(σ) = {t_sig}.")

    # α → 1 → Kullback-Leibler
    if np.isclose(alpha, 1.0):
        log_rho   = _safe_logm(rho,   epsilon)
        log_sigma = _safe_logm(sigma, epsilon)
        # D = tr[ρ (log ρ − log σ)]
        D = np.real_if_close(np.trace(rho @ (log_rho - log_sigma)))
        return float(D)

    # α → 0 limit
    if np.isclose(alpha, 0.0):
        # support projector of ρ
        vals_rho, _ = eigh(rho)
        support = (vals_rho > epsilon).astype(float)
        vals_sig, _ = eigh(sigma)
        return -np.log(np.sum(support * np.clip(vals_sig, epsilon, None)))

    # enforce α ≤ 2 if desired
    if alpha > 2:
        raise ValueError("α must be ≤ 2 for this implementation.")

    # general α ≠ 1
    rho_a = _safe_frac_power(rho,   alpha,     epsilon)
    sig_b = _safe_frac_power(sigma, 1 - alpha, epsilon)
    trace_term = np.trace(rho_a @ sig_b)

    # guard against tiny negatives or Infs
    trace_term = np.real_if_close(trace_term)
    trace_term = float(np.clip(trace_term, epsilon, None))

    return (1.0 / (alpha - 1.0)) * np.log(trace_term)

def renyi_divergence_sym(
    rho: np.ndarray,
    symmetry: str,
    alpha: float = 1.0,
    epsilon: float = 1e-12,
    K = None,
    Ubasis=None
) -> float:
    """
    Computes D_α(ρ || G(ρ)) with spectrum‑level regularization to avoid Infs/NaNs.

    Parameters
    ----------
    rho : np.ndarray
        Density matrix (Hermitian, trace 1).
    symmetry : str
        Symmetry group for the twirling operation (e.g., 'U1', 'SU2').
    alpha : float
        Rényi parameter (α > 0, α ≠ 1 normally; α→1 gives KLD).
    epsilon : float
        Clamping floor for all eigenvalues.

    Returns
    -------
    float
        The Rényi divergence D_α(ρ || σ).
    """
    # basic checks
    if alpha <= 0:
        raise ValueError("α must be > 0.")
    if symmetry not in ['U1', 'SU2', 'Z2', 'ZK']:
        raise ValueError("symmetry must be 'U1' or 'SU2' or 'Z2' or 'ZK'.")
    Ns = int(np.log2(rho.shape[0]))
    if Ubasis is not None:
        U_basis = Ubasis
    else:
        if symmetry == 'U1':
            projectors, U_basis = build_projectors(Ns)
        elif symmetry == 'SU2':
            U_basis = {2:U_CG_2, 3:U_CG_3, 4:U_CG_4, 6:U_CG_6, 8:U_CG_8}[Ns]
        elif symmetry == 'Z2':
            U_basis = U_Z2_gen(Ns)
        elif symmetry == 'ZK':
            U_basis = U_ZK_gen(Ns, K).conj().T
        else:
            raise ValueError("Invalid symmetry group. Choose 'U1', 'SU2', 'Z2' or 'ZK'.")
        
    if symmetry == 'U1':           
        sigma = manual_U1_tw(rho, projectors)
    elif symmetry == 'SU2':
        sigma = manual_N4_SU2_tw(rho)
    elif symmetry == 'Z2':
        sigma = manual_Z2_tw(rho)
    elif symmetry == 'ZK':
        sigma = manual_ZK_tw(rho, K)
        
    rho = U_basis.conj().T @ rho @ U_basis
    sigma = U_basis.conj().T @ sigma @ U_basis
    t_rho = np.trace(rho)
    t_sig = np.trace(sigma)
    
    if not np.allclose(t_rho, 1, atol=1e-6) or not np.allclose(t_sig, 1, atol=1e-6):
        raise ValueError("Both ρ and σ must have trace 1: "
                         f"tr(ρ) = {t_rho}, tr(σ) = {t_sig}.")

    # α → 1 → Kullback-Leibler
    if np.isclose(alpha, 1.0):
        log_rho   = _safe_logm(rho,   epsilon)
        log_sigma = _safe_logm(sigma, epsilon)
        log_sigma_basis = U_basis @ log_sigma @ U_basis.conj().T
        if symmetry == 'U1':           
            log_sigma_basis_tw = manual_U1_tw(log_sigma_basis, projectors)
        elif symmetry == 'SU2':
            log_sigma_basis_tw = manual_N4_SU2_tw(log_sigma_basis)
        elif symmetry == 'Z2':
            log_sigma_basis_tw = manual_Z2_tw(log_sigma_basis)
        elif symmetry == 'ZK':
            log_sigma_basis_tw = manual_ZK_tw(log_sigma_basis, K)
        log_sigma = U_basis.conj().T @ log_sigma_basis_tw @ U_basis
        # D = tr[ρ (log ρ − log σ)]
        D = np.real_if_close(np.trace(rho @ (log_rho - log_sigma)))
        return float(D)

    # α → 0 limit
    if np.isclose(alpha, 0.0):
        # support projector of ρ
        vals_rho, _ = eigh(rho)
        support = (vals_rho > epsilon).astype(float)
        vals_sig, _ = eigh(sigma)
        return -np.log(np.sum(support * np.clip(vals_sig, epsilon, None)))

    # enforce α ≤ 2 if desired
    if alpha > 2:
        raise ValueError("α must be ≤ 2 for this implementation.")

    # general α ≠ 1
    rho_a = _safe_frac_power(rho,   alpha,     epsilon)
    sig_b = _safe_frac_power(sigma, 1 - alpha, epsilon)
    trace_term = np.trace(rho_a @ sig_b)

    # guard against tiny negatives or Infs
    trace_term = np.real_if_close(trace_term)
    trace_term = float(np.clip(trace_term, epsilon, None))

    return (1.0 / (alpha - 1.0)) * np.log(trace_term)


def max_divergence(rho: np.ndarray, sigma: np.ndarray, epsilon=1e-10) -> float:
    """
    Computes the max divergence D_infty(rho || sigma).
    """
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.maximum(eigvals, epsilon)  # Regularization

    sqrt_sigma_inv = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
    sandwiched_matrix = sqrt_sigma_inv @ rho @ sqrt_sigma_inv
    lambda_max = np.max(np.linalg.eigvalsh(sandwiched_matrix))
    
    return np.log(lambda_max)



def decompose_rho_modes(rho, Ub, Op):
    """
    Decomposes the density matrix 'rho' into its frequency (charge difference) modes.
    
    Parameters:
    -----------
    rho : ndarray
        The density matrix in the original (computational) basis.
    Ub : ndarray
        The unitary transformation matrix that rotates the computational basis into the 
        eigenbasis of J_z or J^2 (sorted from the lowest to the highest eigenvalue).
    Op : ndarray
        The magnetization operator or the total angular momentum operator J^2.
        
    Returns:
    --------
    freq_modes : dict
        A dictionary where keys are integer frequency modes (charge differences) and 
        the values are matrices of the same shape as the rotated density matrix
        containing the part of rho associated with that frequency.
    """
    # Rotate the density matrix into the J^2-eigenbasis.
    # In this basis, U_b.T @ J^2 @ U_b is diagonal.
    rho_rot = Ub.conj().T @ rho @ Ub
    
    # The diagonal of the rotated J^2 is the sorted charge vector.
    # (We assume here that this product is exactly diagonal; in numerical code, you might
    # enforce a tolerance.)
    diag_Op = Ub.conj().T @ Op @ Ub
    eigs_Op = np.real(np.diag(diag_Op))
    
    # Initialize dictionary to store matrices for each frequency mode.
    freq_modes = {}
    d = rho_rot.shape[0]
    
    # Loop over each element of rho_rot and assign it to the appropriate frequency.
    # The frequency is defined as the difference between the different values of the 
    # eigval of the Operator of the row and column.
    for i in range(d):
        for j in range(d):
            # frequency mode associated with element (i,j)
            freq = int(eigs_Op[i] - eigs_Op[j])
            # Initialize the mode if it does not exist; same shape as rho_rot.
            if freq not in freq_modes:
                freq_modes[freq] = np.zeros_like(rho_rot, dtype=rho_rot.dtype)
            freq_modes[freq][i, j] = rho_rot[i, j]
            
    return freq_modes

'''
# Test cases
rho_pure = np.array([[1, 0], [0, 0]])  # Pure state
rho_mixed = np.array([[0.5, 0], [0, 0.5]])  # Maximally mixed state
rho_intermediate = np.array([[0.7, 0.3], [0.3, 0.3]])  # Intermediate case

sigma_mixed = np.array([[0.6, 0.4], [0.4, 0.4]])  # Reference state

# Test different values of alpha
alpha_values = [0.0001, .9999, 100]  # Alpha → 0, 1 (KL), and large alpha
test_pairs = {
    "Pure vs Mixed": (rho_pure, sigma_mixed),
    "Mixed vs Mixed": (rho_mixed, sigma_mixed),
    "Intermediate vs Mixed": (rho_intermediate, sigma_mixed)
}

for name, (rho, sigma) in test_pairs.items():
    print(f"\n### {name} ###")
    print(f"  Max Divergence = {max_divergence(rho, sigma)}")
    print(f"  KL-1 Divergence = {renyi_divergence(rho, sigma, 1)}")
    for alpha in alpha_values:
        renyi_val = renyi_divergence(rho, sigma, alpha)
        print(f"  Rényi Divergence (α={alpha}): {renyi_val}")

    print(f"  Large α Approx: {renyi_divergence(rho, sigma, 100)}")
    
'''

# U1 ############
from quspin.basis import spin_basis_1d
# Create a dictionary to hold projectors for each magnetization subsector.
# Here we assume the magnetization m runs from -NA/2 to NA/2 in steps of 1.

def build_projectors(N_A):
    projectors = {}
    U_U1 = np.zeros((2**N_A, 2**N_A), dtype=np.complex128)
    row_index = 0
    old_basis = None
    for m in np.linspace(-.5, .5, N_A+1):
        # Retrieve the list of computational basis states for this magnetization sector.
        # (Assuming spin_basis_1d(NA, m=m) returns an object with a member .states.)
        if old_basis is not None:
            it = 0
            while old_basis.Ns == spin_basis_1d(N_A, m=m).Ns:
                m += 1e-7
                it += 1
                if it > 1000:
                    print("Warning: too many iterations")
                    break
        
        basis_obj = spin_basis_1d(N_A, m=m)
        old_basis = basis_obj
                
        states_m = basis_obj.states
        # print(f"Magnetization m {m:.2f} has", len(states_m), "states: states_m =", states_m)
        for state in states_m[::-1]:
            U_U1[state, row_index] = 1
            row_index += 1
        # Compute the projector onto the subspace spanned by these states.
        projectors[m] = compute_projector(N_A, states_m)
        
    return projectors, U_U1


# projectors2, U_U1_2 = build_projectors(2)
# projectors3, U_U1_3 = build_projectors(3)
# projectors4, U_U1_4 = build_projectors(4)
# projectors6, U_U1_6 = build_projectors(6)
# projectors8, U_U1_8 = build_projectors(8)

# reordered projectors:

# reordered_projectors8 = {}
# for m in projectors8.keys():
#     reordered_projectors8[m] = U_U1_8.T @ projectors8[m] @ U_U1_8

'''
# Test cases
rho_pure = np.array([[1, 0], [0, 0]])  # Pure state
rho_mixed = np.array([[0.5, 0], [0, 0.5]])  # Maximally mixed state
rho_intermediate = np.array([[0.7, 0.3], [0.3, 0.3]])  # Intermediate case

sigma_mixed = np.array([[0.6, 0.4], [0.4, 0.4]])  # Reference state

# Test different values of alpha
alpha_values = [0.0001, .9999, 100]  # Alpha → 0, 1 (KL), and large alpha
test_pairs = {
    "Pure vs Mixed": (rho_pure, sigma_mixed),
    "Mixed vs Mixed": (rho_mixed, sigma_mixed),
    "Intermediate vs Mixed": (rho_intermediate, sigma_mixed)
}

for name, (rho, sigma) in test_pairs.items():
    print(f"\n### {name} ###")
    print(f"  Max Divergence = {max_divergence(rho, sigma)}")
    print(f"  KL-1 Divergence = {renyi_divergence(rho, sigma, 1)}")
    for alpha in alpha_values:
        renyi_val = renyi_divergence(rho, sigma, alpha)
        print(f"  Rényi Divergence (α={alpha}): {renyi_val}")

    print(f"  Large α Approx: {renyi_divergence(rho, sigma, 100)}")
    
'''


#### SU(2) symmetric projections:

def operator_on_site(op, site, N):
    """
    Constructs an operator that acts as 'op' on the 'site'-th spin
    (0-indexed) and as the identity on all other spins.
    
    Parameters:
        op (np.ndarray): A 2x2 operator (e.g., a Pauli matrix or a spin operator)
        site (int): The site (0-indexed) on which to act with 'op'
        N (int): Total number of spins
        
    Returns:
        np.ndarray: The operator in the full Hilbert space of dimension 2^N.
    """
    # Identity operator for a single spin (2x2)
    I = np.eye(2, dtype=complex)
    
    # Build the full operator using Kronecker (tensor) products
    full_op = 1
    for i in range(N):
        if i == site:
            full_op = np.kron(full_op, op)
        else:
            full_op = np.kron(full_op, I)
    return full_op

def build_J2(N):
    """
    Constructs the total angular momentum squared operator J^2 for N spin-1/2 particles.
    
    For each spin-1/2, the spin operators are defined as:
      S_x = 1/2 * sigma_x,
      S_y = 1/2 * sigma_y,
      S_z = 1/2 * sigma_z.
    
    The total operators are
      J_alpha = sum_{i=0}^{N-1} S_alpha^{(i)}   for alpha in {x, y, z},
    and then
      J^2 = J_x^2 + J_y^2 + J_z^2.
    
    Parameters:
        N (int): The number of spin-1/2 particles.
        
    Returns:
        np.ndarray: The operator J^2 as a 2^N x 2^N matrix.
    """
    # Define the Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Spin operators for a single spin (S = 1/2)
    Sx = 0.5 * sigma_x
    Sy = 0.5 * sigma_y
    Sz = 0.5 * sigma_z
    
    # Initialize total operators for the full Hilbert space.
    # They are matrices of dimension 2^N x 2^N.
    # Start with zero matrices.
    dim = 2 ** N
    Jx = np.zeros((dim, dim), dtype=complex)
    Jy = np.zeros((dim, dim), dtype=complex)
    Jz = np.zeros((dim, dim), dtype=complex)
    
    # Sum over each spin site.
    for i in range(N):
        Jx += operator_on_site(Sx, i, N)
        Jy += operator_on_site(Sy, i, N)
        Jz += operator_on_site(Sz, i, N)
    
    # Compute J^2 = Jx^2 + Jy^2 + Jz^2
    J2 = np.dot(Jx, Jx) + np.dot(Jy, Jy) + np.dot(Jz, Jz)
    return J2

def build_Jz(N):
    """
    Constructs the total angular momentum z-component operator Jz for N spin-1/2 particles.
    
    For each spin-1/2, the spin operator is defined as:
      S_z = 1/2 * sigma_z.
    
    The total operator is
      J_z = sum_{i=0}^{N-1} S_z^{(i)}.
    
    Parameters:
        N (int): The number of spin-1/2 particles.
        
    Returns:
        np.ndarray: The operator Jz as a 2^N x 2^N matrix.
    """
    # Define the Pauli matrix for z-component
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    
    # Spin operator for a single spin (S = 1/2)
    Sz = 0.5 * sigma_z
    
    # Initialize the Jz operator
    dim = 2 ** N
    Jz = np.zeros((dim, dim), dtype=complex)
    
    # Sum over each spin site.
    for i in range(N):
        Jz += operator_on_site(Sz, i, N)
    
    return Jz

def build_Jminus(N):
    """
    Constructs the total angular momentum lowering operator J_- for N spin-1/2 particles.
    
    For each spin-1/2, the lowering operator is defined as:
        S_- = 1/2 * (sigma_x - i sigma_y)
    
    The total lowering operator is given by:
        J_- = sum_{i=0}^{N-1} S_-^{(i)},
    where S_-^{(i)} acts on the ith spin.
    
    Parameters:
        N (int): The number of spin-1/2 particles.
        
    Returns:
        np.ndarray: The operator J_- as a 2^N x 2^N matrix.
    """
    # Define the Pauli matrices for x and y components.
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    
    # The spin lowering operator for a single spin (S = 1/2)
    S_minus = 0.5 * (sigma_x + 1j * sigma_y)
    
    # Initialize the full J_minus operator.
    dim = 2 ** N
    J_minus = np.zeros((dim, dim), dtype=complex)
    
    # Sum the lowering operators over all spin sites.
    for i in range(N):
        J_minus += operator_on_site(S_minus, i, N)
    
    return J_minus

##### N = 2

# Helper: return the computational basis ket corresponding to a bit-string label.
# Here the leftmost bit is qubit 1 and the rightmost is qubit 4.
def ket(label):
    # label is a 4-character string, e.g. "1011"
    index = int(label, 2)
    v = np.zeros(4, dtype=complex)
    v[index] = 1.0
    return v

# Build a state as a linear combination of computational basis kets,
# given by a dictionary mapping bit-string labels to coefficients.
def add_state(coeff_dict):
    v = np.zeros(4, dtype=complex)
    for state, coeff in coeff_dict.items():
        v += coeff * ket(state)
    return v


basis_states = []  # list to hold our new basis vectors in the desired order
states_N2 = {}

# ====================================================
# I. J=1 (Triplet) states, total 3 states

# |1, 1> = |11>
state_1_1 = ket("11")
states_N2["1,1"] = state_1_1
basis_states.append(state_1_1)

# |1, 0> = (1/√2)[|10> + |01>]
state_1_0 = add_state({
    "01": 1/np.sqrt(2), "10": 1/np.sqrt(2)
})
states_N2["1,0"] = state_1_0
basis_states.append(state_1_0)

# |1, -1> = |00>
state_1_m1 = ket("00")
states_N2["1,-1"] = state_1_m1
basis_states.append(state_1_m1)

# ====================================================
# II. J=0 (singlet) state.
# |0, 0> = (1/√2)[|10> - |01>]
state_0_0 = add_state({
    "01": 1/np.sqrt(2), "10": -1/np.sqrt(2)
})
states_N2["0,0"] = state_0_0
basis_states.append(state_0_0)

# ====================================================
# Now stack the 4 basis state vectors as columns in a matrix.
U_CG_2 = np.column_stack(basis_states[::-1])

def manual_N2_SU2_tw(rho):
    """
    Manual twirling for N=2.
    """
    rho_CG = U_CG_2.conj().T @ rho @ U_CG_2
    B_00 = rho_CG[0,0]
    B_11 = rho_CG[1:,1:]
    
    Manual_tw = np.zeros_like(rho, dtype=np.complex128)
    J = 0
    Manual_tw[:1,:1] = B_00 * np.eye(2*J+1, dtype=np.complex128)/(2*J+1)
    J = 1
    Manual_tw[1:,1:] = np.trace(B_11) * np.eye(2*J+1, dtype=np.complex128)/(2*J+1)
    return U_CG_2 @ Manual_tw @ U_CG_2.conj().T

##### N = 3

# Helper function to create a computational basis ket for N=3.
# The bit string is given as a 3-character string: e.g., "101" means qubit1 = 1, qubit2 = 0, qubit3 = 1.
def ket(label):
    index = int(label, 2)  # interpret label as binary to get the index in [0,7]
    vec = np.zeros(8, dtype=complex)
    vec[index] = 1.0
    return vec

# Helper: Build a state from a dictionary of bit-string components and coefficients.
def build_state(coeff_dict):
    state = np.zeros(8, dtype=complex)
    for basis_state, coeff in coeff_dict.items():
        state += coeff * ket(basis_state)
    return state

# List to store the 8 basis states (columns of U)
basis_states = []

# =====================================================
# I. j = 3/2 (Quartet)

# 1. |3/2, 3/2> = |111>
# Ket description: all qubits up.
state1 = ket("111")
basis_states.append(state1)

# 2. |3/2, 1/2> = (1/√3)(|110> + |101> + |011>)
# Ket description: uniform superposition of states with two 1's and one 0.
state2 = build_state({"110": 1/np.sqrt(3),
                      "101": 1/np.sqrt(3),
                      "011": 1/np.sqrt(3)})
basis_states.append(state2)

# 3. |3/2, -1/2> = (1/√3)(|100> + |010> + |001>)
# Ket description: uniform superposition of states with one 1 and two 0's.
state3 = build_state({"100": 1/np.sqrt(3),
                      "010": 1/np.sqrt(3),
                      "001": 1/np.sqrt(3)})
basis_states.append(state3)

# 4. |3/2, -3/2> = |000>
# Ket description: all qubits down.
state4 = ket("000")
basis_states.append(state4)

# =====================================================
# II. j = 1/2 (Doublets)

# For the doublet states we have two constructions:
# Set A: Coupling the first two qubits into a triplet.

# 5. |1/2, 1/2>_A = √(1/6)[ 2|110> - |101> - |011> ]
# Ket description: from triplet coupling of qubits 1 and 2.
state5 = build_state({"110": 2/np.sqrt(6),
                      "101": -1/np.sqrt(6),
                      "011": -1/np.sqrt(6)})
basis_states.append(state5)

# 6. |1/2, -1/2>_A = √(1/6)[ 2|001> - |010> - |100> ]
# Ket description: analogous construction for m = -1/2.
state6 = build_state({"001": 2/np.sqrt(6),
                      "010": -1/np.sqrt(6),
                      "100": -1/np.sqrt(6)})
basis_states.append(state6)

# Set B: Coupling the first two qubits into a singlet.

# 7. |1/2, 1/2>_B = (1/√2)[ |101> - |011> ]
# Ket description: from singlet coupling of qubits 1 and 2.
state7 = build_state({"101": 1/np.sqrt(2),
                      "011": -1/np.sqrt(2)})
basis_states.append(state7)

# 8. |1/2, -1/2>_B = (1/√2)[ |010> - |100> ]
# Ket description: analogous construction for m = -1/2.
state8 = build_state({"010": 1/np.sqrt(2),
                      "100": -1/np.sqrt(2)})
basis_states.append(state8)

# =====================================================
# Construct the unitary transformation matrix U.
# Its columns are the basis states defined above.
basis_states = basis_states[::-1]
U_CG_3 = np.column_stack(basis_states)

# Manual Twirling for N=3
def manual_N3_SU2_tw(rho):
    rho_CG = U_CG_3.conj().T @ rho @ U_CG_3
    B_00 = rho_CG[:4,:4]
    B_11 = rho_CG[4:,4:]
    
    Manual_tw = np.zeros_like(rho, dtype=np.complex128)
    J = 0
    temp = np.einsum('ikjk->ij', B_00.reshape(2,2,2,2))
    Manual_tw[:4,:4] = np.kron(temp, np.eye(2*J+1, dtype=np.complex128))/(2*J+1)
    J = 1
    Manual_tw[4:,4:] = np.trace(B_11) * np.eye(2*J+1, dtype=np.complex128)/(2*J+1)
    return U_CG_3 @ Manual_tw @ U_CG_3.conj().T

##### N = 4
# Helper: return the computational basis ket corresponding to a bit-string label.
# Here the leftmost bit is qubit 1 and the rightmost is qubit 4.
def ket(label):
    # label is a 4-character string, e.g. "1011"
    index = int(label, 2)
    v = np.zeros(16, dtype=complex)
    v[index] = 1.0
    return v

# Build a state as a linear combination of computational basis kets,
# given by a dictionary mapping bit-string labels to coefficients.
def add_state(coeff_dict):
    v = np.zeros(16, dtype=complex)
    for state, coeff in coeff_dict.items():
        v += coeff * ket(state)
    return v

basis_states = []  # list to hold our new basis vectors in the desired order
states_N4 = {}

# ====================================================
# I. j=2 (Quintet) states, total 5 states

# |2, 2> = |1111>
state_2_2 = ket("1111")
states_N4["2,2"] = state_2_2
basis_states.append(state_2_2)

# |2, 1> = (1/2)[|1110> + |1101> + |1011> + |0111>]
state_2_1 = add_state({
    "1110": 1/2, "1101": 1/2, "1011": 1/2, "0111": 1/2
})
states_N4["2,1"] = state_2_1
basis_states.append(state_2_1)

# |2, 0> = (1/√6)[|1100> + |1010> + |1001> + |0110> + |0101> + |0011>]
state_2_0 = add_state({
    "1100": 1/np.sqrt(6), "1010": 1/np.sqrt(6), "1001": 1/np.sqrt(6),
    "0110": 1/np.sqrt(6), "0101": 1/np.sqrt(6), "0011": 1/np.sqrt(6)
})
states_N4["2,0"] = state_2_0
basis_states.append(state_2_0)

# |2, -1> = (1/2)[|1000> + |0100> + |0010> + |0001>]
state_2_m1 = add_state({
    "1000": 1/2, "0100": 1/2, "0010": 1/2, "0001": 1/2
})
states_N4["2,-1"] = state_2_m1
basis_states.append(state_2_m1)

# |2, -2> = |0000>
state_2_m2 = ket("0000")
states_N4["2,-2"] = state_2_m2
basis_states.append(state_2_m2)

# ====================================================
# II. j=1 states, total 9 states.
# We choose three coupling paths: Sets A, B, and C.

# ---- Set A (from coupling two triplets: S12=1, S34=1) ----
# |1, 1>_A = 1/2 [ |1110> + |1101> − |1011> − |0111> ]
state_1_1_A = add_state({
    "1110": 1/2, "1101": 1/2, "1011": -1/2, "0111": -1/2
})
states_N4["1,1,0"] = state_1_1_A
basis_states.append(state_1_1_A)
# |1, 0>_A = (1/√2)[|1100> − |0011>]
state_1_0_A = add_state({
    "1100": 1/np.sqrt(2), "0011": -1/np.sqrt(2)
})
states_N4["1,0,0"] = state_1_0_A
basis_states.append(state_1_0_A)
# |1, -1>_A = 1/2 [ |1000> + |0100> − |0010> − |0001> ]
state_1_m1_A = add_state({
    "1000": 1/2, "0100": 1/2, "0010": -1/2, "0001": -1/2
})
states_N4["1,-1,0"] = state_1_m1_A
basis_states.append(state_1_m1_A)

# ---- Set B (S12=1, S34=0: pair 1-2 is triplet, 3-4 is singlet) ----
# |1, 1>_B = 1/√2 [ |1110> − |1101> ]
state_1_1_B = add_state({
    "1110": 1/np.sqrt(2), "1101": -1/np.sqrt(2)
})
states_N4["1,1,1"] = state_1_1_B
basis_states.append(state_1_1_B)
# |1, 0>_B = 1/2 [ |1010> − |1001> + |0110> − |0101> ]
state_1_0_B = add_state({
    "1010": 1/2, "1001": -1/2, "0110": 1/2, "0101": -1/2
})
states_N4["1,0,1"] = state_1_0_B
basis_states.append(state_1_0_B)
# |1, -1>_B = 1/√2 [ |0010> − |0001> ]
state_1_m1_B = add_state({
    "0010": 1/np.sqrt(2), "0001": -1/np.sqrt(2)
})
states_N4["1,-1,1"] = state_1_m1_B
basis_states.append(state_1_m1_B)

# ---- Set C (S12=0, S34=1: pair 1-2 is singlet, 3-4 is triplet) ----
# |1, 1>_C = 1/√2 [ |1011> − |0111> ]
state_1_1_C = add_state({
    "1011": 1/np.sqrt(2), "0111": -1/np.sqrt(2)
})
states_N4["1,1,2"] = state_1_1_C
basis_states.append(state_1_1_C)
# |1, 0>_C = 1/2 [ |1010> + |1001> − |0110> − |0101> ]
state_1_0_C = add_state({
    "1010": 1/2, "1001": 1/2, "0110": -1/2, "0101": -1/2
})
states_N4["1,0,2"] = state_1_0_C
basis_states.append(state_1_0_C)
# |1, -1>_C = 1/√2 [ |1000> − |0100> ]
state_1_m1_C = add_state({
    "1000": 1/np.sqrt(2), "0100": -1/np.sqrt(2)
})
states_N4["1,-1,2"] = state_1_m1_C
basis_states.append(state_1_m1_C)

# ====================================================
# III. j=0 states (Singlets), total 2 states.
# One common construction uses two coupling schemes.

# Singlet A (from two triplets):
# |0,0>_A = √(1/3)[ |1100> − ½(|1010> + |1001> + |0110> + |0101>) + |0011> ]
state_0_0_A = add_state({
    "1100": np.sqrt(1/3),
    "1010": -0.5*np.sqrt(1/3),
    "1001": -0.5*np.sqrt(1/3),
    "0110": -0.5*np.sqrt(1/3),
    "0101": -0.5*np.sqrt(1/3),
    "0011": np.sqrt(1/3)
})
states_N4["0,0,0"] = state_0_0_A
basis_states.append(state_0_0_A)
# Singlet B (from two singlets):
# |0,0>_B = 1/2 [ |1010> − |1001> − |0110> + |0101> ]
state_0_0_B = add_state({
    "1010": 1/2, "1001": -1/2, "0110": -1/2, "0101": 1/2
})
states_N4["0,0,1"] = state_0_0_B
basis_states.append(state_0_0_B)

# ====================================================
# Now stack the 16 basis state vectors as columns in a matrix.
U_CG_4 = np.column_stack(basis_states[::-1])

# Manual Twirling for N=4
def manual_N4_SU2_tw(rho):
    rho_CG = U_CG_4.conj().T @ rho @ U_CG_4
    B_0 = rho_CG[:2,:2] # J = 0
    B_1 = rho_CG[2:-5,2:-5] # J = 1
    B_2 = rho_CG[-5:,-5:] # J = 2
    
    Manual_tw = np.zeros_like(rho, dtype=np.complex128)
    J = 0
    Manual_tw[:2,:2] = B_0
    J = 1
    temp = np.einsum('ikjk->ij', B_1.reshape(3,3,3,3))
    Manual_tw[2:-5,2:-5] = np.kron(temp, np.eye(2*J+1, dtype=np.complex128))/(2*J+1)
    J = 2
    Manual_tw[-5:,-5:] = np.trace(B_2) * np.eye(2*J+1, dtype=np.complex128)/(2*J+1)
    return U_CG_4 @ Manual_tw @ U_CG_4.T.conjugate()


U_CG_6 = np.load("mask_memory/U_CG_6.npy")
def manual_N6_SU2_tw(rho):
    rho_CG = U_CG_6.conj().T @ rho @ U_CG_6
    a0 = 5*1
    a1 = a0+9*3
    a2 = a1+5*5
    # a3 = a2+1*7
    
    B_0 = rho_CG[  :a0,  :a0] # J = 0
    B_1 = rho_CG[a0:a1,a0:a1] # J = 1
    B_2 = rho_CG[a1:a2,a1:a2] # J = 2
    B_3 = rho_CG[a2:  ,a2:  ] # J = 3
    
    Manual_tw = np.zeros_like(rho, dtype=np.complex128)
    J = 0
    Manual_tw[  :a0,  :a0] = B_0
    J = 1
    temp = np.einsum('ikjk->ij', B_1.reshape(9,3,9,3))
    Manual_tw[a0:a1,a0:a1] = np.kron(temp, np.eye(2*J+1, dtype=np.complex128))/(2*J+1)
    J = 2
    temp = np.einsum('ikjk->ij', B_2.reshape(5,5,5,5))
    Manual_tw[a1:a2,a1:a2] = np.kron(temp, np.eye(2*J+1, dtype=np.complex128))/(2*J+1)
    J = 3
    Manual_tw[a2:  ,a2:  ] = np.trace(B_3) * np.eye(2*J+1, dtype=np.complex128)/(2*J+1)
    return U_CG_6 @ Manual_tw @ U_CG_6.T.conjugate()


##### N = 8
from sympy.physics.quantum.cg import CG
from sympy import S
from sympy import sympify

def cg_table_generator(j1, j2):
    """
    Build a table of Clebsch-Gordan coefficients for given j1 and j2.
    
    The table is stored in a dictionary where the keys are tuples
    (j1, m1, j2, m2, J, M) and the values are the corresponding CG coefficients.
    """
    # Dictionary to store the Clebsch-Gordan coefficients
    cg_table = {}

    # J can range from |j1-j2|=0 up to j1+j2=4
    for J in range(abs(j1-j2), j1+j2+1):
        # M goes from -J to +J
        for M in range(-J, J+1):
            # We look for all pairs (m1, m2) with m1+m2 = M
            for m1 in range(-j1, j1+1):
                m2 = M - m1
                # Ensure m2 is in the allowed range for j2
                if -j2 <= m2 <= j2:
                    # Sympy's CG(...) gives the Clebsch-Gordan coefficient
                    val = CG(S(j1), S(m1), S(j2), S(m2), S(J), S(M)).doit()
                    # If the coefficient is nonzero, store it
                    if val != 0:
                        # Convert to a nice string, or keep as a Sympy rational/sqrt expression
                        cg_table[(j1, m1, j2, m2, J, M)] = float(sympify(val))
    return cg_table

def print_table_grouped_by_M(j1, j2):
    r"""
    Prints one large table of Clebsch-Gordan coefficients for fixed j1, j2,
    but groups the columns by the total M value rather than by J.
    
    Layout details:
      - The allowed total M values run from -(j1+j2) to j1+j2.
      - For each M, only those J from J_min = |j1 - j2| to j1+j2
        that satisfy |M| <= J are included.
      - The leftmost columns label the (m1, m2) pair.
      - In each cell of the table, if m1+m2 equals the column's M,
        the cell shows the CG coefficient ⟨j1 m1, j2 m2 | J M⟩ (as stored in cg_table);
        otherwise the cell is left blank.
    
    This function assumes that a global dictionary 'cg_table' exists,
    whose keys are (j1, m1, j2, m2, J, M) and whose values are strings
    representing the CG coefficient (for example, "sqrt(2/5)").
    """
    cg_table = cg_table_generator(j1, j2)
    
    # Overall allowed M values:
    M_min = -int(j1 + j2)
    M_max = int(j1 + j2)
    M_values = list(range(M_min, M_max + 1))[::-1]
    
    # Allowed J values run from |j1-j2| to j1+j2.
    possible_J = [J for J in range(int(abs(j1 - j2)), int(j1 + j2) + 1)][::-1]
    # For each M, allowed J are only those with |M| <= J.
    M_to_J = {M: [J for J in possible_J if abs(M) <= J] for M in M_values}
    
    col_width = 12  # Width for each cell.
    left_label_width = 20  # Width for the left-hand (m1, m2) labels.
    
    # --- Print header rows ---
    # First header row: each block labeled by "M=..." spanning as many subcolumns as allowed J for that M.
    header1 = " " * left_label_width
    for M in M_values:
        block_width = col_width * len(M_to_J[M])
        header1 += f"{('M=' + str(M)):^{block_width}}|"
    print(header1)
    
    # Second header row: within each M block, label the subcolumns by "J=...".
    header2 = " " * left_label_width
    for M in M_values:
        for J in M_to_J[M]:
            header2 += f"{('J=' + str(J)):^{col_width}}"
        header2 += "|"
    print(header2)
    
    # Print a separator line.
    total_width = left_label_width + sum(col_width * len(M_to_J[M]) + 1 for M in M_values)
    print("-" * total_width)
    
    # --- List all (m1, m2) pairs ---
    # Here we list m1 from -j1 to j1 and m2 from -j2 to j2.
    m1_values = list(range(-int(j1), int(j1) + 1))[::-1]
    m2_values = list(range(-int(j2), int(j2) + 1))[::-1]
    # For a nicer ordering, list (m1, m2) pairs sorted by m1 then m2.
    m1m2_pairs = [(m1, m2) for m1 in m1_values for m2 in m2_values]

    # --- Generate table rows ---
    for (m1, m2) in m1m2_pairs:
        # Each row begins with the m1, m2 label.
        row_label = f" m1={m1:2}, m2={m2:2} |"
        row_str = f"{row_label:<{left_label_width}}"
        # The total M is fixed by the (m1, m2) pair.
        total_M = m1 + m2
        for M in M_values:
            for J in M_to_J[M]:
                # Only fill the cell if m1 + m2 equals this column's M.
                if total_M == M:
                    key = (j1, m1, j2, m2, J, M)
                    coef_str = cg_table.get(key, None)
                    if coef_str is None:
                        cell = " " * col_width
                    else:
                        # Attempt to convert the string expression into a float.
                        try:
                            # sympify evaluates expressions like "sqrt(2/5)".
                            coef_val = float(sympify(coef_str))
                            if abs(coef_val) < 1e-14:
                                cell = " " * col_width
                            else:
                                cell = f"{coef_val:^{col_width}.5f}"
                        except Exception:
                            # If conversion fails, display the string as is.
                            cell = f"{coef_str:^{col_width}}"
                else:
                    cell = " " * col_width
                row_str += cell
            row_str += "|"
        print(row_str)

'''
# Example usage:
print("Clebsch-Gordan coefficients for j1=2, j2=1:")
print_table_grouped_by_M(2, 1)
'''

def construct_state(J, M, j1, j2, r1=None, r2=None):
    CG_coeff = cg_table_generator(j1, j2)
    # print('constructing state for j1, j2 =', j1, j2)

    state = np.zeros(2**8, dtype=complex)
    for m2 in range(-j2, j2 + 1):
        for m1 in range(-j1, j1 + 1):
            if m1 + m2 == M:
                # print(j1, m1, j2, m2, J, M, r1, r2, 'coeff =', CG_coeff.get((j1, m1, j2, m2, J, M), 0))
                
                if r1 is None:
                    key1 = f"{j1},{m1}"
                else:
                    key1 = f"{j1},{m1},{r1}"
                if r2 is None:
                    key2 = f"{j2},{m2}"
                else:
                    key2 = f"{j2},{m2},{r2}"
                s1 = states_N4[key1]
                s2 = states_N4[key2]
                state += CG_coeff.get((j1, m1, j2, m2, J, M), 0) * np.kron(s1, s2)
    state /= np.linalg.norm(state)
    return state

# Helper: return the computational basis ket corresponding to a bit-string label.
# Here the leftmost bit is qubit 1 and the rightmost is qubit 4.
def ket(label):
    # label is a 4-character string, e.g. "1011"
    index = int(label, 2)
    v = np.zeros(2**8, dtype=complex)
    v[index] = 1.0
    return v

# Build a state as a linear combination of computational basis kets,
# given by a dictionary mapping bit-string labels to coefficients.
def add_state(coeff_dict):
    v = np.zeros(2**8, dtype=complex)
    for state, coeff in coeff_dict.items():
        v += coeff * ket(state)
    return v

Jz = build_Jz(8)
J2 = build_J2(8)
Jminus = build_Jminus(8)

basis_psis = []  # list to hold our new basis vectors in the desired order
# ====================================================
zero = np.array([1,0])
one = np.array([0,1])

# ^^^^^^^^^^^^^^^^ J = 4 ^^^^^^^^^^^^^^^^ [ 1 state ]
J = 4
# |4, 4> = |11111111>
psi_4_4 = ket("11111111")
basis_psis.append(psi_4_4)

# |4, 3> = (1/√2)[|11111110> + |11111011> + |11110111> + |11101111> + |11011111> + |10111111> + |01111111>]
psi_4_3 = Jminus @ psi_4_4
psi_4_3 /= np.linalg.norm(psi_4_3)
basis_psis.append(psi_4_3)

psi_4_2 = Jminus @ psi_4_3 
psi_4_2 /= np.linalg.norm(psi_4_2)
basis_psis.append(psi_4_2)

psi_4_1 = Jminus @ psi_4_2
psi_4_1 /= np.linalg.norm(psi_4_1)
basis_psis.append(psi_4_1)

psi_4_0 = Jminus @ psi_4_1
psi_4_0 /= np.linalg.norm(psi_4_0)
basis_psis.append(psi_4_0)

psi_4_m1 = Jminus @ psi_4_0
psi_4_m1 /= np.linalg.norm(psi_4_m1)
basis_psis.append(psi_4_m1)

psi_4_m2 = Jminus @ psi_4_m1
psi_4_m2 /= np.linalg.norm(psi_4_m2)
basis_psis.append(psi_4_m2)

psi_4_m3 = Jminus @ psi_4_m2
psi_4_m3 /= np.linalg.norm(psi_4_m3)
basis_psis.append(psi_4_m3)

psi_4_m4 = Jminus @ psi_4_m3
psi_4_m4 /= np.linalg.norm(psi_4_m4)
basis_psis.append(psi_4_m4)

# ^^^^^^^^^^^^^^^^ J = 3 ^^^^^^^^^^^^^^^^ [ 7 states ]
J = 3
# ---- Set A (from coupling two quintets: S1234=2, S5678=2) ----
# |3, 3>_A
psi_3_3_A = construct_state(3, 3, 2, 2)
# CG_coeff[2, 2, 2, 1, 3, 3] * np.kron(state_2_2, state_2_1) + \
#             CG_coeff[2, 1, 2, 2, 3, 3] * np.kron(state_2_1, state_2_2)
basis_psis.append(psi_3_3_A)
# |3, 2>_A
psi_3_2_A = construct_state(3, 2, 2, 2)
# CG_coeff.get((2, 2, 2, 0, 3, 2), 0) * np.kron(state_2_2, state_2_0) + \
#             CG_coeff.get((2, 1, 2, 1, 3, 2), 0) * np.kron(state_2_1, state_2_1) + \
#             CG_coeff.get((2, 0, 2, 2, 3, 2), 0) * np.kron(state_2_0, state_2_2)
basis_psis.append(psi_3_2_A)
# |3, 1>_A
psi_3_1_A = construct_state(3, 1, 2, 2)
# CG_coeff.get((2, 2, 2, -1, 3, 1), 0) * np.kron(state_2_2, state_2_m1) + \
#             CG_coeff.get((2, 1, 2, 0, 3, 1), 0) * np.kron(state_2_1, state_2_0) + \
#             CG_coeff.get((2, 0, 2, 1, 3, 1), 0) * np.kron(state_2_0, state_2_1) + \
#             CG_coeff.get((2, -1, 2, 2, 3, 1), 0) * np.kron(state_2_m1, state_2_2)
basis_psis.append(psi_3_1_A)
# |3, 0>_A
psi_3_0_A = construct_state(3, 0, 2, 2)
basis_psis.append(psi_3_0_A)
# |3, -1>_A
psi_3_m1_A = construct_state(3, -1, 2, 2)
basis_psis.append(psi_3_m1_A)
# |3, -2>_A
psi_3_m2_A = construct_state(3, -2, 2, 2)
basis_psis.append(psi_3_m2_A)
# |3, -3>_A
psi_3_m3_A = construct_state(3, -3, 2, 2)
basis_psis.append(psi_3_m3_A)

# ---- Set B (from coupling quintet and triplet: S1234=2, S5678=1) ----
# |3, M>_B -> there are 3 different triplets, first do magn then degeneracy
for r in range(3):
    for M in range(J, -J-1, -1):
        basis_psis.append(construct_state(3, M, 2, 1, None, r))
    
# ---- Set C (from coupling triplet and quintet: S1234=1, S5678=2) ----
# |3, M>_C -> there are 3 different triplets, first do magn then degeneracy
for r in range(3):
    for M in range(J, -J-1, -1):
        basis_psis.append(construct_state(3, M, 1, 2, r, None))
    
# ^^^^^^^^^^^^^^^^ J = 2 ^^^^^^^^^^^^^^^^ [ 20 states ]
J = 2
# ---- Set A (from coupling two quintets: S1234=2, S5678=2) ----
# |2, M>_A
for M in range(J, -J-1, -1):
    basis_psis.append(construct_state(2, M, 2, 2))
    
# ---- Set B (from coupling quintet and triplet: S1234=2, S5678=1) ----
# |2, M>_B -> there are 3 different triplets, first do magn then degeneracy
for r in range(3):
    for M in range(2, -2-1, -1):
        basis_psis.append(construct_state(2, M, 2, 1, None, r))
        
# ---- Set C (from coupling triplet and quintet: S1234=1, S5678=2) ----
# |2, M>_C -> there are 3 different triplets, first do magn then degeneracy
for r in range(3):
    for M in range(J, -J-1, -1):
        basis_psis.append(construct_state(2, M, 1, 2, r, None))
    
# ---- Set D (from coupling quintet and singlet: S1234=2, S5678=0) ----
# |3, M>_D -> there are 2 different singlets, first do magn then degeneracy
for r in range(2):
    for M in range(J, -J-1, -1):
        basis_psis.append(construct_state(2, M, 2, 0, None, r))
    
# ---- Set E (from coupling singlet and quintet: S1234=0, S5678=2) ----
# |3, M>_E -> there are 2 different singlets, first do magn then degeneracy 
for r in range(2):
    for M in range(J, -J-1, -1):
        basis_psis.append(construct_state(2, M, 0, 2, r, None))

# ---- Set F (from coupling triplet and triplet: S1234=1, S5678=1) ----
# |3, M>_F -> there are 9 different triplets, first do magn then degeneracy
for r1 in range(3):
    for r2 in range(3):
        for M in range(J, -J-1, -1):
            basis_psis.append(construct_state(2, M, 1, 1, r1, r2))
            
            
# ^^^^^^^^^^^^^^^^ J = 1 ^^^^^^^^^^^^^^^^ [ 28 states ]
J = 1
# ---- Set A (from coupling two quintets: S1234=2, S5678=2) ----
for M in range(J, -J-1, -1):
    basis_psis.append(construct_state(J, M, 2, 2))
    
# ---- Set B (from coupling quintet and triplet: S1234=2, S5678=1) ----
# |1, M>_B -> there are 3 different triplets, first do magn then degeneracy
for r in range(3):
    for M in range(J, -J-1, -1):
        basis_psis.append(construct_state(J, M, 2, 1, None, r))
# ---- Set C (from coupling triplet and quintet: S1234=1, S5678=2) ----
# |1, M>_C -> there are 3 different triplets, first do magn then degeneracy
for r in range(3):
    for M in range(J, -J-1, -1):
        basis_psis.append(construct_state(J, M, 1, 2, r, None))
# ---- Set D (from coupling triplet and triplet: S1234=1, S5678=1) ----
# |1, M>_D -> there are 9 different triplets, first do magn then degeneracy
for r1 in range(3):
    for r2 in range(3):
        for M in range(J, -J-1, -1):
            basis_psis.append(construct_state(J, M, 1, 1, r1, r2))
# ---- Set E (from coupling triplet and singlet: S1234=1, S5678=0) ----
for r1 in range(3):
    for r2 in range(2):
        for M in range(J, -J-1, -1):
            basis_psis.append(construct_state(J, M, 1, 0, r1, r2))
# ---- Set F (from coupling singlet and triplet: S1234=0, S5678=1) ----
for r1 in range(2):
    for r2 in range(3):
        for M in range(J, -J-1, -1):
            basis_psis.append(construct_state(J, M, 0, 1, r1, r2))
            
# ^^^^^^^^^^^^^^^^ J = 0 ^^^^^^^^^^^^^^^^ [ 14 states ]
J = 0
# ---- Set A (from coupling two quintets: S1234=2, S5678=2) ----
basis_psis.append(construct_state(J, 0, 2, 2))
# ---- Set B (from coupling triplet and triplet: S1234=1, S5678=1) ----
for r1 in range(3):
    for r2 in range(3):
        basis_psis.append(construct_state(J, 0, 1, 1, r1, r2))
# ---- Set C (from coupling singlet and singlet: S1234=0, S5678=0) ----
for r1 in range(2):
    for r2 in range(2):
        basis_psis.append(construct_state(J, 0, 0, 0, r1, r2))

print(len(basis_psis))

basis_psis = basis_psis[::-1]
U_CG_8 = np.column_stack(basis_psis)

''' Check of the J and M of the states:

a = 14*1
print('J = 0, J(J+1) =', 0*(1+0), 'length =', len(basis_psis[:a]))
fn.print_matrix(np.array([((s @ J2 @ s).real, (s @ Jz @ s).real) for s in basis_psis[:a]]).reshape(-1,2).T)
a, b = a+28*3, a
print('J = 1, J(J+1) =', 1*(1+1), 'length =', len(basis_psis[b:a]))
fn.print_matrix(np.array([((s @ J2 @ s).real, (s @ Jz @ s).real) for s in basis_psis[b:a]]).reshape(-1,2).T)
a, b = a+20*5, a
print('J = 2, J(J+1) =', 2*(1+2), 'length =', len(basis_psis[b:a]))
fn.print_matrix(np.array([((s @ J2 @ s).real, (s @ Jz @ s).real) for s in basis_psis[b:a]]).reshape(-1,2).T)
a, b = a+7*7, a
print('J = 3, J(J+1) =', 3*(1+3), 'length =', len(basis_psis[b:a]))
fn.print_matrix(np.array([((s @ J2 @ s).real, (s @ Jz @ s).real) for s in basis_psis[b:a]]).reshape(-1,2).T)
b = a
print('J = 4, J(J+1) =', 4*(1+4), 'length =', len(basis_psis[b:]))
fn.print_matrix(np.array([((s @ J2 @ s).real, (s @ Jz @ s).real) for s in basis_psis[b:]]).reshape(-1,2).T)

'''

# Manual Twirling for N=8
def manual_N8_SU2_tw(rho):
    rho_CG = U_CG_8.conj().T @ rho @ U_CG_8
    a0 = 14*1
    a1 = a0+28*3
    a2 = a1+20*5
    a3 = a2+7*7
    
    B_0 = rho_CG[:a0,:a0] # block with J = 0
    B_1 = rho_CG[a0:a1,a0:a1] # block with J = 1
    B_2 = rho_CG[a1:a2,a1:a2] # block with J = 2
    B_3 = rho_CG[a2:a3,a2:a3] # block with J = 3
    B_4 = rho_CG[a3:,a3:] # block with J = 4
    
    Manual_tw = np.zeros_like(rho, dtype=np.complex128)
    J = 0
    Manual_tw[:a0,:a0] = B_0
    J = 1
    temp = np.einsum('ikjk->ij', B_1.reshape(28,3,28,3)) # trace on the 3 magn degeneracies
    Manual_tw[a0:a1,a0:a1] = np.kron(temp, np.eye(2*J+1, dtype=np.complex128))/(2*J+1)
    J = 2
    temp = np.einsum('ikjk->ij', B_2.reshape(20,5,20,5))
    Manual_tw[a1:a2,a1:a2] = np.kron(temp, np.eye(2*J+1, dtype=np.complex128))/(2*J+1)
    J = 3
    temp = np.einsum('ikjk->ij', B_3.reshape(7,7,7,7))
    Manual_tw[a2:a3,a2:a3] = np.kron(temp, np.eye(2*J+1, dtype=np.complex128))/(2*J+1)
    J = 4
    Manual_tw[a3:,a3:] = np.trace(B_4) * np.eye(2*J+1, dtype=np.complex128)/(2*J+1)
    
    return U_CG_8 @ Manual_tw @ U_CG_8.T.conjugate()

# U_CG_12 = np.load("mask_memory/U_CG_12.npy")
def manual_N12_SU2_tw(rho):
    rho_CG = U_CG_12.conj().T @ rho @ U_CG_12
    a0 = 132*1
    a1 = a0+297*3
    a2 = a1+275*5
    a3 = a2+154*7
    a4 = a3+54*9
    a5 = a4+11*11
    
    B_0 = rho_CG[:a0,:a0] # block with J = 0
    B_1 = rho_CG[a0:a1,a0:a1] # block with J = 1
    B_2 = rho_CG[a1:a2,a1:a2] # block with J = 2
    B_3 = rho_CG[a2:a3,a2:a3] # block with J = 3
    B_4 = rho_CG[a3:a4,a3:a4] # block with J = 4
    B_5 = rho_CG[a4:a5,a4:a5] # block with J = 5
    B_6 = rho_CG[a5:,a5:] # block with J = 6
        
    Manual_tw = np.zeros_like(rho, dtype=np.complex128)
    J = 0
    Manual_tw[:a0,:a0] = B_0
    J = 1
    temp = np.einsum('ikjk->ij', B_1.reshape(297,3,297,3)) # trace on the 3 magn degeneracies
    Manual_tw[a0:a1,a0:a1] = np.kron(temp, np.eye(2*J+1, dtype=np.complex128))/(2*J+1)
    J = 2
    temp = np.einsum('ikjk->ij', B_2.reshape(275,5,275,5))
    Manual_tw[a1:a2,a1:a2] = np.kron(temp, np.eye(2*J+1, dtype=np.complex128))/(2*J+1)
    J = 3
    temp = np.einsum('ikjk->ij', B_3.reshape(154,7,154,7))
    Manual_tw[a2:a3,a2:a3] = np.kron(temp, np.eye(2*J+1, dtype=np.complex128))/(2*J+1)
    J = 4
    temp = np.einsum('ikjk->ij', B_4.reshape(54,9,54,9))
    Manual_tw[a3:a4,a3:a4] = np.kron(temp, np.eye(2*J+1, dtype=np.complex128))/(2*J+1)
    J = 5
    temp = np.einsum('ikjk->ij', B_5.reshape(11,11,11,11))
    Manual_tw[a4:a5,a4:a5] = np.kron(temp, np.eye(2*J+1, dtype=np.complex128))/(2*J+1)
    J = 6
    Manual_tw[a5:,a5:] = np.trace(B_6) * np.eye(2*J+1, dtype=np.complex128)/(2*J+1)
    
    # fn.print_matrix(Manual_tw)
    return U_CG_12 @ Manual_tw @ U_CG_12.T.conjugate()


from scipy.integrate import quad_vec
from scipy.linalg import expm
from functools import reduce
id_ = np.eye(2)
sx = np.array([[0.,1.+0j],[1.+0j,0.]])
sy = np.array([[0.,-1j],[1j,0.]])
sz = np.diag([1, -1])

def gen_Jx(N):
    """
    Generate the Jx operator for N spins.
    """
    Jx = np.zeros((2**N, 2**N), dtype=np.complex128)
    for i in range(N):
        ops = [id_] * N
        ops[i] = sx
        op = reduce(np.kron, ops)
        Jx += op
    return Jx
def gen_Jy(N):
    """
    Generate the Jy operator for N spins.
    """ 
    Jy = np.zeros((2**N, 2**N), dtype=np.complex128)
    for i in range(N):
        ops = [id_] * N
        ops[i] = sy
        op = reduce(np.kron, ops)
        Jy += op
    return Jy
def gen_Jz(N):
    """
    Generate the Jz operator for N spins.
    """ 
    Jz = np.zeros((2**N, 2**N), dtype=np.complex128)
    for i in range(N):
        ops = [id_] * N
        ops[i] = sz
        op = reduce(np.kron, ops)
        Jz += op
    return Jz


## Z2



def U_Z2_gen(n):
    """
    Return the 2^n × 2^n unitary U that reorders the computational basis
    into the Z2‐parity basis (even-parity states first, then odd-parity).
    
    Args:
        n (int): number of qubits.
    Returns:
        U (np.ndarray): a 2^n×2^n permutation (unitary) matrix.
    """
    dim = 2**n
    # compute parity (0=even, 1=odd) of each basis index
    parities = [bin(i).count("1") % 2 for i in range(dim)]
    # collect indices in even then odd order
    even_idx = [i for i, p in enumerate(parities) if p == 0]
    odd_idx  = [i for i, p in enumerate(parities) if p == 1]
    perm = even_idx + odd_idx

    # build permutation matrix: U[new_row, old_col] = 1
    U = np.zeros((dim, dim), dtype=complex)
    for new_row, old_col in enumerate(perm):
        U[new_row, old_col] = 1
    return U


def manual_Z2_tw(rho):
    """
    Manually apply the Z2 transformation to the density matrix rho.
    """
    # Get the dimensions of the density matrix
    dim = rho.shape[0]
    Ns = int(np.log2(dim))
    
    # Create a permutation matrix for Z2 transformation
    perm = np.zeros((dim, dim), dtype=complex)
    U_z2 = U_Z2_gen(Ns)
    
    # Fill the permutation matrix
    rho_Z2 = U_z2.T @ rho @ U_z2
    
    perm[:2**(Ns-1), :2**(Ns-1)] = rho_Z2[:2**(Ns-1), :2**(Ns-1)]
    perm[2**(Ns-1):, 2**(Ns-1):] = rho_Z2[2**(Ns-1):, 2**(Ns-1):]
    
    return U_z2 @ perm @ U_z2.T




def gate_Z2(params=None):
    """
    Create a 4x4 gate that combines two 2x2 gates A and B into a single 4x4 gate.
    The input parameters are the elements of the 2x2 gates A and B.
    """
        
    def Rz(phi):
        """Rotation about Z by angle φ."""
        return np.array([[np.exp(-0.5j*phi), 0],
                        [0,                np.exp(0.5j*phi)]])

    def Ry(theta):
        """Rotation about Y by angle θ."""
        return np.array([[ np.cos(0.5*theta), -np.sin(0.5*theta)],
                        [ np.sin(0.5*theta),  np.cos(0.5*theta)]])

    def random_bloch(params=None):
        """
        Returns a random U(2) via Z-Y-Z Euler decomposition:
        U = Rz(α) · Ry(β) · Rz(γ)
        with α, β, γ ∈ [0, 2π).
        """
        if params is not None:
            α, β, γ = params
        else:
            # Generate random angles in the range [0, 2π)
            α, β, γ = np.random.uniform(0, 2*np.pi, size=3)
        return Rz(α) @ Ry(β) @ Rz(γ)

    def combine_A_B(A, B):
        """
        Embed 2×2 gates A,B into 4×4 as
            [ A   0   0   A ]
            [ 0   B   B   0 ]
            [ 0   B   B   0 ]
            [ A   0   0   A ]
        i.e. in matrix form:
        U[0,0] = A[0,0], U[0,3] = A[0,1], U[3,0] = A[1,0], U[3,3] = A[1,1],
        U[1:3,1:3] = B.
        """
        U = np.zeros((4,4), dtype=complex)
        # even‐parity subspace (|00>, |11>) mixing:
        U[0,0] = A[0,0]
        # U[0,3] = A[0,1]
        # U[3,0] = A[1,0]
        U[3,3] = A[1,1]
        # odd‐parity subspace (|01>, |10>) mixing:
        U[1:3,1:3] = B
        return U
    
    if params is not None:
        a0, a1, a2 = params[:3]
        b0, b1, b2 = params[3:6]
    else:
        # Generate random angles in the range [0, 2π)
        a0, a1, a2 = np.random.uniform(0, 2*np.pi, size=3)
        b0, b1, b2 = np.random.uniform(0, 2*np.pi, size=3)
    A = random_bloch([a0, a1, a2])
    B = random_bloch([b0, b1, b2])
    return combine_A_B(A, B)


def permute_ZK_state(psi, K):
    """
    Permute the 2^n-dimensional state vector `psi` into ZK-parity order:
    all K-parity components first, then all (K+1)-parity components.
    
    Args:
        psi (np.ndarray): 1D complex array of length 2^n.
    Returns:
        np.ndarray: permuted state vector of the same shape.
    """
    dim = psi.shape[0]
    # compute parity mask: True where popcount(i) is even
    # use Python 3.8+ int.bit_count for fast popcount:
    parities = [bin(i).count("1") % K for i in range(dim)]

    perm = []
    for k in range(K):
        perm += [i for i, p in enumerate(parities) if p == k]
    # collect even- then odd-parity amplitudes
    return psi[perm]


def sector_sizes(Ns, K):
    """
    Compute the sizes of the subsectors for Z_K symmetry in an Ns-qubit system.
    
    Parameters:
    - Ns (int): Number of qubits.
    - K (int): Order of the Z_K symmetry.
    
    Returns:
    - List[int]: A list of length K where the q-th entry is the dimension of the subsector
                 with total charge q (mod K).
    """
    import math
    sizes = [0] * K
    for s in range(Ns + 1):
        count = math.comb(Ns, s)
        sizes[s % K] += count
    return np.array(sizes)


def U_ZK_gen(Ns, K):
    """
    Return the 2^Ns x 2^Ns unitary U that reorders the computational basis
    into the ZK-parity basis (even-parity states first, then odd-parity).
    
    Args:
        Ns (int): number of qubits.
        K (int): number of qubits in each sector.
    Returns:
        U (np.ndarray): a 2^Ns x 2^Ns permutation (unitary) matrix.
    """
    dim = 2**Ns
    # compute parity (0=even, 1=odd) of each basis index
    parities = [bin(i).count("1") % K for i in range(dim)]
    # collect indices in even then odd order
    perm = []
    for k in range(K):
        perm += [i for i, p in enumerate(parities) if p == k]

    # build permutation matrix: U[new_row, old_col] = 1
    U = np.zeros((dim, dim), dtype=complex)
    for new_row, old_col in enumerate(perm):
        U[new_row, old_col] = 1
    return U

def manual_Z2_tw(rho):
    """
    Manually apply the Z_2 transformation to the density matrix rho.
    """
    # Get the dimensions of the density matrix
    dim = rho.shape[0]
    Ns = int(np.log2(dim))
    
    # Create a permutation matrix for Z2 transformation
    perm = np.zeros((dim, dim), dtype=complex)
    U_z2 = U_Z2_gen(Ns)
    
    # Fill the permutation matrix
    rho_Z2 = U_z2.T @ rho @ U_z2
    
    perm[:2**(Ns-1), :2**(Ns-1)] = rho_Z2[:2**(Ns-1), :2**(Ns-1)]
    perm[2**(Ns-1):, 2**(Ns-1):] = rho_Z2[2**(Ns-1):, 2**(Ns-1):]
    
    return U_z2 @ perm @ U_z2.T


def manual_ZK_tw(rho, K):
    """
    Manually apply the Z_K transformation to the density matrix rho.
    """
    # Get the dimensions of the density matrix
    dim = rho.shape[0]
    Ns = int(np.log2(dim))
    
    # Create a permutation matrix for Z2 transformation
    perm = np.zeros((dim, dim), dtype=complex)
    U_ZK = U_ZK_gen(Ns, K)
    
    # Fill the permutation matrix
    rho_ZK = U_ZK @ rho @ U_ZK.T.conj()
    
    subspace_dims = sector_sizes(Ns, K)
    for i, (start, end) in enumerate(zip(np.r_[0, np.cumsum(subspace_dims[:-1])], np.cumsum(subspace_dims))):
        perm[start:end, start:end] = rho_ZK[start:end, start:end]
    return U_ZK.T.conj() @ perm @ U_ZK

def generate_Zk_gate(K, alphaT):
    """
    Generate a 2^K x 2^K unitary U = exp(i H / alpha) that preserves Z_K symmetry:
      [U, Q] = 0
    where Q |b> = exp(2πi * (popcount(b) % K) / K) |b>.

    Parameters
    ----------
    K : int
        Number of qubits (so dimension D = 2**K).
    alpha : float
        Scale parameter: U = exp(i H / alpha).

    Returns
    -------
    U : complex128 ndarray, shape (2**K, 2**K)
        The symmetry-preserving unitary.
    """
    blocks = []
    sectors_K = sector_sizes(K, K)
    U_ZK_g = U_ZK_gen(K, K)
    
    for size in sectors_K:
        # Build a GUE Hermitian block of size n:
        X = (np.random.randn(size, size) + 1j*np.random.randn(size, size)) / np.sqrt(2)
        Hblock = X + X.conj().T
        Ublock = expm(1j * Hblock / alphaT)
        
    # create a block-diagonal matrix with the blocks:
        blocks.append(Ublock)
        
    # Create the block-diagonal matrix:
    matrix = reduce(lambda a, b: np.block([[a, np.zeros((a.shape[0], b.shape[1]))], 
                                         [np.zeros((b.shape[0], a.shape[1])), b]]), blocks)
    return U_ZK_g.conj().T @ matrix @ U_ZK_g

# @njit(cache=True)
def asymmetry_modes(rho: np.ndarray, Ls: np.ndarray) -> np.ndarray:
    """
    Split rho into its modes of asymmetry.

    Parameters
    ----------
    rho : (D, D) ndarray
        Density matrix, where D = sum(Ls).
    Ls : (K,) ndarray of int
        Sector sizes [l0, l1, ..., l_{K-1}].

    Returns
    -------
    modes : (K, D, D) ndarray
        modes[omega] = matrix containing only those blocks of rho
        where the sector-index difference is omega (i.e. between
        sector i on the row and sector i+omega on the column).
    """
    # Number of sectors and total dimension
    Ls = np.asarray(Ls, dtype=int)
    K = Ls.shape[0]
    D = Ls.sum()
    if rho.shape != (D, D):
        raise ValueError(f"rho must be {D}×{D}, got {rho.shape}")

    # Compute starting offsets of each sector
    offsets = np.concatenate([[0], np.cumsum(Ls)])
    # Prepare output array
    modes = np.zeros((K, D, D), dtype=rho.dtype)

    # For each ω from 0 to K−1, pick blocks with sector‐index difference = ω
    for omega in range(K):
        M = modes[omega]
        # loop over sectors i such that i+omega < K
        for i in range(K - omega):
            start_i, end_i = offsets[i], offsets[i+1]
            start_j, end_j = offsets[i+omega], offsets[i+omega+1]
            # copy the block (i, i+ω)
            M[start_i:end_i, start_j:end_j] = rho[start_i:end_i, start_j:end_j]
        # everything else stays zero
    return modes



@njit(parallel=True, fastmath=True, cache=True)
def asymmetry_modes_pure(psi: np.ndarray, Ls: np.ndarray) -> np.ndarray:
    """
    Split |psi> into its modes of asymmetry without building rho=|psi><psi|.
    
    Parameters
    ----------
    psi : (D,) ndarray
        The state vector, normalized so that sum(|psi|^2)=1.
    Ls : (K,) ndarray of int
        Sector sizes [l0, l1, ..., l_{K-1}], sum(Ls)=D.
    
    Returns
    -------
    modes : (K,) ndarray
        modes[omega] = sum_i sqrt(p_i * p_{i+omega}), where
        p_i = sum_{r in sector i} |psi_r|^2.
    """
    Q = Ls.shape[0]
    
    # compute cuts: where each sector starts and ends
    cuts = np.empty(Q+1, np.int64)
    cuts[0] = 0
    for i in range(Q):
        cuts[i+1] = cuts[i] + Ls[i]
    
    # sector populations p_i = sum_{r in sector i} |psi_r|^2
    p = np.zeros(Q, np.float64)
    for i in prange(Q):
        r0, r1 = cuts[i], cuts[i+1]
        acc = 0.0
        for r in range(r0, r1):
            # |psi_r|^2
            acc += (psi[r].real * psi[r].real + psi[r].imag * psi[r].imag)
        p[i] = acc
    
    # build the omega-weights
    modes = np.zeros(Q, np.float64)
    for omega in prange(Q):
        acc = 0.0
        # only pairs i, j=i+omega within bounds
        for i in range(Q-omega):
            acc += np.sqrt(p[i] * p[i+omega])
        modes[omega] = acc
    
    return modes


def compute_norm(op):
    """
    Compute the trace-norm (sum of singular values) of a matrix/operator safely.

    Parameters
    ----------
    op : array-like, shape (m, n)
        The input operator.

    Returns
    -------
    float
        The nuclear norm of `op`.
    """
    # 1) Coerce to a 2D ndarray
    A = np.asarray(op, dtype=np.complex128)
    if A.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {A.shape}")

    # 2) Compute singular values via SVD (more stable than sqrtm of A A^*)
    #    Only singular values are needed, so compute_uv=False
    s = np.linalg.svd(A, compute_uv=False)

    # 3) Sum singular values (trace‐norm) and return as a real float
    return float(np.sum(s))