import numpy as np
import functions as fn
import time

def print_time(t, T, start_time):

    end_time = time.time()
    elapsed =  - start_time
    avg_per_iter = elapsed / t
    remaining = avg_per_iter * (T - t)

    # format ETA as H:MM:SS
    eta_h = int(remaining) // 3600
    eta_m = (int(remaining) % 3600) // 60
    eta_s = int(remaining) % 60

    print(
        f"[{t}/{T}] "
        f"Elapsed: {elapsed:.1f}s, "
        f"ETA: {eta_h:d}:{eta_m:02d}:{eta_s:02d}"
    )
    return end_time


class Circuit:
    def __init__(self, N, T, gates, order=None, symmetry=None, K=None):
        """
        Initialize the circuit object.
        - N is the number of qubits.
        - T is the number of time steps.
        - gates is either the list of gates or of parameters: 
            [circuit_type, geometry, Js] or [circuit_type, geometry]
            where if u1 is used, Js = [J, Jz] and if su2 is used, Js = J = Jz.
          if no Js is reported, the gates are set randomly.
        - initial_state is either the state or the parameters:
            [state_type, p, state_phases, theta]
        - order is the order of the gates.
        """
        self.N = N
        self.T = T
        self.order = order
        self.gates = gates
        self.symmetry = symmetry
        self.projectors = None
        if symmetry == 'ZK':
            self.K = K
        else:
            self.K = 2
            
    def generate_unitary(self, masks_dict):
        """
        Generate the unitary operator for the circuit.
        """
        # apply the circuit to each state of the computational basis
        unitary = np.zeros((2**self.N, 2**self.N), dtype=np.complex128)
        for i in range(2**self.N):
            state = np.zeros((2**self.N,), dtype=np.complex128)
            state[i] = 1
            state = fn.apply_U(state, self.gates, self.order, masks_dict, (2 if self.symmetry!='ZK' else self.K))
            unitary[:, i] = state
        return unitary

    def run(self, masks_dict, sites_to_keep, alphas, state):
        """
        Run the circuit and calculate the QFIs and EA.
        """                
        renyi = np.zeros((len(alphas), self.T + 1))
        # state_evolution = np.zeros((self.T + 1, 2**self.N), dtype=np.complex128)
        norms_s = np.zeros((self.T + 1, self.Ns + 1), dtype=np.float64)
        snapshots = np.zeros((len(self.snapshots_t), 2**self.N), dtype=np.complex128)        
        rho_s = fn.ptrace(state, sites_to_keep)
        
        if self.symmetry == 'U1':           
            rho_s_tw = fn.manual_U1_tw(rho_s, self.projectors)
            
        rho_s_U1 = self.U_U1_s.conj().T @ rho_s @ self.U_U1_s 
        rho_modes_s = fn.asymmetry_modes(rho_s_U1, self.sectors_s)
        
        # state_evolution[0] = state
        norms_s[0, :] = [fn.compute_norm(rho_om) for rho_om in rho_modes_s]
        renyi[:, 0] = [fn.renyi_divergence(rho_s, rho_s_tw, alpha) for alpha in alphas]
        
        t_snap = 0
        snapshots[t_snap, :] = state
        
        start_time = time.time()
        
        for t in range(1,self.T+1):
            
            
            state = fn.apply_U(state, self.gates, self.order, masks_dict, (None if self.symmetry!='ZK' else self.K))            
            rho_s = fn.ptrace(state.copy(), sites_to_keep)
            
            if self.symmetry == 'U1':           
                rho_s_tw = fn.manual_U1_tw(rho_s, self.projectors)
                
            rho_s_U1 = self.U_U1_s.conj().T @ rho_s @ self.U_U1_s 
            rho_modes = fn.asymmetry_modes(rho_s_U1, self.sectors_s)
            
            # state_evolution[t] = state
            norms_s[t, : ] = [fn.compute_norm(rho_om) for rho_om in rho_modes]
            
            if t in self.snapshots_t:
                t_snap += 1
                snapshots[t_snap, :] = state
                
            if t % 100 == 0:
                start_time = print_time(t, self.T, start_time)
            
            renyi[:, t] = [fn.renyi_divergence(rho_s, rho_s_tw, alpha) for alpha in alphas]

                
        return renyi, norms_s, snapshots #

    def compute_hamiltonian(self, masks_dict):
        """
        Compute the Hamiltonian of the circuit.
        """
        return fn.compute_hamiltonian(self.gates, self.order, masks_dict, self.N)