import numpy as np
import functions as fn
from tqdm import tqdm
from circuit_obj import Circuit
from itertools import product
from scipy.special import comb
import argparse
import os  # Add this import at the top of the file

##############################################################################################

N = 20

# Circuit parameters
T = 10
parser = argparse.ArgumentParser(description="Run circuit simulations.")
parser.add_argument("--circuit_to_run", type=float, required=True, help="Circuit realization to run")
parser.add_argument("--theta_to_run", type=float, required=True, help="Theta (* pi) parameter to run")
args = parser.parse_args()

circuit_to_run = int(args.circuit_to_run)
circuit_realizations = 100
symmetry = ['U1', 'SU2', 'Z2', 'ZK'][0]
geometry = ['random', 'brickwork'][1]
alphaT = 1 # Parameter to slice the circuit thinner
if symmetry == 'ZK':
    K = 8

# Mask parameters
sites_to_keep = range((N-8), N)
qkeep = np.array(sites_to_keep)
qthrow = np.arange(N)[np.isin(np.arange(N), qkeep, invert=True)]
Ns = len(sites_to_keep)

Ne = N - Ns
projectors_s, U_U1_s = fn.build_projectors(Ns) 
projectors_e, U_U1_e = fn.build_projectors(Ne)
sectors_se = np.array([int(comb(N, int(N/2+m))) for m in np.arange(-N/2, N/2+1)])
sectors_s = np.array([int(comb(Ns, int(Ns/2+m))) for m in np.arange(-Ns/2, Ns/2+1)])
sectors_e = np.array([int(comb(Ne, int(Ne/2+m))) for m in np.arange(-Ne/2, Ne/2+1)])

masks_dict = fn.load_mask_memory(N, (2 if symmetry!='ZK' else K))

alphas = np.array([1]) # np.r_[np.linspace(0.1, 2, 10)]
theta = args.theta_to_run * np.pi
# thetas = np.array([.3, .35, .4, .45, .5])*np.pi # theta

##############################################################################################

circuits = []
circuit_realizations_max = 100
if symmetry == 'U1':
    try:
        h_list_all = np.load(f'data/U1_rnd_parameters.npy')
        print('Loaded U1 parameters')
    except FileNotFoundError:
        h_list_all = np.random.uniform(-np.pi, np.pi, 5*N*circuit_realizations_max).reshape(circuit_realizations_max, N, 5) /alphaT
        np.save(f'data/U1_rnd_parameters.npy', h_list_all)
        print('Generated U1 parameters')
elif symmetry == 'SU2':
    try:
        h_list_all = np.load(f'data/SU2_rnd_parameters.npy')
    except FileNotFoundError:
        h_list_all = np.random.uniform(-np.pi, np.pi, 1*N*circuit_realizations_max).reshape(circuit_realizations_max, N, 1) /alphaT
        np.save(f'data/SU2_rnd_parameters.npy', h_list_all)
        
snapshots_t = np.array([t for t in [0, 1, 2, 10, 100, 1000, 9999] if t <= T])

for circuit_realization in range(circuit_realizations):
    if symmetry == 'U1':
        h_list = h_list_all[circuit_realization]
        gates = [fn.gen_u1([*h]) for h in h_list]
    elif symmetry == 'SU2':
        h_list = h_list_all[circuit_realization]
        gates = [fn.gen_su2(*h) for h in h_list]

    order = fn.gen_gates_order(N, geometry=geometry)    
    circuit = Circuit(N=N, T=T, gates=gates, order=order, symmetry=symmetry)
    circuit.projectors = projectors_s
    circuit.U_U1_s = U_U1_s
    circuit.sectors_s = sectors_s
    circuit.Ns = Ns
    circuit.snapshots_t = snapshots_t
    circuits.append(circuit)
    
##############################################################################################

def compute_circuit(theta):
    circuit = circuits[circuit_to_run]
    rho = fn.initial_state(N, sites_to_keep, theta, 'homogenous') # initial_state_test(theta)
    return circuit.run(masks_dict, sites_to_keep, alphas, rho)

print(f'theta: {theta/np.pi:.2f} pi, circuit: {circuit_to_run}')
renyi, norms_s, evo = compute_circuit(theta)

renyi_dir = f'results/renyi/{symmetry}/theta{theta/np.pi:.2f}/T{T}/circuit_real{circuit_to_run}'
print('name folder renyi:', renyi_dir)
os.makedirs(renyi_dir, exist_ok=True)
np.savez(f'{renyi_dir}/data.npz', renyi)

norms_dir = f'results/norms_s/{symmetry}/theta{theta/np.pi:.2f}/T{T}/circuit_real{circuit_to_run}'
print('name folder norms_s:', norms_dir)
os.makedirs(norms_dir, exist_ok=True)
np.savez(f'{norms_dir}/data.npz', norms_s)

evo_dir = f'results/evo/{symmetry}/theta{theta/np.pi:.2f}/T{T}/circuit_real{circuit_to_run}'
print('name folder evo:', evo_dir)
os.makedirs(evo_dir, exist_ok=True)
np.savez(f'{evo_dir}/data.npz', evo)
