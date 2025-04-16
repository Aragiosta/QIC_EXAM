# For external bash handling of the sim parameters
import argparse
# For time diagnostics
from timeit import default_timer as timer
# All the mathsy stuff
import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt
# Specific modules for the simulation
from qic_ssh import init_MPS as MPS
from qic_ssh import setup

# Define the parser
parser = argparse.ArgumentParser(description='QIAC exam')

# Declare an argument (`--algo`), saying that the 
# corresponding value should be stored in the `algo` 
# field, and using a default value if the argument 
# isn't given
parser.add_argument('--topological', action="store", dest='topological', default='True')
parser.add_argument('--n_sites', action="store", dest='n_sites', default=0, type=int)
parser.add_argument('--model', action="store", dest='model', default="ideal")

# Now, parse the command line arguments and store the 
# values in the `args` variable
args = parser.parse_args()
if args.topological == 'True':
    args.topological = True
else:
    args.topological = False
# For when debbuging is on Friday
args.model = "ideal"
args.topological = False
args.n_sites = 6

# List of computational time used by the main blocks of the execution
times = {}

# Time the setup of the system
start = timer()

par = setup.Param(args.n_sites)
# Now try to use it and reproduce the paper
if args.model == "ideal":
    J = setup.ideal(par, topological=args.topological)
elif args.model == "paper":
    J = setup.paper(par, topological=args.topological)
else:
    J = np.empty((par.n_qbits, par.n_qbits))

model_params = {
    'bc_MPS': 'finite',
    'L': par.n_qbits,
    'lattice': 'Chain',
    'bc_x': 'open',
    'order': 'default',
    'conserve': 'None'
}

# chain = tenpy.models.lattice.Chain(par.n_qbits, SpinHalfSite(conserve='None'), bc='open', bc_MPS='finite')

# model_params['lattice'] = chain

paper = MPS.XYSystem(model_params=model_params, J=J)
# Then compute time difference
times['Init'] = timer() - start

# We must first find the starting WF for the DMRG algorithm
# file_path = r"DATA_SERVER/data_8_topo_ideal/Eigvecs8_ideal_topoTrue.data"
# if os.path.exists(file_path):
#     eigstates = np.loadtxt(file_path, dtype=complex, delimiter=',')
#     p_state = eigstates[:, 0:5]
# else:
#     p_state = None
# ALTERNATIVELY: find eigvecs starting from generic random wf

# Time the eigensolver of the MB hamiltonian
start = timer()
p_state = np.random.rand(par.lin_size, 4)
alg_params = {
            'trunc_params': {
                'chi_max': 30,
                'svd_min': 1.e-10,
            },
            'max_sweeps': 40,
            # 'min_sweeps': 10
}
paper.do_DMRG(alg_params, p_state)
# Then compute time difference
times['Eig'] = timer() - start

# Time the entanglement spectrum of the GS
start = timer()
# Compute entanglement of GS
ent_entropy = paper.eigvecs[0].entanglement_spectrum()[par.n_qbits // 2 - 1]
# Then compute time difference
times['Eig'] = timer() - start

# Time the computation of the order parameters
start = timer()
# Compute order parameters
z_parameter = paper.z_string()
x_parameter = paper.x_string()
# Then compute time difference
times['Strings'] = timer() - start

# Time the entanglement spectrum of the GS
start = timer()

# Compute correlation matrices
zz_corr = paper.eigvecs[0].correlation_function('Sz', 'Sz', hermitian=True)

xx_corr = paper.eigvecs[0].correlation_function('Sx', 'Sx', hermitian=True)
# Then compute time difference
times['2p_corr'] = timer() - start

# e_N = utils.order_eigvals(par, paper.eigvals, paper.eigvecs)
# plt.eventplot(e_N, orientation='vertical')
# plt.show()

# # Data production
parameters = '''N_qbits: %s \n Int_matrix: \n%s''' % (par.n_qbits, J)

# # # Better not - leads to huge files
# # # np.savetxt(
# # #     f"Ham{par.n_qbits}_{args.model}_topo{args.topological}.data",
# # #     paper.H.toarray(),
# # #     fmt="%10.8f",
# # #     delimiter=", ",
# # #     header=f"Hamiltonian of {par.n_qbits} sites obeying the {args.model} model")

start = timer()
np.savetxt(
    f"Eigvals{par.n_qbits}_{args.model}_topo{args.topological}.data",
    paper.eigvals,
    fmt="%10.8f",
    delimiter=", ",
    header=f"parameters: {parameters}")

# np.savetxt(
#     f"Eigvecs{par.n_qbits}_{args.model}_topo{args.topological}.data",
#     np.array([paper.eigvecs[ii].get_theta() for ii in range(4)]).reshape(par.lin_size, 4),
#     fmt="%10.8f",
#     delimiter=", ",
#     header=f"parameters: {parameters}. \n Columns are eigstates of the {args.model} model")

np.savetxt(
    f"GS{par.n_qbits}_{args.model}_topo{args.topological}.data",
    paper.eigvecs[0].get_theta(0, par.n_qbits).to_ndarray().reshape(-1),
    fmt="%10.8f",
    delimiter=", ",
    header=f"parameters: {parameters}. \n Columns are eigstates of the {args.model} model")

import json

# with open(f"EnergyToNumber{par.n_qbits}_{args.model}_topo{args.topological}.data", 'w') as file:
#     file.write(json.dumps(energy_list, indent='\t')) # use `json.loads` to do the reverse

with open(f"SweepStats{par.n_qbits}_{args.model}_topo{args.topological}.data", 'w') as file:
    file.write(json.dump(paper.dmrg_sweep_stats, indent='\t')) # use `json.loads` to do the reverse

with open(f"UpdateStats{par.n_qbits}_{args.model}_topo{args.topological}.data", 'w') as file:
    file.write(json.dumps(paper.dmrg_update_stats, indent='\t')) # use `json.loads` to do the reverse

# np.savetxt(
#     f"EnergyToNumber{par.n_qbits}_{args.model}_topo{args.topological}.data",
#     energy_list,
#     delimiter=", ",
#     header=f"parameters: {parameters}")

np.savetxt(
    f"EntEntropy{par.n_qbits}_{args.model}_topo{args.topological}.data",
    ent_entropy,
    fmt="%10.8f",
    delimiter=", ",
    header=f"parameters: {parameters} \n Bipartite entropy on the GS of the {args.model} model")

np.savetxt(
    f"StringParameters{par.n_qbits}_{args.model}_topo{args.topological}.data",
    [z_parameter, x_parameter],
    fmt="%10.8f",
    delimiter=", ",
    header=f"parameters: {parameters} \n z and x string VEV of the {args.model} model")

np.savetxt(
    f"ZZCorrelators{par.n_qbits}_{args.model}_topo{args.topological}.data",
    zz_corr,
    fmt="%10.8f",
    delimiter=", ",
    header=f"zz correlator on the GS of the {args.model} model\n parameters: {parameters}")

np.savetxt(
    f"XXCorrelators{par.n_qbits}_{args.model}_topo{args.topological}.data",
    xx_corr,
    fmt="%10.8f",
    delimiter=", ",
    header=f"xx correlator on the GS of the {args.model} model\n parameters: {parameters}")

times['IO_handling'] = timer() - start

with open(f"Times{par.n_qbits}_{args.model}_topo{args.topological}.data", 'w') as file:
    file.write(json.dumps(times, indent='\t')) # use `json.loads` to do the reverse