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
from qic_ssh import init_ED as ED
from qic_ssh import setup
from qic_ssh import utils

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

if globals().get('topological', None) is not None:
    args.n_sites = globals().get('n_sites', None)
    args.topological = globals().get('topological', None)
    args.model = globals().get('model', None)

if args.topological == 'True':
    args.topological = True
else:
    args.topological = False

# For when debbuging is on Friday
args.model = "paper"
args.topological = True
args.n_sites = 8

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

# J[np.abs(J) <= 1e-13] = 0
paper = ED.XYSystem(J)

# Then compute time difference
times['Init'] = timer() - start

# Time the eigensolver of the MB hamiltonian
start = timer()
# paper.eig(k=par.lin_size)
paper.eig(k=par.lin_size)
# Then compute time difference
times['Eig'] = timer() - start

# Time the creation of the energy-n. part. graph
start = timer()
# energy_list = utils.find_num_fermions(par, paper.eigvecs[:, paper.eigvals == max(paper.eigvals)])
energy_list = utils.order_eigvals(par, paper.eigvals, paper.eigvecs)
# Then compute time difference
times['N_part'] = timer() - start

# Time the entanglement spectrum of the GS
start = timer()
# Compute entanglement of GS
ent_entropy = utils.entanglement_spectrum(par, paper.eigvecs[:, 0])
# Then compute time difference
times['Eig'] = timer() - start

# Time the computation of the order parameters
start = timer()
# Compute order parameters
z_parameter = utils.z_string(par, paper.eigvecs[:, 0])
x_parameter = utils.x_string(par, paper.eigvecs[:, 0])
# Then compute time difference
times['Strings'] = timer() - start

# Time the entanglement spectrum of the GS
start = timer()
# Compute correlation matrices
zz_corr = np.array([
    utils.zz_correlation(par, paper.eigvecs[:, 0], [ii, jj])
    for (ii, jj), J in np.ndenumerate(J)
]).reshape(par.n_qbits, par.n_qbits)

xx_corr = np.array([
    utils.xx_correlation(par, paper.eigvecs[:, 0], [ii, jj])
    for (ii, jj), J in np.ndenumerate(J)
]).reshape(par.n_qbits, par.n_qbits)
# Then compute time difference
times['2p_corr'] = timer() - start


# e_N = utils.order_eigvals(par, paper.eigvals, paper.eigvecs)
# plt.eventplot(e_N, orientation='vertical')
# plt.show()

# Data production
parameters = '''N_qbits: %s \n Int_matrix: \n %s''' % (par.n_qbits, J)

# # Better not - leads to huge files
# # np.savetxt(
# #     f"Ham{par.n_qbits}_{args.model}_topo{args.topological}.data",
# #     paper.H.toarray(),
# #     fmt="%10.8f",
# #     delimiter=", ",
# #     header=f"Hamiltonian of {par.n_qbits} sites obeying the {args.model} model")

# start = timer()
# np.savetxt(
#     f"Eigvals{par.n_qbits}_{args.model}_topo{args.topological}.data",
#     paper.eigvals,
#     fmt="%10.8f",
#     delimiter=", ",
#     header=f"parameters: {parameters}")

# np.savetxt(
#     f"Eigvecs{par.n_qbits}_{args.model}_topo{args.topological}.data",
#     paper.eigvecs,
#     fmt="%10.8f",
#     delimiter=", ",
#     header=f"parameters: {parameters}. \n Columns are eigstates of the {args.model} model")

# np.savetxt(
#     f"GS{par.n_qbits}_{args.model}_topo{args.topological}.data",
#     paper.eigvecs[:, 0],
#     fmt="%10.8f",
#     delimiter=", ",
#     header=f"parameters: {parameters}. \n Columns are eigstates of the {args.model} model")

# import json

# with open(f"EnergyToNumber{par.n_qbits}_{args.model}_topo{args.topological}.data", 'w') as file:
#     file.write(json.dumps(energy_list, indent='\t')) # use `json.loads` to do the reverse

# # np.savetxt(
# #     f"EnergyToNumber{par.n_qbits}_{args.model}_topo{args.topological}.data",
# #     energy_list,
# #     delimiter=", ",
# #     header=f"parameters: {parameters}")

# np.savetxt(
#     f"EntEntropy{par.n_qbits}_{args.model}_topo{args.topological}.data",
#     np.array([ent_entropy]),
#     fmt="%10.8f",
#     delimiter=", ",
#     header=f"parameters: {parameters} \n Bipartite entropy on the GS of the {args.model} model")

# np.savetxt(
#     f"StringParameters{par.n_qbits}_{args.model}_topo{args.topological}.data",
#     [z_parameter, x_parameter],
#     fmt="%10.8f",
#     delimiter=", ",
#     header=f"parameters: {parameters} \n z and x string VEV of the {args.model} model")

# np.savetxt(
#     f"ZZCorrelators{par.n_qbits}_{args.model}_topo{args.topological}.data",
#     zz_corr,
#     fmt="%10.8f",
#     delimiter=", ",
#     header=f"zz correlator on the GS of the {args.model} model\n parameters: {parameters}")

# np.savetxt(
#     f"XXCorrelators{par.n_qbits}_{args.model}_topo{args.topological}.data",
#     xx_corr,
#     fmt="%10.8f",
#     delimiter=", ",
#     header=f"xx correlator on the GS of the {args.model} model\n parameters: {parameters}")

# with open(f"Times{par.n_qbits}_{args.model}_topo{args.topological}.data", 'w') as file:
#     file.write(json.dumps(times, indent='\t')) # use `json.loads` to do the reverse

# times['IO_handling'] = timer() - start