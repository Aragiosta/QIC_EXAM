import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt
from qic_ssh import init_ED as ED
from qic_ssh import setup
from qic_ssh import utils
# For external bash handling of the sim parameters
import argparse

# Define the parser
parser = argparse.ArgumentParser(description='QIAC exam')

# Declare an argument (`--algo`), saying that the 
# corresponding value should be stored in the `algo` 
# field, and using a default value if the argument 
# isn't given
parser.add_argument('--topological', action="store", dest='topological', default=True)
parser.add_argument('--n_sites', action="store", dest='n_sites', default=0, type=int)
parser.add_argument('--model', action="store", dest='model', default="ideal")

# Now, parse the command line arguments and store the 
# values in the `args` variable
args = parser.parse_args()
# For when debbuging is on Friday
args.model = "ideal"
args.topological = True
args.n_sites = 8

par = setup.Param(args.n_sites)
# Now try to use it and reproduce the paper
if args.model == "ideal":
    J = setup.ideal(par, topological=args.topological)
elif args.model == "paper":
    J = setup.paper(par, topological=args.topological)
else:
    J = np.empty((par.n_qbits, par.n_qbits))

paper = ED.XYSystem(J)
paper.eig(k=par.lin_size)

# energy_list = order_eigvals(par, paper.eigvals, paper.eigvecs)
energy_list = utils.find_num_fermions(par, paper.eigvecs[:, paper.eigvals == max(paper.eigvals)])

# Compute entanglement of GS
ent_entropy = utils.bipartite_entropy(par, paper.eigvecs[:, 0])

# Compute order parameters
z_parameter = utils.z_string(par, paper.eigvecs[:, 0])
x_parameter = utils.x_string(par, paper.eigvecs[:, 0])

# Compute correlation matrices
zz_corr = np.array([
    utils.zz_correlation(par, paper.eigvecs[:, 0], [ii, jj])
    for (ii, jj), J in np.ndenumerate(J)
]).reshape(par.n_qbits, par.n_qbits)

xx_corr = np.array([
    utils.xx_correlation(par, paper.eigvecs[:, 0], [ii, jj])
    for (ii, jj), J in np.ndenumerate(J)
]).reshape(par.n_qbits, par.n_qbits)

e_N = utils.order_eigvals(par, paper.eigvals, paper.eigvecs)
plt.eventplot(e_N, orientation='vertical')
plt.show()
# Data production

# np.savetxt(
#     f"Ham{par.n_qbits}_{args.model}_topo{args.topological}.data",
#     paper.H.toarray(),
#     fmt="%10.8f",
#     delimiter=", ",
#     header=f"Hamiltonian of {par.n_qbits} sites obeying the {args.model} model")

# np.savetxt(
#     f"Eigvals{par.n_qbits}_{args.model}_topo{args.topological}.data",
#     paper.eigvals,
#     fmt="%10.8f",
#     delimiter=", ",
#     header=f"Energies of {par.n_qbits} sites obeying the {args.model} model")

# np.savetxt(
#     f"Eigvecs{par.n_qbits}_{args.model}_topo{args.topological}.data",
#     paper.eigvecs,
#     fmt="%10.8f",
#     delimiter=", ",
#     header=f"Columns are eigstates of {par.n_qbits} sites of the {args.model} model")

# np.savetxt(
#     f"EnergyToNumber{par.n_qbits}_{args.model}_topo{args.topological}.data",
#     energy_list,
#     delimiter=", ",
#     header=f"Number of particles in the GS of {par.n_qbits} sites of the {args.model} model")

# np.savetxt(
#     f"EntEntropy{par.n_qbits}_{args.model}_topo{args.topological}.data",
#     np.array([ent_entropy]),
#     fmt="%10.8f",
#     delimiter=", ",
#     header=f"Bipartite entanglement entropy on the GS of {par.n_qbits} sites of the {args.model} model")

# np.savetxt(
#     f"StringParameters{par.n_qbits}_{args.model}_topo{args.topological}.data",
#     [z_parameter, x_parameter],
#     fmt="%10.8f",
#     delimiter=", ",
#     header=f"z and x string exp. values on the GS of {par.n_qbits} sites of the {args.model} model")

# np.savetxt(
#     f"ZZCorrelators{par.n_qbits}_{args.model}_topo{args.topological}.data",
#     zz_corr,
#     fmt="%10.8f",
#     delimiter=", ",
#     header=f"zz 2-point correlator on the GS of {par.n_qbits} sites of the {args.model} model")

# np.savetxt(
#     f"XXCorrelators{par.n_qbits}_{args.model}_topo{args.topological}.data",
#     xx_corr,
#     fmt="%10.8f",
#     delimiter=", ",
#     header=f"xx 2-point correlator on the GS of {par.n_qbits} sites of the {args.model} model")
