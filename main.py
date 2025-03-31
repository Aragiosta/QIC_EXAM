import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt
from qic_ssh import init_ED as ED
from qic_ssh import setup

def find_num_fermions(par, eigspace):
    # Function that given a COMPLETE eigenspace of the hamiltonian returns
    # the associated number of particles. This implies diagonalizing the subspace
    # and thus scales quite badly.
    # eigspace[:, i] should be the i-th eigvector of the starting basis

    # Create first the number operator in the computational basis - it's diagonal
    occupation_list = np.array([sum(map(int,"{0:b}".format(number)))
        for number in range(par.lin_size)])
    # Then create and project N_op into the eigenspace
    N_op = sparse.dia_array((occupation_list, 0), shape=(par.lin_size, par.lin_size))
    N_op = eigspace.T.conj() @ N_op @ eigspace
    return np.linalg.eigvalsh(N_op)

def get_reduced_density_matrix(psi, loc_dim, n_sites, keep_indices, print_rho=False):
    """
    Parameters
    ----------
    psi : ndarray
        state of the Quantum Many-Body system
    loc_dim : int
        local dimension of each single site of the QMB system
    n_sites : int
        total number of sites in the QMB system
    keep_indices (list of ints):
        Indices of the lattice sites to keep.
    print_rho : bool, optional
        If True, it prints the obtained reduced density matrix], by default False

    Returns
    -------
    ndarray
        Reduced density matrix
    """
    if not isinstance(psi, np.ndarray):
        raise TypeError(f'density_mat should be an ndarray, not a {type(psi)}')

    if not np.isscalar(loc_dim) and not isinstance(loc_dim, int):
        raise TypeError(f'loc_dim must be an SCALAR & INTEGER, not a {type(loc_dim)}')

    if not np.isscalar(n_sites) and not isinstance(n_sites, int):
        raise TypeError(f'n_sites must be an SCALAR & INTEGER, not a {type(n_sites)}')

    # Ensure psi is reshaped into a tensor with one leg per lattice site
    psi = psi.reshape(*[loc_dim for _ in range(int(n_sites))])
    # Determine the environmental indices
    all_indices = list(range(n_sites))
    env_indices = [i for i in all_indices if i not in keep_indices]
    new_order = np.append(keep_indices, env_indices)
    # Rearrange the tensor to group subsystem and environment indices
    psi_tensor = np.transpose(psi, axes=new_order)
    # print(f"Reordered psi_tensor shape: {psi_tensor.shape}")
    # Determine the dimensions of the subsystem and environment for the bipartition
    subsystem_dim = np.prod([loc_dim for i in keep_indices])
    env_dim = np.prod([loc_dim for i in env_indices])
    # Reshape the reordered tensor to separate subsystem from environment
    psi_partitioned = psi_tensor.reshape((subsystem_dim, env_dim))
    # Compute the reduced density matrix by tracing out the env-indices
    RDM = np.tensordot(psi_partitioned, np.conjugate(psi_partitioned), axes=([1], [1]))
    # Reshape rho to ensure it is a square matrix corresponding to the subsystem
    RDM = RDM.reshape((subsystem_dim, subsystem_dim))

    # PRINT RHO
    if print_rho:
        print('----------------------------------------------------')
        print(f'DENSITY MATRIX TRACING SITES ({str(env_indices)})')
        print('----------------------------------------------------')
        print(RDM)

    return RDM

def bipartite_entropy(par, psi):
    # Entanglement entropy of the state psi
    # Compute RDM over the odd sites
    rho_A = get_reduced_density_matrix(psi, 2, par.n_qbits, [ii for ii in range(0, par.n_qbits, 2)])
    eigvals = np.linalg.eigvalsh(rho_A)
    # Purge the smallest eigenvalues
    eigvals = eigvals[eigvals >= 1e-8]
    # Then Von Neumann's entropy
    return  - np.sum(eigvals * np.log2(eigvals))

def order_eigvals(par, eigvals, eigvecs):
    # Returns a list of lists. Each list is the collection
    # of eigenstates with a specific "number of fermions"
    # It basically builds graphs in figures 3A and 3B.
    energy_per_number = [list() for qbits in range(par.n_qbits)]
    # n_op = sum([
    #     sparse.kron(
    #         sparse.kron(sparse.eye(2**ii), sparse.dia_array(([1, 0], 0), shape=(2, 2))),
    #         sparse.eye(2**(par.n_qbits - ii - 1))
    #     ) for ii in range(par.n_qbits)])
    occupation_list = np.array([sum(map(int,"{0:b}".format(number)))
        for number in range(2**par.n_qbits)])

    for ii in range(eigvecs.shape[1]):
        # The idea is that once we have the COMPLETE eigenspace of the energy operator
        # associated to the energy E we can just - NOPE, does not work

        # Surely there is a way without using the whole eigenspectrum, for now
        # forget about this and just work with the ground state
        if (np.abs(n_fermions - np.round(n_fermions))  >= 1e-10).all():
            print("Error: noninteger number of fermions")
        else: n_fermions = np.round(n_fermions)

        energy_per_number[n_fermions].append(eigvals[ii])

    return energy_per_number

def zz_correlation(par, psi, keep_indices):
    # To find the 2-point zz correlation expectation value of sites i, i+1
    # "i" should be an even number and "j" should be i+1 as in the paper

    # Start by tracing out the unwanted sites
    # Ensure psi is reshaped into a tensor with one leg per lattice site
    psi = psi.reshape(*[2 for _ in range(par.n_qbits)])
    # Determine the environmental indices
    all_indices = list(range(par.n_qbits))
    env_indices = [ii for ii in all_indices if ii not in keep_indices]
    new_order = np.append(keep_indices, env_indices)
    # Rearrange the tensor to group subsystem and environment indices
    psi_tensor = np.transpose(psi, axes=new_order)
    # Determine the dimensions of the subsystem and environment for the bipartition
    subsystem_dim = 4
    env_dim = par.lin_size - subsystem_dim
    # Reshape the reordered tensor to separate subsystem from environment
    psi_partitioned = psi_tensor.reshape((subsystem_dim, env_dim))
    # Compute the reduced density matrix by tracing out the env-indices
    RDM = np.tensordot(psi_partitioned, np.conjugate(psi_partitioned), axes=([1], [1]))
    # Reshape rho to ensure it is a square matrix corresponding to the subsystem
    RDM = RDM.reshape((subsystem_dim, subsystem_dim))
    # Now compute the desired two-site operator's expectation value
    return np.trace(RDM @ np.kron(np.ndarray([[1, 0],[0, -1]]), np.ndarray([[1, 0],[0, -1]])))

def xx_correlation(par, psi, keep_indices):
    # To find the 2-point zz correlation expectation value of sites i, i+1
    # "i" should be an even number and "j" should be i+1 as in the paper

    # Start by tracing out the unwanted sites
    # Ensure psi is reshaped into a tensor with one leg per lattice site
    psi = psi.reshape(*[2 for _ in range(par.n_qbits)])
    # Determine the environmental indices
    all_indices = list(range(par.n_qbits))
    env_indices = [ii for ii in all_indices if ii not in keep_indices]
    new_order = np.append(keep_indices, env_indices)
    # Rearrange the tensor to group subsystem and environment indices
    psi_tensor = np.transpose(psi, axes=new_order)
    # Determine the dimensions of the subsystem and environment for the bipartition
    subsystem_dim = 4
    env_dim = par.lin_size - subsystem_dim
    # Reshape the reordered tensor to separate subsystem from environment
    psi_partitioned = psi_tensor.reshape((subsystem_dim, env_dim))
    # Compute the reduced density matrix by tracing out the env-indices
    RDM = np.tensordot(psi_partitioned, np.conjugate(psi_partitioned), axes=([1], [1]))
    # Reshape rho to ensure it is a square matrix corresponding to the subsystem
    RDM = RDM.reshape((subsystem_dim, subsystem_dim))
    # Now compute the desired two-site operator's expectation value
    return np.trace(RDM @ np.kron(np.ndarray([[0, 1],[1, 0]]), np.ndarray([[0, 1],[1, 0]])))

def z_string(par, psi):
    string_op = 1.
    
    for ii in range(par.n_qbits):
        if ii == 0 or ii == par.n_qbits - 1:
            operator = sparse.eye(2)
        elif ii == 1 or ii == par.n_qbits - 2:
            operator = sparse.dia_array(([1., -1.], [0]), shape=(2, 2))
        else:
            operator = sparse.dia_array(([1.j, -1.j], [0]), shape=(2, 2))
        string_op = sparse.kron(string_op, operator)
    
    return - psi.T.conj() @ string_op @ psi

def x_string(par, psi):
    string_op = 1.
    
    for ii in range(par.n_qbits):
        if ii == 0 or ii == par.n_qbits - 1:
            operator = sparse.eye(2)
        elif ii == 1 or ii == par.n_qbits - 2:
            operator = sparse.dia_array([[0, 1],[1, 0]])
        else:
            operator = sparse.linalg.expm(np.pi * 0.5 * np.array([[0, 1.j],[1.j, 0]]))
        string_op = sparse.kron(string_op, operator)
    
    return - psi.T.conj() @ string_op @ psi

par = setup.Param(10)
# Now try to use it and reproduce the paper
J = setup.paper(par)
# Test run with ideal SSH
# J = setup.ideal(par, topological=False)
# J = setup.ideal(par)

paper = ED.XYSystem(J)
paper.eig()

# energy_list = order_eigvals(par, paper.eigvals, paper.eigvecs)
energy_list = find_num_fermions(par, paper.eigvecs[:, paper.eigvals == max(paper.eigvals)])

# Compute entanglement of GS
ent_entropy = bipartite_entropy(par, paper.eigvecs[:, 0])

# Compute order parameters
z_parameter = z_string(par, paper.eigvecs[:, 0])
x_parameter = x_string(par, paper.eigvecs[:, 0])

plt.plot(np.abs(paper.eigvecs[:, 0])**2)
plt.show()
print(paper.eigvals)