import numpy as np
from scipy import sparse

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
    # Entanglement entropy of the state psi from the sublattice A/B bipartition
    # Compute RDM over the odd sites
    rho_A = get_reduced_density_matrix(psi, 2, par.n_qbits, list(range(0, par.n_qbits, 2)))
    eigvals = np.linalg.eigvalsh(rho_A)
    # Purge the smallest eigenvalues
    eigvals = eigvals[eigvals >= 1e-8]
    # Then Entanglement spectrum
    return  eigvals

def half_chain_entropy(par, psi):
    # Entanglement entropy of the state psi from an half-chain division
    # Compute RDM over first half of the chain
    rho_A = get_reduced_density_matrix(psi, 2, par.n_qbits, list(range(par.n_qbits // 2)))
    eigvals = np.linalg.eigvalsh(rho_A)
    # Purge the smallest eigenvalues
    eigvals = eigvals[eigvals >= 1e-15]
    # Then Entanglement spectrum
    return  eigvals

def entanglement_spectrum(par, eigvecs):
    # Compute the half chain entanglement entropy for all calculated eigenstates
    if len(eigvecs.shape) == 1:
        # we only have one wf
        ent_spectrum = half_chain_entropy(par, eigvecs)
    else:
        ent_spectrum = list()
        for ii in range(eigvecs.shape[1]):
            ent_spectrum.append(half_chain_entropy(par, eigvecs[:, ii]))

    return ent_spectrum

def order_eigvals(par, eigvals, eigvecs):
    # Function that given a COMPLETE eigenspace of the hamiltonian returns
    # the associated number of particles. This implies diagonalizing the subspace
    # and thus scales quite badly.
    # eigspace[:, i] should be the i-th eigvector of the starting basis
    # It basically builds graphs in figures 3A and 3B.

    # Careful that in order for this to work we need the COMPLETE energy eigenstate,
    # as having a partial eigenstate means that we cannot distinguish actual degenerate
    # n_op eigenvalues from superposition of eigenvalues.

    # Initiate empty list of lists to store the particle numbers-energy lists
    fermions_to_energy = [list() for ii in range(par.n_qbits + 1)]
    # Create first the diagonal of the number operator in the computational basis
    occupation_list = np.array([sum(map(int,"{0:b}".format(number)))
        for number in range(par.lin_size)])
    # Then create N_op
    N_op = sparse.dia_array((occupation_list, 0), shape=(par.lin_size, par.lin_size))

    # And find the energy subspaces:
    # creates an array of indices, sorted by unique element
    idx_sort = np.argsort(eigvals)

    # sorts eigvals array so all unique elements are together 
    eigvals = eigvals[idx_sort]

    # returns unique values, index of first occurrence of value, and count for each element
    vals, idx_start, counts = np.unique(
        eigvals.round(decimals=8),
        return_counts=True,
        return_index=True
    )

    # And for all COMPLETE subspaces project and diagonalize
    for (val, idx, count) in zip(vals, idx_start, counts):
        eigspace = eigvecs[:, idx_sort[idx:idx+count]]
        n_fermions = np.linalg.eigvalsh(eigspace.T.conj() @ N_op @ eigspace)

        if (np.abs(np.round(n_fermions) - n_fermions) >= 1e-8).any():
            print("#########################################################################")
            print("Relevantly noninteger number of fermions with max value:")
            print(f"{max(np.round(n_fermions) - n_fermions)}")
            print("#########################################################################")
        
        n_fermions = np.round(n_fermions).astype(int)

        for num in n_fermions:
            fermions_to_energy[num].append(val)
    
    return fermions_to_energy

def zz_correlation(par, psi, keep_indices):
    # To find the 2-point zz correlation expectation value of sites i, j
    # "i" should be an even number and "j" should be i+1 as in the paper

    # if i = j, then the correlation is 1
    if keep_indices[0] == keep_indices[1]:
        return 1.
    
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
    env_dim = 2**(par.n_qbits - 2)
    # Reshape the reordered tensor to separate subsystem from environment
    psi_partitioned = psi_tensor.reshape((subsystem_dim, env_dim))
    # Compute the reduced density matrix by tracing out the env-indices
    RDM = np.tensordot(psi_partitioned, np.conjugate(psi_partitioned), axes=([1], [1]))
    # Reshape rho to ensure it is a square matrix corresponding to the subsystem
    RDM = RDM.reshape((subsystem_dim, subsystem_dim))
    # Now compute the desired two-site operator's expectation value
    return np.trace(RDM @ np.kron(np.array([[1., 0],[0, -1.]]), np.array([[1., 0],[0, -1.]])))

def xx_correlation(par, psi, keep_indices):
    # To find the 2-point zz correlation expectation value of sites i, i+1
    # "i" should be an even number and "j" should be i+1 as in the paper
    
    # if i = j, then the correlation is 1
    if keep_indices[0] == keep_indices[1]:
        return 1.

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
    env_dim = 2**(par.n_qbits - 2)
    # Reshape the reordered tensor to separate subsystem from environment
    psi_partitioned = psi_tensor.reshape((subsystem_dim, env_dim))
    # Compute the reduced density matrix by tracing out the env-indices
    RDM = np.tensordot(psi_partitioned, np.conjugate(psi_partitioned), axes=([1], [1]))
    # Reshape rho to ensure it is a square matrix corresponding to the subsystem
    RDM = RDM.reshape((subsystem_dim, subsystem_dim))
    # Now compute the desired two-site operator's expectation value
    return np.trace(RDM @ np.kron(np.array([[0, 1.],[1., 0]]), np.array([[0, 1.],[1., 0]])))

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
    
    return - psi.T.conj() @ string_op.dot(psi)

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
    
    return - psi.T.conj() @ string_op.dot(psi)

