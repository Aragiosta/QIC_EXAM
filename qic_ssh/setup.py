import numpy as np

class Param:
    """
    Container for holding all simulation parameters

    Parameters
    ----------
    Nqbits: int,
        Total number of qbits of the system
    lattice_angle: float,
        angle (in radians) in between the sublattices
        and the magnetic field. Default set to the magic angle
        arccos(1/sqrt(3))
    """
    def __init__(self,
                n_qbits: int,
                lattice_angle: float=np.arccos(1/np.sqrt(3))) -> None:

        self.n_qbits = n_qbits
        self.n_qbits_sub = int(n_qbits / 2)
        self.lin_size = 2**n_qbits
        self.lattice_angle = lattice_angle

def paper(par: Param, topological: bool=True):
    # To define the model, we need the interaction matrix J
    # For starters, we reproduce the experimental results of Leseleuc
    # Interaction matrix with chiral symmetry, nn hopping and no nnn allowed
    # The system is composed of:
    #   - Two parallel lines of sites, at inclination same_lattice_angle wrt B
    #   - sites on these lines, equally spaced
    # We want chiral symmetry so we impose the magic angle
    # And equally spaced sites, the standard distance is normalized to 1
    pos_A = np.array(list(range(par.n_qbits_sub)))
    pos_A = np.array([pos_A * np.sin(par.lattice_angle), pos_A * np.cos(par.lattice_angle)])

    pos_B = pos_A.copy()
    pos_B[0] += np.sin(par.lattice_angle + np.arctan(4. / 3.)) * 10. / 12.
    pos_B[1] += np.cos(par.lattice_angle + np.arctan(4. / 3.)) * 10. / 12.

    if not topological:
        pos_A[:, 0] = pos_A[:, -1] + pos_B[:, -1] - pos_B[:, -2]
        pos_A = np.roll(pos_A, -1, 1)

    J = np.empty((par.n_qbits, par.n_qbits))
    for ii in range(par.n_qbits):
        for jj in range(par.n_qbits):
            R_ij = np.empty((2))
            if ii % 2 == 0:
                if jj % 2 == 0:
                    R_ij = pos_A[:, int(ii / 2)] - pos_A[:, int(jj / 2)]
                if jj % 2 == 1:
                    R_ij = pos_A[:, int(ii / 2)] - pos_B[:, int((jj - 1) / 2)]
            if ii % 2 == 1:
                if jj % 2 == 0:
                    R_ij = pos_B[:, int((ii - 1) / 2)] - pos_A[:, int(jj / 2)]
                if jj % 2 == 1:
                    R_ij = pos_B[:, int((ii - 1) / 2)] - pos_B[:, int((jj - 1) / 2)]
            
            r_ij = np.linalg.norm(R_ij)
            if r_ij == 0:
                J[ii, jj] = 0
                continue

            cos_ij = R_ij[1] / r_ij
            J[ii, jj] = (3 * cos_ij**2 - 1) / np.pow(r_ij, 3)
    
    return J

def ideal(par: Param, topological: bool=True):
    # We want to recreate the ideal SSH model with spins.
    # To this end, we fix the two hopping amplitudes J,J' to the same values
    # used in the paper, although we neglect nnn contributions.
    J_hopping = 2.42
    J_prime_hopping = -0.92
    # J_hopping = 0.1
    # J_prime_hopping = 10.

    if topological:
        # Just switch the hopping amplitudes
        J_hopping, J_prime_hopping = [J_prime_hopping, J_hopping]
    
    J = np.zeros((par.n_qbits, par.n_qbits))
    for ii in range(par.n_qbits - 1):
        if ii % 2 == 0: J[ii, ii + 1] = J_hopping
        else: J[ii, ii + 1] = J_prime_hopping
    
    # Apply PBC
    # J[0, -1] = J_prime_hopping

    J = J + J.T

    return J