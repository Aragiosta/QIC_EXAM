import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from qic_ssh import init_ED as ED

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
        self.lattice_angle = lattice_angle

def setup_paper(par: Param, topological: bool=True):
    # To define the model, we need the interaction matrix J
    # For starters, we reproduce the experimental results of Leseleuc
    # Interaction matrix with chiral symmetry, nn hopping and no nnn allowed
    # The system is composed of:
    #   - Two parallel lines of sites, at inclination same_lattice_angle wrt B
    #   - sites on these lines, equally spaced
    # We want chiral symmetry so we impose the magic angle
    # And equally spaced sites, the standard distance is normalized to 1
    pos_A = np.array(list(range(par.n_qbits_sub)))
    pos_A = np.array([pos_A * np.cos(par.lattice_angle), pos_A * np.sin(par.lattice_angle)])

    pos_B = pos_A.copy()
    pos_B[0] += np.sin(par.lattice_angle + np.pi / 3.0)
    pos_B[1] += np.cos(par.lattice_angle + np.pi / 3.0)

    if topological:
        pos_A = np.roll(pos_A, -1, 1)
        pos_A[:, -1] = pos_B[:, -1] + pos_A[:, -2] - pos_B[:, -2]

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

            cos_ij = R_ij[0] / r_ij
            J[ii, jj] = (3 * cos_ij**2 - 1) / np.pow(r_ij, 3)
    
    return J

def order_eigvals(par, eigvals, eigvecs):
    # Returns a list of lists. Each list is the collection
    # of eigenstates with a specific "number of fermions"
    # It basically builds graphs in figures 3A and 3B.
    energy_per_number = [list() for qbits in range(par.n_qbits)]
    occupation_number_list = [
        fermions
        for fermions in range(par.n_qbits + 1)
        for reps in range(sp.special.comb(par.n_qbits, fermions, exact=True))
    ]
    for ii in range(eigvecs.shape[1]):
        # Compute expectation value of number operator,
        # should be integer since N commutes with H
        n_fermions = np.dot(occupation_number_list, np.abs(eigvecs[:, ii])**2)
        # Should be an integer, better check it:
        if np.abs(n_fermions - np.round(n_fermions))  >= 1e-10: print("Error: noninteger number of fermions")
        else: n_fermions = int(n_fermions)

        energy_per_number[n_fermions].append(eigvals[ii])

    return energy_per_number

par = Param(8)
# Now try to use it and reproduce the paper
J = setup_paper(par, False)
paper = ED.XYSystem(par.n_qbits, J)
paper.eig()

energy_list = order_eigvals(par, paper.eigvals, paper.eigvecs)

plt.eventplot(energy_list)
plt.show()
print(paper.eigvals)