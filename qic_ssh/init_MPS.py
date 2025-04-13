import numpy as np
import scipy as sp
import tenpy
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tenpy.models.model import CouplingMPOModel
from tenpy.algorithms import dmrg
from tenpy.linalg.np_conserved import Array

class XYSystem(CouplingMPOModel):
    def __init__(self, model_params, J):
        self.J = J
        # self.z_param =

        self.eigvals = []
        self.eigvecs = []

        CouplingMPOModel.__init__(self, model_params)

    def init_sites(self, model_params):
        # No idea of what i'm doing, I believe it conserves Sz
        conserve = model_params.get('conserve', None)
        return SpinHalfSite(conserve=conserve)

    def init_terms(self, model_params):
        # Apparently one cannot override init_terms without everything blowing up
        for dx in range(self.J.shape[0]):
            # We basically add each diagonal of the matrix as a coupling
            # between sites at distance dx
            # Save strength entries:
            strength = - np.diagonal(self.J, offset=dx)
            # Create 2-body interaction, already hermitian
            self.add_coupling(strength, 0, 'Sx',
                0, 'Sx', dx, plus_hc=False)
            self.add_coupling(strength, 0, 'Sy',
                0, 'Sy', dx, plus_hc=False)

    def do_DMRG(self, alg_params, p_state=None, k=4):
        if not isinstance(p_state, np.ndarray):
            print("Error: expected useful wavefunction, got None.")
            return

        if len(p_state.shape) == 1:
            # we only have one wf and must add a virtual axis
            print("Number of wavefunctions equal to 1, changing k accordingly to 1")
            k = 1
            p_state = p_state[:, np.newaxis]
        elif k > p_state.shape[1]:
            print("Error: number of eigvecs smaller than number of input wavefunction,")
            print(f"changing k to {p_state.shape[1]}")
            k = p_state.shape[1]

        leg_labels = [f"p{ii}" for ii in range(self.lat.N_sites)]
        train_shape = tuple([2 for _ in range(self.lat.N_sites)])

        psi = []
        # To extend the first k eigenstates, do DMRG k times
        # and each time add the found eigstate to the orthogonal_to parameters
        ortho_space = []
        for ii in range(k):
            wavefunction = p_state[:, ii]
            array_from_wf = Array.from_ndarray_trivial(wavefunction.reshape(train_shape), labels=leg_labels)
            psi.append(MPS.from_full(self.lat.mps_sites(), array_from_wf, bc=self.lat.bc_MPS))
            #Then initialize engine
            eng = dmrg.SingleSiteDMRGEngine(psi[ii], self, alg_params, orthogonal_to=ortho_space)
            # and run
            results = eng.run()
            
            self.eigvals.append(results[0])
            self.eigvecs.append(results[1])
            # then remove the eigenvector that we found
            ortho_space.append(psi[ii])
        
        # finally convert the results into numpy arrays
        self.eigvals = np.array(self.eigvals)
        self.eigvecs = np.array(self.eigvecs)
    
    def z_string(self):
        exp_z_site = Array.from_ndarray_trivial(
            sp.linalg.expm(np.pi * 0.5 * np.array([[1.j, 0.],[0., -1.j]])),
            labels=['wL', 'wR', 'p', 'p*']
        )
        z_site = Array.from_ndarray_trivial(
            np.array([[1., 0.], [0., -1.]]),
            labels=['wL', 'wR', 'p', 'p*']
        )
        operator = MPO(self.lat.mps_sites(),
            ['Id', z_site, *[exp_z_site] * self.lat.N_sites, z_site, 'Id'])
        return self.eigvecs[0].expectation_value(operator)
    
    def x_string(self):
        exp_x_site = Array.from_ndarray_trivial(
            sp.linalg.expm(np.pi * 0.5 * np.array([[0., 1.j], [1.j, 0.]]))
        )
        x_site = Array.from_ndarray_trivial(
            np.array([[0., 1.], [1., 0.]])
        )
        operator = ['Id', x_site, *[exp_x_site] * self.lat.N_sites, x_site, 'Id']        
        return self.eigvecs[0].expectation_value(operator)
