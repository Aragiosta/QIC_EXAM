import numpy as np
import tenpy
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import MPS
from tenpy.models.model import CouplingMPOModel
from tenpy.algorithms import dmrg

class XYSystem(CouplingMPOModel):
    def init_sites(self, model_params):
        # No idea of what i'm doing, I believe it conserves Sz
        conserve = 'Sz'
        return SpinHalfSite(conserve=conserve)

    def init_terms(self, model_params):
        for (ii, jj), J in np.ndenumerate(model_params.get('J')):
            if J == 0: continue
            if jj == ii: continue

            first_site = np.min([ii, jj])
            dist = np.abs(ii-jj)
            last_site = np.max([ii, jj])

            # Create 2-body interaction
            self.add_coupling(J / 2, first_site, 'Sp',
                last_site, 'Sm', dist, plus_hc=True)
    
    def do_DMRG(self, alg_params):
        # Missing initial computation of starting psi
        psi = MPS.from_product_state(self.lat.mps_sites(), p_state, bc=self.lat.bc_MPS)
        alg_params = {
            'trunc_params': {
                'chi_max': 30,
                'svd_min': 1.e-7,
            },
            'max_sweeps': 40,
        }
        eng = dmrg.TwoSiteDMRGEngine(psi, self, alg_params)
        self.gs_energy, self.gs_psi = eng.run()