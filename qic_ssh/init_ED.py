from scipy import sparse
import numpy as np

class XYSystem:
    sigma_x = sparse.coo_array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = sparse.coo_array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = sparse.coo_array([[1, 0], [0, -1]], dtype=complex)
    sigma_p = 0.5 * (sigma_x + 1j * sigma_y)
    sigma_m = 0.5 * (sigma_x - 1j * sigma_y)

    def __init__(self, int_mat):
        N_sites = int_mat.shape[0]

        self.eigvals = None
        self.eigvecs = None

        self.H = sparse.coo_array((2**N_sites, 2**N_sites), dtype=complex)
        for (ii, jj), J in np.ndenumerate(int_mat):
            if J == 0: continue
            if jj == ii: continue

            first_site = np.min([ii, jj])
            dist = np.abs(ii-jj)
            last_site = np.max([ii, jj])

            # first part
            self.H += J * sparse.kron(
                sparse.kron(
                    sparse.kron(
                        sparse.kron(sparse.eye(2**first_site), self.sigma_p),
                        sparse.eye(2**(dist - 1))),
                    self.sigma_m),
                sparse.eye(2**(N_sites - 1 - last_site)))

            # adjoint part
            self.H += J * sparse.kron(
                sparse.kron(
                    sparse.kron(
                        sparse.kron(sparse.eye(2**first_site), self.sigma_m),
                        sparse.eye(2**(dist - 1))),
                    self.sigma_p),
                sparse.eye(2**(N_sites - 1 - last_site)))
        self.H = 0.5 * (self.H + self.H.T.conj())

    def eig(self, k=None, eigvecs=True):
        if k is None:
            # k = self.H.shape[0] - 2
            # k = 10
            k = int(2 * np.log2(self.H.shape[0]) + 1)
        elif k == self.H.shape[0]:
            # Must find all eigvals thus passing to toarray() - bad scaling
            self.eigvals, self.eigvecs = np.linalg.eigh(self.H.toarray())
        elif k < self.H.shape[0] - 1:
            if eigvecs:
                self.eigvals, self.eigvecs = sparse.linalg.eigsh(self.H, k)
            else:
                self.eigvals = sparse.linalg.eigsh(self.H, k, return_eigenvectors=False)
