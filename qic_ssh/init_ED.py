import scipy.sparse as sparse
import numpy as np

class XYSystem:
    sigma_x = sparse.coo_array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = sparse.coo_array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = sparse.coo_array([[1, 0], [0, -1]], dtype=complex)
    sigma_p = sigma_x + 1j * sigma_y
    sigma_m = sigma_x - 1j * sigma_y

    def __init__(self, N_sites, int_mat):
        self.H = sparse.coo_array((2**N_sites, 2**N_sites), dtype=complex)
        for (i, j), J in np.ndenumerate(int_mat):
            if i >= j:
                continue

            self.H += J * sparse.kron(
                sparse.kron(
                    sparse.kron(
                        sparse.kron(sparse.eye(2**i), self.sigma_p),
                        sparse.eye(2**(j - i - 1))),
                    self.sigma_m),
                sparse.eye(2**(N_sites - 1 - j)))

        self.H = 0.5 * (self.H + self.H.T.conj())

    def eig(self, k=None, eigvecs=True):
        if k is None:
            # k = self.H.shape[0] - 2
            # k = 10
            k = int(2 * np.log2(self.H.shape[0]) + 1)
        if eigvecs:
            self.eigvals, self.eigvecs = sparse.linalg.eigsh(self.H, k)
        else:
            self.eigvals = sparse.linalg.eigsh(self.H, k, return_eigenvectors=False)
