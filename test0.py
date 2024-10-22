import numpy as np

from block_davidson import block_davidson


n = 256
mat = -1.0 * np.eye(n, k=-1) + 2.0 * np.eye(n) - 1.0 * np.eye(n, k=+1)
mat = 0.5 * (mat + mat.T)

with np.printoptions(threshold=np.inf, linewidth=np.inf):
    eigvals, eigvecs = np.linalg.eigh(mat)
    print(eigvals[0])
    print(eigvecs[:, 0])
    eigvals, eigvecs = block_davidson(mat, k=2, maxiter=1000000)
    print(eigvals[0])
    print(eigvecs[:, 0])
