import numpy as np

from block_davidson import block_davidson


n = 256
mat = np.diag(np.arange(1, n + 1, dtype=float))
sparcity = 1e-3
rng = np.random.default_rng(42)
mat += sparcity * rng.normal(size=(n, n))
mat = 0.5 * (mat + mat.T)

with np.printoptions(threshold=np.inf, linewidth=np.inf):
    eigvals, eigvecs = np.linalg.eigh(mat)
    print(eigvals[0])
    print(eigvecs[:, 0])
    eigvals, eigvecs = block_davidson(mat, k=2)
    print(eigvals[0])
    print(eigvecs[:, 0])
