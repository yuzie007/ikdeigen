from timeit import timeit

import numpy as np
from scipy.linalg import polar, qr

n = 256
mat = np.diag(np.arange(1, n + 1, dtype=float))
sparcity = 1e-3
rng = np.random.default_rng(42)
mat += sparcity * rng.normal(size=(n, n))
mat = 0.5 * (mat + mat.T)

print(timeit(lambda: np.linalg.qr(mat), number=10))
print(timeit(lambda: qr(mat), number=10))
print(timeit(lambda: polar(mat), number=10))
