"""Block Davidson algorithm."""

import numpy as np


def block_davidson(
    matrix: np.ndarray,
    k: int = 1,
    tol: float = 1e-9,
    maxiter: int = 1000,
):
    """Calculate eigenvalues/eigenvectors using the block Davidson algorithm.

    Parameters
    ----------
    matrix : (n, n) np.ndarray
        Matrix whose smallest eigenvalues/eigenvectors are computed.
    k : int, default = 1
        Number of smallest eigenvalues/eigenvectors to be computed.
    tol : float, default = 1e-6
        Convergence tolerance for residual norms.
    maxiter : int, default = 1000
        Maximum number of iterations.

    Returns
    -------
    eigvals : (k,) np.ndarray
        k lowest eigenvalues.
    eigvecs : (n, k) np.ndarray
        Corresponding eigenvectors.

    """
    n = matrix.shape[0]

    # Find indices of smallest diagonal elements
    indices = np.argsort(np.diag(matrix))[: 2 * k]

    # Initial guess of the eigenvectors
    vecs = np.zeros((n, 2 * k))
    vecs[indices, np.arange(2 * k)] = 1.0

    for iteration in range(maxiter):
        print(iteration, end="\r")

        # Project the matrix into the subspace
        projected_matrix = vecs.T @ matrix @ vecs

        # Solve the eigenproblem for the projected matrix
        eigvals, eigvecs_subspace = np.linalg.eigh(projected_matrix)

        # Compute Ritz vectors
        ritz_vecs = vecs @ eigvecs_subspace

        # Compute residuals
        residuals = ritz_vecs * eigvals - matrix @ ritz_vecs

        # Check convergence
        residual_norms = np.linalg.norm(residuals, axis=0)
        if residual_norms[:k].max() < tol:
            print(f"Converged in {iteration} iterations")
            return eigvals[:k], ritz_vecs[:, :k]

        # Update the guess of the eigenvectors
        vecs[:, :k] = ritz_vecs[:, :k]

        # Expand the subspace with preconditioned residual vectors
        q = residuals[:, :k] / (np.diag(matrix)[:, None] - eigvals[:k])
        q = q / np.linalg.norm(q, axis=0)
        vecs[:, k:] = q

        # Reorthogonalize the guess of the eigenvectors
        vecs = np.linalg.qr(vecs)[0]

    raise RuntimeError("Not converged")
