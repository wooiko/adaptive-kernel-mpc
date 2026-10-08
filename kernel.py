"""Kernel ridge regression with an RBF kernel, written out in NumPy.

One Gram-matrix factorisation gives four things:
  * the model coefficients            alpha = (K + lam I)^-1 y
  * the exact leave-one-out error     e_i = alpha_i / [(K + lam I)^-1]_ii
  * the analytical gradient of the prediction with respect to its inputs
  * the Gaussian-process posterior variance for the same kernel
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh


def rbf(A, B, sigma):
    """k(a, b) = exp(-|a - b|^2 / (2 sigma^2)) for all rows of A against all rows of B."""
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.exp(-np.maximum(d2, 0.0) / (2.0 * sigma ** 2))


class KRR:
    def __init__(self, sigma, lam):
        self.sigma, self.lam = float(sigma), float(lam)

    def fit(self, X, y):
        self.X = np.asarray(X, dtype=float)
        K = rbf(self.X, self.X, self.sigma)
        self._chol = cho_factor(K + self.lam * np.eye(len(K)), lower=True)
        self.alpha = cho_solve(self._chol, y)
        return self

    def predict(self, Xs):
        return rbf(np.atleast_2d(Xs), self.X, self.sigma) @ self.alpha

    def gradient(self, x):
        """d f / d x at one point, in closed form: O(n d), no finite differences."""
        k = rbf(x[None, :], self.X, self.sigma)[0]
        return -((k * self.alpha) @ (x[None, :] - self.X)) / self.sigma ** 2

    def variance(self, Xs):
        """GP posterior variance with the same kernel and the same factorisation:
        var(x) = k(x, x) - k(x)^T (K + lam I)^-1 k(x), where k(x, x) = 1 for RBF.
        Small where training data are dense, close to 1 far away from them."""
        ks = rbf(np.atleast_2d(Xs), self.X, self.sigma)
        return 1.0 - np.einsum("ij,ji->i", ks, cho_solve(self._chol, ks.T))

    def loo_residuals(self):
        """Exact leave-one-out residuals without refitting n times."""
        inv_diag = np.diag(cho_solve(self._chol, np.eye(len(self.X))))
        return self.alpha / inv_diag


def loo_grid_search(X, y, sigmas, lams):
    """Pick (sigma, lam) by minimum leave-one-out RMSE. For each sigma the Gram
    matrix is eigen-decomposed once; every lam then costs only O(n^2)."""
    table = np.empty((len(sigmas), len(lams)))
    for i, s in enumerate(sigmas):
        w, V = eigh(rbf(X, X, s))
        Vty, V2 = V.T @ y, V ** 2
        for j, lam in enumerate(lams):
            alpha = V @ (Vty / (w + lam))
            inv_diag = V2 @ (1.0 / (w + lam))
            table[i, j] = np.sqrt(np.mean((alpha / inv_diag) ** 2))
    i, j = np.unravel_index(np.argmin(table), table.shape)
    return float(sigmas[i]), float(lams[j]), table
