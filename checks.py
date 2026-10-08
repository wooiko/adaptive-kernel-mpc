"""Numerical self-checks of the closed-form results used in kernel.py."""
import time

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from kernel import KRR, loo_grid_search


def run(Z, tz, sigma, lam, sigmas, lams, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(Z), size=min(150, len(Z)), replace=False)
    Zs, ts = Z[idx], tz[idx]
    m = KRR(sigma, lam).fit(Zs, ts)

    # 1. analytical gradient against central finite differences
    h, worst = 1e-6, 0.0
    for x in Zs[:20]:
        fd = np.array([(m.predict(x + h * e)[0] - m.predict(x - h * e)[0]) / (2 * h)
                       for e in np.eye(len(x))])
        g = m.gradient(x)
        worst = max(worst, float(np.linalg.norm(fd - g) / np.linalg.norm(g)))

    # 2. closed-form leave-one-out residuals against n actual refits
    brute = np.array([ts[i] - KRR(sigma, lam).fit(np.delete(Zs, i, 0), np.delete(ts, i))
                      .predict(Zs[i])[0] for i in range(len(Zs))])
    loo_dev = float(np.max(np.abs(brute - m.loo_residuals())))

    # 3. variance from the KRR factorisation against a library Gaussian process
    gp = GaussianProcessRegressor(RBF(length_scale=sigma), alpha=lam, optimizer=None).fit(Zs, ts)
    Zq = Z[rng.choice(len(Z), size=200, replace=False)]
    var_dev = float(np.max(np.abs(gp.predict(Zq, return_std=True)[1] ** 2 - m.variance(Zq))))

    # 4. cost of the whole hyper-parameter search
    sub = rng.choice(len(Z), size=min(1000, len(Z)), replace=False)
    t0 = time.perf_counter()
    loo_grid_search(Z[sub], tz[sub], sigmas, lams)
    return {"gradient_max_rel_dev": worst, "loo_max_abs_dev": loo_dev,
            "variance_max_abs_dev": var_dev, "loo_grid_seconds": time.perf_counter() - t0,
            "loo_grid_size": int(len(sigmas) * len(lams)), "loo_grid_n": int(len(sub))}
