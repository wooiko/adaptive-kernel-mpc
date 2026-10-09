"""Numerical self-checks of the closed-form results used in kernel.py and cstr.py."""
import time

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import cstr
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
    # 2b. the eigen-decomposition shortcut used for tuning against the same residuals
    i, j = int(np.argmin(np.abs(np.asarray(sigmas) - sigma))), int(np.argmin(np.abs(np.asarray(lams) - lam)))
    grid = loo_grid_search(Zs, ts, [sigmas[i]], [lams[j]])[2][0, 0]
    loo_grid_rel_dev = float(abs(grid - np.sqrt(np.mean(m.loo_residuals() ** 2))) / grid)

    # 3. variance from the KRR factorisation against a library Gaussian process
    gp = GaussianProcessRegressor(RBF(length_scale=sigma), alpha=lam, optimizer=None).fit(Zs, ts)
    Zq = Z[rng.choice(len(Z), size=200, replace=False)]
    var_dev = float(np.max(np.abs(gp.predict(Zq, return_std=True)[1] ** 2 - m.variance(Zq))))

    # 4. cost of the whole hyper-parameter search
    sub = rng.choice(len(Z), size=min(1000, len(Z)), replace=False)
    t0 = time.perf_counter()
    loo_grid_search(Z[sub], tz[sub], sigmas, lams)
    return {"gradient_max_rel_dev": worst, "loo_max_abs_dev": loo_dev, "loo_grid_rel_dev": loo_grid_rel_dev,
            "variance_max_abs_dev": var_dev, "loo_grid_seconds": time.perf_counter() - t0,
            "loo_grid_size": int(len(sigmas) * len(lams)), "loo_grid_n": int(len(sub))}


def gradient_physical(model, X, seed=0, n=20):
    """Gradient in physical units (pipeline.StaticKRR.gradient) against central
    differences of the physical-unit prediction; largest relative deviation."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    for x in X[rng.choice(len(X), size=n, replace=False)]:
        h = 1e-6 * np.maximum(np.abs(x), 1.0)
        fd = np.array([(model.predict(x + hj * e)[0] - model.predict(x - hj * e)[0]) / (2 * hj)
                       for hj, e in zip(h, np.eye(len(x)))])
        g = model.gradient(x)
        worst = max(worst, float(np.linalg.norm(fd - g) / np.linalg.norm(g)))
    return worst


def saddle_node(p=cstr.CSTRParams(), h=1e-4):
    """The stability limit from cstr.saddle_node satisfies both conditions of a saddle-node
    point of the energy balance f(T, Tc) (Ca eliminated): f = 0 and df/dT = 0 (central
    differences). Returns both residuals, in K/min."""
    Tc, _, T = cstr.saddle_node(p)
    f = lambda t: cstr._T_balance(t, Tc, p)[0]
    return {"f": float(abs(f(T))), "df_dT": float(abs((f(T + h) - f(T - h)) / (2 * h)))}
