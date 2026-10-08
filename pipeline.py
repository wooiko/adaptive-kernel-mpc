"""Building blocks of the demo: commissioning, tuning, scoring."""
import numpy as np
from sklearn.linear_model import RidgeCV

import data
from adaptive import (LAG, AdaptiveIdentifier, Scaler, regression_set, regressor, run_online)
from kernel import KRR, loo_grid_search

SIGMAS = np.logspace(-0.3, 1.5, 10)   # kernel width, in normalised units
LAMS = np.logspace(-5, -1, 9)         # regularisation
DEFAULT_SIGMA, DEFAULT_LAM = 3.0, 1e-2
WINDOW = 1000        # sliding-window length, samples
REFIT_EVERY = 20     # accepted samples between refits of the window model
KAPPA = 3.5          # anomaly threshold, in standard deviations of the one-step residual


def tune(Z, tz, n_max=1000, seed=0, tol=0.02):
    """Leave-one-out grid search on a random subset of the rows.
    The LOO surface is flat near its minimum, so among all settings within `tol` of
    the best one we take the strongest regularisation and then the narrowest kernel:
    the smoothest model whose uncertainty still grows quickly away from the data."""
    idx = np.random.default_rng(seed).choice(len(Z), size=min(n_max, len(Z)), replace=False)
    _, _, table = loo_grid_search(Z[idx], tz[idx], SIGMAS, LAMS)
    ok = table <= table.min() * (1.0 + tol)
    j = max(np.where(ok.any(0))[0])
    i = min(np.where(ok[:, j])[0])
    return float(SIGMAS[i]), float(LAMS[j]), table, idx


def commission(y, u, kappa=KAPPA, seed=0):
    """Start-up on a block of historical data.
    1. Bootstrap: scaler, SVR filter and a first KRR from a random subset of raw rows
       (default kernel settings; the SVR loss is robust to the spikes still in there).
    2. Pass the block through the filter once to clean it.
    3. Tune (sigma, lam) by leave-one-out on the cleaned data.
    Returns everything needed to continue online."""
    noise = data.noise_std_estimate(y)
    X, t, _ = regression_set(y, u)
    sc = Scaler().fit(X, t)
    idx = np.sort(np.random.default_rng(seed).choice(len(X), size=min(WINDOW, len(X)), replace=False))
    boot = AdaptiveIdentifier(DEFAULT_SIGMA, DEFAULT_LAM, sc, sc.x(X[idx]), sc.t(t[idx]), noise,
                              window=WINDOW, kappa=kappa, adapt=False)
    y_init = np.where(np.isnan(y[:LAG]), np.nanmedian(y), y[:LAG])
    first = run_online(boot, y, u, y_init)
    Xc, tc, _ = regression_set(first["y_clean"], u)
    Zc, tzc = sc.x(Xc), sc.t(tc)
    sigma, lam, table, idx = tune(Zc, tzc, seed=seed)
    return {"scaler": sc, "noise": noise, "sigma": sigma, "lam": lam, "loo_table": table,
            "Z": Zc, "tz": tzc, "tune_idx": idx, "y_clean": first["y_clean"],
            "flagged": first["flagged"], "Xc": Xc, "tc": tc}


def identifier(c, **kw):
    kw = {"window": WINDOW, "refit_every": REFIT_EVERY, "kappa": KAPPA, **kw}
    return AdaptiveIdentifier(c["sigma"], c["lam"], c["scaler"], c["Z"], c["tz"], c["noise"], **kw)


class StaticKRR:
    """KRR fitted once on the tuning subset, in physical units."""
    def __init__(self, c):
        self.sc = c["scaler"]
        self.m = KRR(c["sigma"], c["lam"]).fit(c["Z"][c["tune_idx"]], c["tz"][c["tune_idx"]])

    def predict(self, X): return self.sc.t_inv(self.m.predict(self.sc.x(np.atleast_2d(X))))
    def variance(self, X): return self.m.variance(self.sc.x(np.atleast_2d(X)))


def fit_arx(c):
    """Linear reference model on the same regressors and the same cleaned data."""
    return RidgeCV(alphas=np.logspace(-8, 0, 17)).fit(c["Xc"], c["tc"])


def free_run(model, y_init, u):
    """Multi-step simulation: the model is fed its OWN past predictions, never the
    measured output. This is what a controller or a what-if study relies on."""
    y_hat = np.empty(len(u))
    y_hat[:LAG] = y_init[:LAG]
    for k in range(LAG - 1, len(u) - 1):
        y_hat[k + 1] = model.predict(regressor(y_hat, u, k)[None, :])[0]
    return y_hat


def rmse(a, b, mask=None):
    e = (a - b) if mask is None else (a - b)[mask]
    return float(np.sqrt(np.nanmean(e ** 2)))


def fit_percent(y, y_hat):
    return float(100.0 * (1.0 - np.linalg.norm(y - y_hat) / np.linalg.norm(y - y.mean())))


def precision_recall(flag, truth):
    tp = int(np.sum(flag & truth))
    return {"flagged": int(flag.sum()), "true_anomalies": int(truth.sum()), "caught": tp,
            "precision": tp / max(int(flag.sum()), 1), "recall": tp / max(int(truth.sum()), 1)}
