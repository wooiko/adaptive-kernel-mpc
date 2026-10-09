"""Simulated plant logs and basic data-quality numbers."""
import numpy as np
import pandas as pd

from cstr import simulate, steady_state

TS = 0.1          # sampling period, min
NOISE_STD = 0.002  # sensor noise, mol/L


def streams(seed, scenario):
    """Independent random streams for one experiment: (generator for the input sequence,
    seed for the log's noise and spikes). Different (seed, scenario) pairs never share
    a stream."""
    a, b = np.random.SeedSequence([seed, scenario]).spawn(2)
    return np.random.default_rng(a), b


def aprbs(n, lo, hi, min_hold, max_hold, rng):
    """Amplitude-modulated pseudo-random binary sequence: random levels, random hold times."""
    u = np.empty(n)
    k = 0
    while k < n:
        hold = rng.integers(min_hold, max_hold + 1)
        u[k:k + hold] = rng.uniform(lo, hi)
        k += hold
    return u


def make_log(Tc, seed, anomaly_rate=0.02, k0_scale=None, gap_at=None):
    """Run the reactor on the input sequence Tc and return a 'raw plant log'.

    Added on purpose: Gaussian sensor noise, isolated spikes of 5-25 noise standard
    deviations (a share `anomaly_rate` of the samples) and, optionally, a logger gap.
    Ca_true and is_anomaly are the ground truth; they are used only for scoring.
    seed: anything numpy.random.default_rng accepts (an int or a list of ints)."""
    rng = np.random.default_rng(seed)
    n = len(Tc)
    states = simulate(Tc, TS, steady_state(Tc[0]), k0_scale=k0_scale)
    Ca_true = states[:, 0]
    Ca = Ca_true + rng.normal(0.0, NOISE_STD, n)
    is_anom = np.zeros(n, dtype=bool)
    idx = rng.choice(np.arange(5, n), size=int(anomaly_rate * n), replace=False)
    Ca[idx] += rng.choice([-1, 1], len(idx)) * rng.uniform(5, 25, len(idx)) * NOISE_STD
    is_anom[idx] = True
    if gap_at is not None:
        Ca[gap_at:gap_at + 8] = np.nan
        is_anom[gap_at:gap_at + 8] = False     # a spike lost in the gap cannot be detected
    return pd.DataFrame({"t_min": np.arange(n) * TS, "Tc_K": Tc, "Ca_mol_L": Ca,
                         "Ca_true": Ca_true, "T_true": states[:, 1], "is_anomaly": is_anom})


REQUIRED_COLUMNS = ("t_min", "Tc_K", "Ca_mol_L")


def validate_log(df, min_samples):
    """Checks a supplied plant log before anything is fitted; returns the sampling period
    in minutes, taken from the time column. Raises ValueError with the reason."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df]
    if missing:
        raise ValueError(f"log: missing columns {missing}; required: {list(REQUIRED_COLUMNS)}")
    for c in REQUIRED_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="raise")
    if len(df) < min_samples:
        raise ValueError(f"log: {len(df)} samples; at least {min_samples} are needed "
                         "(start-up block long enough for the window and threshold calibration)")
    if df["t_min"].isna().any() or df["Tc_K"].isna().any():
        raise ValueError("log: missing values in t_min or Tc_K (only Ca_mol_L may have gaps; "
                         "a missing input cannot be replaced by a prediction)")
    dt = np.diff(df["t_min"].to_numpy())
    ts = float(np.median(dt))
    if ts <= 0 or np.max(np.abs(dt - ts)) > 0.01 * ts:
        raise ValueError(f"log: sampling is not uniform (median step {ts:g} min, steps from "
                         f"{dt.min():g} to {dt.max():g} min); resample it or mark lost samples "
                         "as empty Ca_mol_L on a uniform time grid")
    if np.ptp(df["Tc_K"].to_numpy()) == 0:
        raise ValueError("log: Tc_K is constant; the input must vary for identification")
    if df["Ca_mol_L"].notna().sum() < min_samples:
        raise ValueError(f"log: {int(df['Ca_mol_L'].notna().sum())} measured values of Ca_mol_L; "
                         f"at least {min_samples} are needed")
    return ts


def noise_std_estimate(y, k=4.0, iters=2):
    """Sensor noise from second differences: for white noise of std s on a smooth
    signal, std(y[k] - 2 y[k-1] + y[k-2]) = sqrt(6) s. In the reference run the
    noise-free signal's second difference has an RMS of 4 % of sqrt(6) s, which adds
    under 0.1 % in quadrature. Scale: median absolute
    deviation. A spike enters three consecutive second differences, so 2 % of spikes
    contaminate 6 % of them and inflate the plain MAD by about 7.6 %
    (Phi^-1(0.5 + 0.25 / 0.94) / Phi^-1(0.75)). Therefore second differences farther
    than k robust standard deviations from the median, together with their two
    neighbours, are removed and the MAD is recomputed (`iters` times). For Gaussian
    noise the trimming at k = 4 removes about 2e-4 of the values."""
    d2 = np.diff(y[~np.isnan(y)], 2)
    keep = np.ones(len(d2), bool)
    for _ in range(iters + 1):
        m = np.median(d2[keep])
        s = 1.4826 * np.median(np.abs(d2[keep] - m))
        out = np.abs(d2 - m) > k * s
        keep = ~(out | np.r_[out[1:], False] | np.r_[False, out[:-1]])
    return float(s / np.sqrt(6))


MEDIAN_WINDOW = 7  # reference detector: rolling-median window, samples


def rolling_median_dev(y, window=MEDIAN_WINDOW, causal=True):
    """Distance from a short rolling median. causal=True uses the current and the
    window-1 previous samples (usable online); causal=False is centred and needs
    (window-1)/2 future samples, i.e. a decision delay of 3 samples for window 7."""
    med = pd.Series(y).rolling(window, center=not causal, min_periods=1).median().to_numpy()
    return np.abs(y - med)


def spike_mask_offline(y, noise_std, k=4.0, window=MEDIAN_WINDOW):
    """Offline, non-causal spike mask for historical data (used only to choose clean
    rows when calibrating thresholds): centred rolling median, k noise std."""
    return rolling_median_dev(y, window, causal=False) > k * noise_std


def describe(df, y="Ca_mol_L", u="Tc_K"):
    uu = df[u].to_numpy()
    hist, _ = np.histogram(uu, bins=10)
    return {"n_samples": int(len(df)), "n_missing": int(df[y].isna().sum()),
            "noise_std_est": noise_std_estimate(df[y].to_numpy()),
            "input_min": float(uu.min()), "input_max": float(uu.max()),
            "input_levels": int(np.sum(np.diff(uu) != 0) + 1),
            "input_emptiest_decile_share": float(hist.min() / len(uu))}
