"""Part 2, one seed: K-MPC in the closed loop (stage 2 of the TZ: adaptive K-MPC, scenario S1).

    python run_mpc.py [--seed 0]

Writes outputs/mpc_metrics.json and outputs/fig6_s1.png, fig6_step_time.png.
One BLAS thread (set before numpy is imported): step times and results are reproducible.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import closed_loop as cl  # noqa: E402
import data  # noqa: E402
import mpc  # noqa: E402
import pipeline as pl  # noqa: E402
import safety  # noqa: E402
from run_demo import SCENARIO, STARTUP_SHARE, TC_RANGE  # noqa: E402

OUT = Path(__file__).parent / "outputs"
RHO_PROVISIONAL = 1.0   # move weight in normalised units (r0 = rho / SIGMA_U^2): an a-priori value
                        # (0.01 mol/L of error ~ 1 K of move); replaced by tuning on S0 in stage 3


def commissioning(seed):
    """Start-up exactly as in part 1 (run_demo.simulated_run / stationary): the first 5/8 of the
    stationary log of this seed. Returns the commissioning result and a hash of the start-up block
    (the same block for every controller, P10)."""
    rng, noise_seed = data.streams(seed, SCENARIO["stationary"])
    df = data.make_log(data.aprbs(8000, *TC_RANGE, 10, 60, rng), noise_seed, gap_at=2500)
    n = int(STARTUP_SHARE * len(df))
    y, u = df["Ca_mol_L"].to_numpy()[:n], df["Tc_K"].to_numpy()[:n]
    h = hashlib.sha256(np.ascontiguousarray(np.c_[y, u]).tobytes()).hexdigest()[:16]
    return pl.commission(y, u, seed=seed), h


def run_s1(c, seed, lim, hold_d_on_reject=False, rho=RHO_PROVISIONAL):
    sc = cl.scenario_s1(seed, lim)
    ident = pl.identifier(c, horizon=mpc.NP)
    ctrl = mpc.KMPC(mpc.window_model(ident), lim, rho, hold_d_on_reject=hold_d_on_reject)
    log = cl.run_loop(sc["plant"], sc["x0"], ident, ctrl, sc["r"], sc["add"], sc["missing"], sc["u0"])
    return sc, log, cl.run_metrics(log, sc["starts"], safety.TC_MIN, lim, mpc.DU_MAX)


def figures(sc, log, lim, path_s1, path_time):
    import matplotlib.pyplot as plt
    import figures as fg   # part 1 style
    t = np.arange(len(log["r"])) * data.TS
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1]})
    ax[0].plot(t, log["y_meas"], ".", ms=2.5, color=fg.MUTED, label="measurement")
    ax[0].plot(t, log["Ca"], color=fg.BLUE, lw=1.6, label="true Ca")
    ax[0].plot(t, log["r"], color=fg.INK, lw=1.0, ls="--", label="set-point")
    ax[0].set(title="S1: set-point steps, adaptive K-MPC, nominal plant", ylabel="Ca, mol/L")
    lo, hi = np.nanmin(log["r"]) - 0.02, np.nanmax(log["r"]) + 0.02
    ax[0].set_ylim(lo, hi)
    ax[1].plot(t, log["u"], color=fg.ORANGE, lw=1.4, label="Tc")
    ax[1].axhline(lim, color=fg.INK, lw=1.0, ls=":", label=f"Tc_max = {lim:.2f} K")
    ax[1].axhline(safety.TC_MIN, color=fg.MUTED, lw=1.0, ls=":", label=f"Tc_min = {safety.TC_MIN:.0f} K")
    ax[1].set(ylabel="Tc, K")
    ax[2].plot(t, log["mult"], color=fg.VIOLET, lw=1.2, label="supervisor multiplier m_k")
    ax[2].set(ylabel="m_k", xlabel="time, min", ylim=(0.9, 2.1))
    hs, ls = [], []
    for a in ax:
        h, l_ = a.get_legend_handles_labels(); hs += h; ls += l_
    fig.legend(hs, ls, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path_s1)
    plt.close(fig)
    ms = log["t_ctrl_ms"][~np.isnan(log["t_ctrl_ms"])]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(ms, bins=40, color=fg.BLUE)
    p95 = np.percentile(ms, 95)
    ax.axvline(p95, color=fg.INK, lw=1.2, ls="--", label=f"95th percentile {p95:.1f} ms")
    ax.axvline(60.0, color=fg.ORANGE, lw=1.2, ls=":", label="limit 60 ms")
    ax.set(title="K-MPC step time, one BLAS thread", xlabel="ms", ylabel="steps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_time)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    with threadpool_limits(1):
        lim = safety.tc_max()
        c, block_hash = commissioning(args.seed)
        res = {"seed": args.seed, "startup_block_sha256_16": block_hash,
               "machine": {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
                           "python": platform.python_version(), "blas_threads": 1},
               "settings": {"Np": mpc.NP, "Nc": mpc.NC, "du_max": mpc.DU_MAX, "tc_min": safety.TC_MIN,
                            "tc_max": lim["tc_max"], "sigma_y": mpc.SIGMA_Y, "sigma_u": mpc.SIGMA_U,
                            "rho": RHO_PROVISIONAL, "rho_status": "provisional, a priori (not tuned)",
                            "asymmetric": True, "n_iter": 1, "osqp": mpc.OSQP_SETTINGS,
                            "feas_tol": mpc.FEAS_TOL, "ignition_T": cl.IGNITION_T, "band": cl.BAND},
               "tc_max": {k: lim[k] for k in ("tc_max", "worst_limit", "worst_vertex", "margin",
                                              "ca_nominal_at_tc_max")},
               "commissioning": {"sigma": c["sigma"], "lam": c["lam"]},
               "S1": {}}
        for name, hold in (("adaptive K-MPC", False), ("adaptive K-MPC, d held on rejected samples", True)):
            sc, log, m = run_s1(c, args.seed, lim["tc_max"], hold)
            m["tc_levels"] = sc["tc_levels"].tolist()
            res["S1"][name] = m
            if not hold:
                figures(sc, log, lim["tc_max"], OUT / "fig6_s1.png", OUT / "fig6_step_time.png")
    (OUT / "mpc_metrics.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
