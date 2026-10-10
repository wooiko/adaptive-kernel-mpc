"""Part 2, one seed: every controller of TZ section 7 on every scenario of section 8.

    python run_mpc.py --tune      # S0 only: choose rho, beta (all MPCs) and tau_c (PID); writes outputs/mpc_tuning.json
    python run_mpc.py [--seed 0]  # S1-S6 with the frozen settings; writes outputs/mpc_metrics.json and figures

Two worker processes, one BLAS thread each (set before numpy is imported): results are reproducible;
only the time fields change between runs.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

import closed_loop as cl  # noqa: E402
import controllers as ctl  # noqa: E402
import data  # noqa: E402
import mpc  # noqa: E402
import pipeline as pl  # noqa: E402
import safety  # noqa: E402
from run_demo import SCENARIO, STARTUP_SHARE, TC_RANGE  # noqa: E402

OUT = Path(__file__).parent / "outputs"
TUNING_FILE = OUT / "mpc_tuning.json"
REFERENCE_SEED = 0
WORKERS = 2
RHOS = (0.3, 1.0, 3.0, 10.0, 30.0)        # move weight, normalised (r0 = rho / SIGMA_U^2)
BETAS = (0.03, 0.1, 0.3, 1.0)             # gain of the disturbance filter (1 = no filter)
TAU_C_FACTORS = (1, 2, 4, 8, 16)          # SIMC closed-loop time constant, multiples of the effective dead time
NP_SENSITIVITY = (10, 20, 40)             # TZ decision R4: reported, Np stays 20
TUNE_TOL = 0.05                           # selection on S0: settings within 5 % of the lowest IAE are candidates,
                                          # among them the smallest total variation of Tc (the smoothest input)

# name: (controller kind, identifier adapts (else frozen at commissioning), supervisor multiplier used)
CONFIGS = {
    "PID": ("pid", False, False),
    "linear MPC (OE)": ("lmpc", False, False),
    "K-MPC static": ("kmpc_static", False, False),
    "K-MPC adaptive, no supervisor": ("kmpc", True, False),
    "K-MPC adaptive": ("kmpc", True, True),
    "NMPC on the same KRR": ("nmpc", True, True),
    "K-MPC adaptive, symmetric penalty": ("kmpc_sym", True, True),
    "K-MPC adaptive, no filter": ("kmpc_nofilter", True, True),
    "K-MPC adaptive, reference without spikes": ("kmpc", True, True),
    "NMPC on the true model (oracle)": ("true", True, False),
}
STANDARD = ("PID", "linear MPC (OE)", "K-MPC static", "K-MPC adaptive, no supervisor", "K-MPC adaptive",
            "NMPC on the same KRR")
PLAN = {"S1": STANDARD, "S2": STANDARD, "S3": STANDARD, "S6": STANDARD, "S5b": STANDARD,
        "S4": STANDARD + ("K-MPC adaptive, no filter", "K-MPC adaptive, reference without spikes"),
        "S5a": STANDARD + ("K-MPC adaptive, symmetric penalty",)}
ORACLE_SCENARIOS = ("S1", "S5a")           # oracle NMPC: reference seed only (TZ 7)


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


def make(name, c, lim, s, Np=mpc.NP):
    """Controller and identifier of one configuration. s: frozen settings (rho, beta, PID)."""
    kind, adapt, _ = CONFIGS[name]
    ident = pl.identifier(c, horizon=Np, adapt=adapt, use_filter=(kind != "kmpc_nofilter"))
    if kind == "pid":
        ctrl = ctl.PID(s["pid"]["Kc"], s["pid"]["tau_i"], lim)
    elif kind == "lmpc":
        ctrl = ctl.linear_mpc(pl.LinearOE(c), lim, s["rho"], s["beta"], Np=Np)
    elif kind == "kmpc_static":
        ctrl = mpc.KMPC(mpc.static_model(pl.StaticKRR(c)), lim, s["rho"], s["beta"], Np=Np)
    elif kind == "nmpc":
        ctrl = ctl.NMPC(mpc.window_model(ident), lim, s["rho"], s["beta"], Np=Np)
    elif kind == "true":
        ctrl = ctl.TrueNMPC(lim, s["rho"], Np=Np)
    else:
        ctrl = mpc.KMPC(mpc.window_model(ident), lim, s["rho"], s["beta"], asymmetric=(kind != "kmpc_sym"), Np=Np)
    return ctrl, ident


def job(args):
    """One closed-loop run: (commissioning, seed, Tc_max, scenario, configuration, settings, Np, keep_log)."""
    c, seed, lim, scen, name, s, Np, keep = args
    with threadpool_limits(1):
        sc = (cl.scenario_s4(seed, lim, spikes=False) if name == "K-MPC adaptive, reference without spikes"
              else cl.SCENARIOS[scen](seed, lim))
        ctrl, ident = make(name, c, lim, s, Np)
        t0 = time.perf_counter()
        log = cl.run_loop(sc["plant"], sc["x0"], ident, ctrl, sc["r"], sc["add"], sc["missing"], sc["u0"],
                          supervisor=CONFIGS[name][2])
        wall = time.perf_counter() - t0
        m = cl.run_metrics(log, sc["starts"], safety.TC_MIN, lim, mpc.DU_MAX)
        m["wall_s"] = wall
        m.update(scenario_extras(scen, sc, log, m))
        keep_keys = ("Ca", "T", "u", "r", "y_meas", "mult", "regime", "flagged_rt", "fallback", "t_ctrl_ms")
        return scen, name, m, ({k: np.asarray(log[k]).tolist() for k in keep_keys} if keep else None)


def scenario_extras(scen, sc, log, m):
    """Scenario-specific numbers (TZ 8)."""
    seg = m["segments"]
    if scen == "S2":
        return {"ss_error_max_abs_hold": max(abs(seg[i]["ss_error"]) for i in sc["hold_segments"])}
    if scen == "S4":
        fr = log["flagged_rt"] & ~sc["spike"] & ~sc["missing"]
        return {"false_rejections": int(fr.sum()), "restored_samples": int(log["restored"].sum()),
                "spikes_caught": int(np.sum(log["flagged_rt"] & sc["spike"])), "spikes": int(sc["spike"].sum())}
    if scen == "S6":
        return {"settling_after_return_min": [seg[1]["settling_min"], seg[3]["settling_min"]]}
    return {}


def s4_moves(results, logs):
    """Inputs moved by spikes: deviation of u from the run without spikes (same noise, gap and set-points)."""
    ref = logs.get(("S4", "K-MPC adaptive, reference without spikes"))
    if ref is None:
        return
    for name in ("K-MPC adaptive", "K-MPC adaptive, no filter"):
        lg = logs.get(("S4", name))
        if lg is None:
            continue
        du = np.abs(np.asarray(lg["u"]) - np.asarray(ref["u"]))[cl.LAG:]
        results["S4"][name]["u_dev_from_no_spike_run"] = {"sum_K": float(du.sum()), "max_K": float(du.max()),
                                                          "samples_over_0.1K": int(np.sum(du > 0.1))}


# ---------------------------------------------------------------- tuning on S0

def select(rows):
    best = min(r["iae"] for r in rows)
    cand = [r for r in rows if r["iae"] <= best * (1 + TUNE_TOL) and r.get("violations", 0) == 0]
    return min(cand, key=lambda r: r["tv"])


def tune(c, lim):
    """S0 only (P3): rho and beta for all MPCs on the adaptive K-MPC; tau_c of the SIMC PI; Np sensitivity."""
    fop = ctl.fit_fopdt(c["y_clean"], c["u"])
    base = {"rho": 1.0, "beta": 1.0}
    mpc_jobs = [(c, REFERENCE_SEED, lim, "S0", "K-MPC adaptive", {**base, "rho": r, "beta": b}, mpc.NP, False)
                for r in RHOS for b in BETAS]
    pid = []
    for f in TAU_C_FACTORS:
        Kc, ti = ctl.simc_pi(fop["K"], fop["tau"], fop["theta"], f * fop["theta"])
        pid.append({"tau_c": f * fop["theta"], "Kc": Kc, "tau_i": ti})
    pid_jobs = [(c, REFERENCE_SEED, lim, "S0", "PID", {**base, "pid": p}, mpc.NP, False) for p in pid]
    with ProcessPoolExecutor(WORKERS) as ex:
        out = list(ex.map(job, mpc_jobs + pid_jobs))
    grid = [{"rho": j[5]["rho"], "beta": j[5]["beta"], "iae": o[2]["iae_total"], "tv": o[2]["tc_total_variation"],
             "violations": o[2]["violations"]} for j, o in zip(mpc_jobs, out)]
    pid_res = [{**p, "iae": o[2]["iae_total"], "tv": o[2]["tc_total_variation"], "violations": o[2]["violations"]}
               for p, o in zip(pid, out[len(mpc_jobs):])]
    m, p = select(grid), select(pid_res)
    s = {"rho": m["rho"], "beta": m["beta"], "pid": {k: p[k] for k in ("tau_c", "Kc", "tau_i")}}
    with ProcessPoolExecutor(WORKERS) as ex:
        np_out = list(ex.map(job, [(c, REFERENCE_SEED, lim, "S0", "K-MPC adaptive", s, n, False)
                                   for n in NP_SENSITIVITY]))
    np_res = [{"Np": n, "iae": o[2]["iae_total"], "tv": o[2]["tc_total_variation"], "step_ms": o[2]["step_ms"]}
              for n, o in zip(NP_SENSITIVITY, np_out)]
    return {"seed": REFERENCE_SEED, "scenario": "S0",
            "rule": f"lowest TV of Tc among settings with IAE within {TUNE_TOL:.0%} of the lowest IAE, no violations",
            "fopdt": fop, "mpc_grid": grid, "pid_grid": pid_res, "np_sensitivity": np_res, "chosen": s}


# ---------------------------------------------------------------- figure

def s1_figure(lg, lim, path):
    import matplotlib.pyplot as plt
    import figures as fg   # part 1 style
    t = np.arange(len(lg["r"])) * data.TS
    fig, ax = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1]})
    ax[0].plot(t, lg["y_meas"], ".", ms=2.5, color=fg.MUTED, label="measurement")
    ax[0].plot(t, lg["Ca"], color=fg.BLUE, lw=1.6, label="true Ca")
    ax[0].plot(t, lg["r"], color=fg.INK, lw=1.0, ls="--", label="set-point")
    ax[0].set(title="S1: set-point steps, adaptive K-MPC, nominal plant", ylabel="Ca, mol/L")
    ax[0].set_ylim(np.min(lg["r"]) - 0.02, np.max(lg["r"]) + 0.02)
    ax[1].plot(t, lg["u"], color=fg.ORANGE, lw=1.4, label="Tc")
    ax[1].axhline(lim, color=fg.INK, lw=1.0, ls=":", label=f"Tc_max = {lim:.2f} K")
    ax[1].axhline(safety.TC_MIN, color=fg.MUTED, lw=1.0, ls=":", label=f"Tc_min = {safety.TC_MIN:.0f} K")
    ax[1].set(ylabel="Tc, K")
    ax[2].plot(t, lg["mult"], color=fg.VIOLET, lw=1.2, label="supervisor multiplier m_k")
    ax[2].set(ylabel="m_k", xlabel="time, min", ylim=(0.9, 2.1))
    hs, ls = [], []
    for a in ax:
        h, l_ = a.get_legend_handles_labels(); hs += h; ls += l_
    fig.legend(hs, ls, loc="lower center", ncol=4)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def step_time_figure(logs, path):
    """Step time of K-MPC (adaptive) and NMPC on the same KRR in S1, one BLAS thread."""
    import matplotlib.pyplot as plt
    import figures as fg
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.logspace(0, 2.3, 50)
    for name, col in (("K-MPC adaptive", fg.BLUE), ("NMPC on the same KRR", fg.ORANGE)):
        ms = np.asarray(logs[("S1", name)]["t_ctrl_ms"], float)
        ms = ms[np.isfinite(ms)]
        ax.hist(ms, bins=bins, color=col, alpha=0.7, label=f"{name}: p95 {np.percentile(ms, 95):.1f} ms")
    ax.axvline(60.0, color=fg.INK, lw=1.2, ls=":", label="limit 60 ms")
    ax.set_xscale("log")
    ax.set(title="Controller step time in S1, one BLAS thread", xlabel="ms", ylabel="steps")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=REFERENCE_SEED)
    ap.add_argument("--tune", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    lim = safety.tc_max()
    seed = REFERENCE_SEED if args.tune else args.seed
    with threadpool_limits(1):
        c, block_hash = commissioning(seed)
    if args.tune:
        t0 = time.perf_counter()
        res = tune(c, lim["tc_max"])
        res["wall_s"] = time.perf_counter() - t0
        TUNING_FILE.write_text(json.dumps(res, indent=2))
        print(json.dumps(res["chosen"], indent=2))
        return
    s = json.loads(TUNING_FILE.read_text())["chosen"]
    jobs = [(c, seed, lim["tc_max"], scen, name, s, mpc.NP, scen in ("S1", "S4"))
            for scen, names in PLAN.items() for name in names]
    if seed == REFERENCE_SEED:
        jobs += [(c, seed, lim["tc_max"], scen, "NMPC on the true model (oracle)", s, mpc.NP, False)
                 for scen in ORACLE_SCENARIOS]
    sizes = {k: len(f(seed, lim["tc_max"])["r"]) for k, f in cl.SCENARIOS.items()}
    jobs.sort(key=lambda j: -sizes[j[3]] * (4 if j[4].startswith("NMPC") else 1))   # longest first
    t0 = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as ex:
        out = list(ex.map(job, jobs))
    wall = time.perf_counter() - t0
    results, logs = {}, {}
    for scen, name, m, lg in out:
        results.setdefault(scen, {})[name] = m
        if lg is not None:
            logs[(scen, name)] = lg
    s4_moves(results, logs)
    res = {"seed": seed, "startup_block_sha256_16": block_hash,
           "machine": {"platform": platform.platform(), "processor": platform.processor() or platform.machine(),
                       "python": platform.python_version(), "blas_threads": 1, "workers": WORKERS},
           "settings": {"Np": mpc.NP, "Nc": mpc.NC, "du_max": mpc.DU_MAX, "tc_min": safety.TC_MIN,
                        "tc_max": lim["tc_max"], "sigma_y": mpc.SIGMA_Y, "sigma_u": mpc.SIGMA_U, **s,
                        "tuning": f"outputs/mpc_tuning.json (S0, seed {REFERENCE_SEED})",
                        "osqp": mpc.OSQP_SETTINGS, "feas_tol": mpc.FEAS_TOL, "ignition_T": cl.IGNITION_T,
                        "band": cl.BAND, "configs": {k: dict(zip(("kind", "adapt", "supervisor"), v))
                                                      for k, v in CONFIGS.items()}},
           "tc_max": {k: lim[k] for k in ("tc_max", "worst_limit", "worst_vertex", "margin", "ca_nominal_at_tc_max")},
           "commissioning": {"sigma": c["sigma"], "lam": c["lam"]},
           "wall_s_total": wall, "results": {k: results[k] for k in sorted(results)}}
    (OUT / "mpc_metrics.json").write_text(json.dumps(res, indent=2))
    if ("S1", "K-MPC adaptive") in logs:
        s1_figure(logs[("S1", "K-MPC adaptive")], lim["tc_max"], OUT / "fig6_s1.png")
        step_time_figure(logs, OUT / "fig6_step_time.png")
    print(f"done in {wall:.0f} s")


if __name__ == "__main__":
    main()
