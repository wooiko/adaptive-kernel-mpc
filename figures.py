"""Figures for the report (static PNG)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2.0, "figure.dpi": 130,
})


def _save(fig, path, handles_from, ncol, bottom=0.07):
    fig.legend(*handles_from.get_legend_handles_labels(), loc="lower center", ncol=ncol)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    fig.savefig(path)
    plt.close(fig)


def filter_fig(t, y_raw, run, path, n=1200):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    s = slice(0, n)
    fl = run["flagged"][s]
    ax.plot(t[s], y_raw[s], ".", ms=3, color=MUTED, label="raw measurement")
    ax.plot(t[s], run["y_clean"][s], color=BLUE, lw=1.2, label="after the SVR filter")
    ax.plot(t[s][fl], y_raw[s][fl], "o", ms=7, mfc="none", mec=ORANGE, mew=1.6,
            label="rejected as anomalous")
    ax.set(title="Level 1: SVR filter on unseen data", xlabel="time, min", ylabel="Ca, mol/L")
    _save(fig, path, ax, 3, 0.09)


def free_run_fig(t, ref, sims, ref_name, path, n=900):
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    s = slice(0, n)
    ax[0].plot(t[s], ref[s], color=INK, lw=1.0, label="process (" + ref_name + ")")
    for (name, ys), c in zip(sims.items(), (BLUE, ORANGE)):
        ax[0].plot(t[s], ys[s], color=c, lw=1.6, label=name)
        ax[1].plot(t[s], (ys - ref)[s], color=c, lw=1.2)
    ax[0].set(title="Level 2: free-run simulation on unseen data (models are fed their own predictions)",
              ylabel="Ca, mol/L")
    ax[1].axhline(0, color=MUTED, lw=0.8)
    ax[1].set(title="Simulation error", ylabel="model - process, mol/L", xlabel="time, min")
    _save(fig, path, ax[0], 3)


def _rolling_rmse(e, w=300):
    e2 = np.where(np.isnan(e), 0.0, e ** 2)
    return np.sqrt(np.convolve(e2, np.ones(w) / w, mode="same"))


def drift_fig(t, true, runs, k0_scale, path):
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True, gridspec_kw={"height_ratios": [1, 2.4]})
    ax[0].plot(t, 100 * k0_scale, color=INK, lw=1.4)
    ax[0].set(title="Catalyst activity (reaction-rate constant, % of initial)", ylabel="%")
    for (name, o), c in zip(runs.items(), (BLUE, AQUA, ORANGE)):
        ax[1].plot(t, _rolling_rmse(o["pred"] - true), color=c, label=name)
    ax[1].set(title="One-step prediction error against the true output (rolling RMSE, 30 min)",
              ylabel="RMSE, mol/L", xlabel="time, min", ylim=(0, None))
    _save(fig, path, ax[1], 3)


def supervisor_fig(t, u, runs, lam, tc_range, path):
    fig, ax = plt.subplots(3, 1, figsize=(9, 6.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.6, 1]})
    ax[0].axhspan(*tc_range, color=GRID, alpha=0.7, lw=0)
    ax[0].step(t, u, where="post", color=INK, lw=1.2)
    ax[0].set(title="Coolant temperature (grey band: range seen in training)", ylabel="Tc, K")
    for (name, o), c in zip(runs.items(), (BLUE, ORANGE)):
        ax[1].plot(t, o["var"], color=c, lw=1.2, label=name)
        ax[2].plot(t, o["mult"], color=c, lw=1.4)
    for th, lab in ((lam, "cautious above"), (3 * lam, "conservative above")):
        ax[1].axhline(th, color=MUTED, lw=0.8, ls="--")
        ax[1].text(t[-1], th, " " + lab, va="bottom", ha="right", fontsize=8, color=MUTED)
    ax[1].set(title="Level 3: GP variance at the current operating point", ylabel="variance (log scale)",
              yscale="log")
    ax[2].set(title="Control-move penalty multiplier R / R0", ylabel="R / R0", xlabel="time, min",
              ylim=(0.9, 2.1))
    _save(fig, path, ax[1], 2)
