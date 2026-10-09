# Adaptive kernel model of a nonlinear reactor (CSTR) in Python

A complete example of process identification with a three-level kernel architecture:
from a raw plant-style log to a validated nonlinear model that rejects bad measurements,
follows slow drift and knows where it can be trusted.

**Process.** Continuous stirred-tank reactor with an exothermic reaction A -> B, a standard
textbook benchmark (Henson & Seborg, *Nonlinear Process Control*, 1997). Input: coolant
temperature. Output: concentration of A. The process gain changes several-fold across the
operating range (19-fold between 292 and 303 K), so a single linear model cannot describe it.
Near the top of the range the reactor has three steady states, and the low-temperature one
disappears at Tc = 303.23 K, 0.23 K above the identified range (see "Scope").

**Architecture.** Three kernel methods share one RBF kernel, each with its own job:

| Level | Method | Job |
|---|---|---|
| 1 | Support vector regression (SVR) | rejects anomalous measurements before they reach the model |
| 2 | Kernel ridge regression (KRR) on a sliding window | the process model; refitted as new data arrive |
| 3 | Gaussian process (GP) variance | checks whether the trajectory planned over the next 2 min runs through known data and raises the penalty on control moves |

KRR is written out in NumPy (`kernel.py`). One matrix factorisation gives the model, the
exact leave-one-out error for tuning, the analytical gradient and the GP variance.

**What the script does**

1. Simulates an identification experiment with noise, spikes and a logger gap.
2. Start-up on historical data: cleans it with the filter, tunes the kernel by exact leave-one-out and
   calibrates the anomaly threshold on out-of-sample residuals.
3. Tests the filter and the model on unseen data; compares with linear ARX and OE models in free-run simulation.
4. Runs a slow-drift scenario (catalyst deactivation) with and without each level; the main measure is
   the free run of each quarter by the model frozen at its start.
5. Runs an excursion below the training range and one above it, past the stability limit, and shows
   the supervisor's reaction.
6. Checks the closed-form results numerically and writes `outputs/report.md`.
7. `run_seeds.py` repeats everything over 20 seeds and adds the spread to the report.

**Run**

    pip install -r requirements.txt     # Python 3.13; the versions of the reference run
    python run_demo.py                  # one seed, about 4 minutes on 2 cores
    python run_seeds.py                 # 20 seeds, about 37 minutes on 2 cores
    python run_demo.py --data log.csv   # your own log (t_min, Tc_K, Ca_mol_L): steps 2-3 and 6

**Files**

| File | Purpose |
|---|---|
| `cstr.py` | reactor equations, simulator, steady states and stability limit |
| `data.py` | simulated plant logs, noise estimate, reference detectors |
| `kernel.py` | RBF kernel, KRR, exact leave-one-out, gradient, GP variance |
| `adaptive.py` | SVR filter, sliding window, GP supervisor, online loop |
| `pipeline.py` | start-up, tuning, threshold calibration, reference models, scoring |
| `checks.py` | numerical self-checks |
| `figures.py`, `report.py` | figures and report template |
| `run_demo.py` | the scenarios |
| `run_seeds.py` | the scenarios over many seeds |

**Results** (`outputs/report.md`; errors in mol/L, sensor noise 0.002; reference run = seed 0;
spread = median [25th; 75th percentile] over seeds 0-19)

| Test | Reference run | 20 seeds |
|---|---|---|
| Free-run RMSE on unseen data: kernel / linear OE / linear ARX | 0.0055 / 0.0104 / 0.0111 | 0.0055 / 0.0085 / 0.0104 |
| Free-run RMSE ratio, linear OE / kernel | 1.9 | 1.55 [1.37; 1.82] |
| Anomaly filter on unseen data, precision / recall | 0.82 / 1.00 | 0.90 [0.89; 0.93] / 0.98 [0.97; 1.00] |
| Same, causal rolling median calibrated the same way | 0.75 / 0.81 | 0.79 [0.74; 0.84] / 0.79 [0.76; 0.82] |
| One-step error without / with the filter | 0.0034 / 0.0020 | ratio 1.70 [1.56; 1.74] |
| Slow drift, free run of the last quarter: static model / sliding window (both filtered) | 0.0257 / 0.0094 | ratio 3.48 [3.21; 3.75] |
| Excursion below the range, sliding window: warning before the input returns | 18 samples | 18 [18; 18] |

![free run](outputs/fig2_free_run.png)
![supervisor](outputs/fig4_supervisor.png)
![hot side](outputs/fig5_hot_side.png)

**Scope.** Simulation study on a textbook process; not plant data. No controller yet:
the supervisor's penalty multiplier is computed for a model-predictive controller that
is the next part of this project. At 304 K the reactor ignites (in the reference run the
temperature rises from 320 to 458 K), and the GP supervisor warned at least 1 min before Ca
left the start-up range in only 11 of 20 seeds, so the controller needs hard constraints.
A fixed margin on Tc from 303.23 K is not reliable: 5 % less heat transfer moves that limit to
301.55 K, inside the identified range, while Ca at the limit stays between 0.739 and 0.776 mol/L
across the cases checked (report, section 4). The controller will therefore use
Tc <= 303.0 K (edge of the data) and Ca >= 0.79 mol/L.

**Background.** The three-level architecture (SVR filter, sliding-window KRR, GP supervisor
on one shared kernel) comes from my PhD thesis, where it was developed for iron-ore magnetic
separation. This repository is a new implementation on an open textbook process; it contains
no code or data from the thesis. Changes made for a dynamic process are listed in the report.

- O. Volovetskyi. *Predictive Control of a Nonlinear Magnetic Separation Process Based on
  Kernel Functions.* PhD thesis, Kryvyi Rih National University, 2026.
- O. Volovetskyi. Local Linearization of Kernel Models in Real Time as a Basis for Fast
  Optimization in MPC. *Automation of Technological and Business Processes*, 17(4), 98-103,
  2025. DOI: 10.15673/atbp.v17i4.3319
- O. Volovetskyi. Utilizing Gaussian Process Regression for Nonlinear Magnetic Separation
  Process Identification. *IAPGOS*, 14(3), 21-28, 2024. DOI: 10.35784/iapgos.5954
- O. Volovetskyi. Methods of Filtering and Regression for Forecasting Noisy Timeseries Based
  on Machine Learning. *Herald of Advanced Information Technology*, 8(1), 13-27, 2025.
  DOI: 10.15276/hait.08.2025.1

**License.** MIT.
