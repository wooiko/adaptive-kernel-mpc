# Adaptive kernel model of a CSTR: validation report

Process: continuous stirred-tank reactor, exothermic reaction A -> B. Input: coolant temperature Tc. Output: concentration Ca. All errors are in mol/L.
Data: simulated experiments on the textbook model (Henson & Seborg, 1997) with sensor noise, isolated spikes in 2 % of the samples and a logger gap added on purpose.
Sections 1-7 are one realisation (seed 0); section 8 repeats the experiments over many seeds.

The identification system has three levels that share one RBF kernel:

1. an SVR filter rejects anomalous measurements;
2. kernel ridge regression (KRR) on a sliding window of 1000 samples is the process model;
3. a Gaussian-process (GP) supervisor measures how far the trajectory planned from the current operating point is from the training data.

## 1. Data

- Samples: 8000, sampling period 0.1 min; missing values: 8.
- Estimated sensor noise: 0.00201 (standard deviation; second differences, robust scale with outlier trimming).
- Input range: 292.0 to 303.0 K, 221 distinct levels; the least-visited tenth of the range holds 6.7 % of samples.
- Start-up (cleaning, tuning, threshold calibration, first model): first 5000 samples. Test: last 3000 samples, never used for fitting, tuning or calibration.

## 2. Level 1: anomaly filter

A measurement is rejected when it differs from the SVR one-step prediction by more than a threshold. The threshold is calibrated on start-up data so that a chosen share of normal samples is rejected: the filter is replayed as online (an SVR on the last 1000 rows, refitted every 20 rows, predicts the rows that follow), and the threshold is the corresponding quantile of these out-of-sample residuals, divided by the in-sample robust scale of the SVR that produced them (3759 rows). The residuals have heavier tails than a normal distribution, so the threshold is not a number of Gaussian standard deviations.
Default share: 0.5 %; threshold 3.15 times the in-sample robust scale. On the test data the filter rejected 76 samples.

Precision / recall against the known injected spikes, test data. Every row is the full adaptive system. The rolling medians (window 7) are calibrated to the same share on the same start-up data.

| Target share of normal samples rejected | SVR filter | Rolling median, causal | Rolling median, centred (non-causal: needs 3 future samples) |
|---|---|---|---|
| 1.0 % | 0.69 / 0.98 (61 of 62 caught, 89 flagged; 0.95 % of normal samples rejected) | 0.68 / 0.85 (53 of 62 caught, 78 flagged; 0.85 % of normal samples rejected) | 0.63 / 1.00 (62 of 62 caught, 99 flagged; 1.26 % of normal samples rejected) |
| 0.5 % | 0.82 / 1.00 (62 of 62 caught, 76 flagged; 0.48 % of normal samples rejected) | 0.75 / 0.81 (50 of 62 caught, 67 flagged; 0.58 % of normal samples rejected) | 0.83 / 1.00 (62 of 62 caught, 75 flagged; 0.44 % of normal samples rejected) |
| 0.2 % | 0.97 / 1.00 (62 of 62 caught, 64 flagged; 0.07 % of normal samples rejected) | 0.90 / 0.69 (43 of 62 caught, 48 flagged; 0.17 % of normal samples rejected) | 0.91 / 1.00 (62 of 62 caught, 68 flagged; 0.20 % of normal samples rejected) |

At the default share the SVR filter has precision 0.82 and recall 1.00; the causal rolling median, the detector usable online without delay, has 0.75 and 0.81. The centred median sees 3 future samples and could only run with a 0.3-min delay.
With the filter the one-step error of the window model is 0.0020; without it, 0.0034 (1.7 times larger).

![filter](fig1_filter.png)

## 3. Level 2: process model

KRR and two linear references predict the next sample of Ca from the last two samples of Ca and Tc: ARX (least squares on the one-step error) and OE (least squares on the free-run error). Kernel width 3.16 and regularisation 0.0316 (normalised units) were chosen by exact leave-one-out cross-validation.
Errors are measured against the noise-free simulated output. Free-run: the model is fed its own predictions, as inside a controller.

| Model | One-step RMSE | Free-run RMSE | Free-run max error | Free-run fit |
|---|---|---|---|---|
| ARX (linear, equation error) | 0.0019 | 0.0111 | 0.0399 | 67.9 % |
| OE (linear, output error) | 0.0040 | 0.0104 | 0.0442 | 69.9 % |
| Kernel ridge | 0.0017 | 0.0055 | 0.0318 | 84.1 % |

The kernel model's free-run error is 2.0 times smaller than ARX's and 1.9 times smaller than OE's.
ARX coefficients are biased by the noise in the lagged outputs; OE removes that bias, so the comparison with OE is the one that isolates the effect of the nonlinearity.
The error is not uniform: where Ca is in its lowest tenth (below 0.848, the hot end, where the reactor gain is steepest) the kernel model's free-run RMSE is 0.0092; elsewhere it is 0.0049.

![free run](fig2_free_run.png)

## 4. Operating range and stability of the reactor

Steady states of the simulated reactor on the cold (low-temperature) branch, computed from the model equations:

| Tc, K | Ca, mol/L | Gain dCa/dTc, mol/(L K) | Slowest time constant, min | Steady states |
|---|---|---|---|---|
| 292.0 | 0.9435 | -0.0047 | 0.89 | 1 |
| 297.0 | 0.9116 | -0.0087 | 0.77 | 1 |
| 300.0 | 0.8773 | -0.0154 | 0.95 | 3 |
| 302.0 | 0.8348 | -0.0310 | 1.34 | 3 |
| 303.0 | 0.7871 | -0.0879 | 2.38 | 3 |

Across the identified range the gain grows 19-fold and the dynamics at the upper edge are 3.1 times slower than at their fastest. In the upper part of the range three steady states coexist. The cold steady state disappears at Tc = 303.23 K (saddle-node point), 0.23 K above the upper edge of the identified range; above it only the hot steady state remains, and it is unstable (eigenvalues with positive real part), so the reactor ignites and cannot settle (section 7).

The saddle-node point moves strongly with the process parameters, while Ca at that point moves little:

| Parameters | Tc at the limit, K | Ca at the limit, mol/L | T at the limit, K |
|---|---|---|---|
| nominal | 303.23 | 0.747 | 335.4 |
| UA -5 % (fouling) | 301.55 | 0.762 | 334.4 |
| UA -10 % | 299.74 | 0.776 | 333.4 |
| Tf +1 K | 302.75 | 0.747 | 335.4 |
| k0 x0.8 (end of the drift scenario) | 307.52 | 0.739 | 338.9 |

A fixed margin on Tc is therefore not reliable: with 5 % less heat transfer the limit already lies inside the identified range. Constraints for a controller:
- Tc <= 303.0 K, the edge of the data (no extrapolation of the model);
- Ca >= 0.79 mol/L, the steady-state Ca at the edge of the data rounded up. It stays above Ca at the limit in every case of the table, by at least 0.014 mol/L (7 times the sensor noise).
Closed-loop behaviour with these constraints is not tested: there is no controller in this part.

## 5. Slow drift of the process

Catalyst activity falls linearly to 80 % over 3000 samples. The steady-state Ca at the end is higher by 0.014 at 292 K, 0.026 at 297 K, 0.103 at 303 K.

One-step RMSE against the true output, by quarter of the run:

| Strategy | Q1 | Q2 | Q3 | Q4 | Whole run |
|---|---|---|---|---|---|
| Static model, no filter | 0.0048 | 0.0038 | 0.0036 | 0.0039 | 0.0041 |
| Static model + SVR filter | 0.0036 | 0.0023 | 0.0034 | 0.0044 | 0.0035 |
| Sliding window, no filter | 0.0039 | 0.0040 | 0.0030 | 0.0026 | 0.0034 |
| SVR filter + sliding window | 0.0026 | 0.0017 | 0.0015 | 0.0015 | 0.0019 |

The one-step error is dominated by the measurement history and hardly grows with the drift. Without adaptation the model is the same with or without the filter, so the two static rows of the next table coincide. The test that matters for a controller is the free run: each quarter is simulated by the model frozen at the start of that quarter. RMSE / largest error:

| Strategy | Q1 | Q2 | Q3 | Q4 |
|---|---|---|---|---|
| Static model, no filter | 0.0098 / 0.0273 | 0.0119 / 0.0273 | 0.0207 / 0.0372 | 0.0257 / 0.0508 |
| Static model + SVR filter | 0.0098 / 0.0273 | 0.0119 / 0.0273 | 0.0207 / 0.0372 | 0.0257 / 0.0508 |
| Sliding window, no filter | 0.0098 / 0.0273 | 0.0120 / 0.0255 | 0.0112 / 0.0252 | 0.0081 / 0.0166 |
| SVR filter + sliding window | 0.0098 / 0.0273 | 0.0120 / 0.0327 | 0.0101 / 0.0269 | 0.0094 / 0.0222 |

Contributions, separated:
- filter (both models static), one-step RMSE: 0.0041 without, 0.0035 with;
- adaptation (both with the filter), free-run RMSE in Q4: 0.0257 static, 0.0094 with the sliding window (2.7 times);
- a frozen filter on a drifting process rejects valid data: static filter 0.50 / 0.90 (53 of 59 caught, 107 flagged; 1.84 % of normal samples rejected); filter refitted with the window 0.77 / 0.98 (58 of 59 caught, 75 flagged; 0.58 % of normal samples rejected).

![drift](fig3_drift.png)

## 6. Level 3: leaving the training range (cold side)

The coolant temperature moves to 284-290 K, never seen during start-up, and returns. The supervisor receives the largest GP variance along the free-run of the model over the next 20 samples (2 min) under the planned input; in this open-loop test the plan is the input schedule itself, in a predictive controller it is the candidate input sequence. It is cautious above 0.0316 and conservative above 0.0949; the penalty on control moves rises continuously up to twofold. The thresholds are a heuristic: the GP prior variance is fixed at 1, so the variance measures data coverage, not a calibrated error.

| | Static model | Sliding window |
|---|---|---|
| GP variance inside the range, current point (median) | 0.00033 | 0.00035 |
| GP variance outside, current point (median) | 0.17816 | 0.00059 |
| GP variance outside, along the planned trajectory (median) | 0.28111 | 0.00272 |
| GP variance outside with the input set to 292.0 K (median) | 0.01364 | 0.00559 |
| GP variance outside with the input set to 297.5 K (median) | 0.01852 | 0.02163 |
| GP variance outside with the input set to 303.0 K (median) | 0.02778 | 0.02877 |
| Time outside not normal: current point only | 99 % | 2 % |
| Time outside not normal: planned trajectory (supervisor) | 100 % | 10 % |
| Time inside not normal (supervisor) | 3.2 % | 3.0 % |
| Warning before the input leaves the range, samples: current point / trajectory | 0 / 18 | 0 / 17 |
| Warning before the input returns, samples: current point / trajectory | already warning / already warning | 0 / 18 |
| Samples from leaving the range to the first normal sample (supervisor) | never | 5 |
| One-step RMSE inside | 0.0025 | 0.0022 |
| One-step RMSE outside | 0.0280 | 0.0021 |
| Filter outside: precision / recall | 0.67 / 0.89 | 0.75 / 1.00 |
| Samples outside in filter jump mode | 472 | 8 |

A frozen model stays uncertain for as long as the process is outside the range. The sliding window learns the neighbourhood of the current operating point within a few samples, so the variance at the current point alone soon reads normal. Normal then means only that the planned trajectory runs through data the window has seen; the probed variances show the model's coverage of other inputs. Because the supervisor checks the planned trajectory, it warns before the input leaves the range and before it returns, by up to the horizon length.
With the window frozen (static model) the filter cannot agree with the new data, so it stays in jump mode, where single spikes are caught by the model-free trend test.

![supervisor](fig4_supervisor.png)

## 7. Hot side: beyond the stability limit

The coolant temperature is held at 304.0 K, 1 K above the identified range and beyond the saddle-node point, for 60 samples (6 min), then returns.
- The reactor ignites: Ca leaves the start-up range (below 0.789) at sample 324, 24 samples after the step; Ca falls to 0.003 and the reactor temperature rises from 320.3 to 458.1 K. After the return it settles back on the cold branch (Ca 0.923, T 317.9 K).
- Before ignition the largest variance along the planned trajectory, which already contains 304 K, is 0.0204 (static) and 0.0227 (window), against the cautious threshold 0.0316. The variance measures the distance from the data, not the closeness of the stability limit, and 1 K beyond the data that distance is small.
- The first warning comes at sample 326 (static) and 327 (window); Ca leaves the start-up range at sample 324. A warning only raises the move penalty; it does not stop the input.

The supervisor is therefore not a substitute for a hard input constraint (section 4); section 8 shows how often it warns in time.

![hot side](fig5_hot_side.png)

## 8. Spread over 20 seeds

All simulated experiments repeated with seeds 0-19 (new input sequences, noise, spikes and start-up for each seed). Median [25th; 75th percentile]:

| Quantity | Median [IQR] |
|---|---|
| Estimated / true sensor noise | 1.006 [0.989; 1.015] |
| Free-run RMSE, kernel model | 0.0055 [0.0050; 0.0065] |
| Free-run RMSE, ARX | 0.0104 [0.0098; 0.0113] |
| Free-run RMSE, OE | 0.0085 [0.0078; 0.0093] |
| Free-run RMSE ratio ARX / kernel | 1.89 [1.64; 2.10] |
| Free-run RMSE ratio OE / kernel | 1.55 [1.37; 1.82] |
| SVR filter precision (default share) | 0.90 [0.89; 0.93] |
| SVR filter recall (default share) | 0.98 [0.97; 1.00] |
| SVR filter: normal samples rejected | 0.2 % [0.2; 0.3] |
| Causal median precision (same share) | 0.79 [0.74; 0.84] |
| Causal median recall (same share) | 0.79 [0.76; 0.82] |
| One-step RMSE ratio without / with filter | 1.70 [1.56; 1.74] |
| Drift, one-step: static no filter / full system | 2.26 [2.18; 2.41] |
| Drift, one-step: static no filter / static + filter | 1.33 [1.24; 1.40] |
| Drift, free-run Q4: static + filter / full system | 3.48 [3.21; 3.75] |
| Cold excursion, window: time outside not normal | 12.6 % [9.2; 16.1] |
| Cold excursion, window: warning before return, samples | 18 [18; 18] |
| Cold excursion, window: samples to normal after leaving | 10 [8; 25] |
| Cold excursion, static: time outside not normal | 99.8 % [95.8; 100.0] |
| Hot side: largest planned-trajectory variance before ignition / th1 | 1.17 [0.69; 1.84] |
| Hot side: first warning relative to Ca leaving the start-up range, samples (negative = before) | -34 [-43; 2] |

Hot side: the supervisor warned at least 10 samples (1 min) before Ca left the start-up range in 11 of 20 seeds; even when it warns, it only raises the move penalty.
The linear OE model was more accurate than the kernel model in 1 of 20 seeds.

## 9. Numerical self-checks

- Analytical gradient against central differences: largest relative deviation 7.8e-09 (normalised units), 1.9e-09 (physical units, as needed for MPC).
- Closed-form leave-one-out residuals against actual refits: largest deviation 3.2e-14.
- GP variance from the KRR factorisation against a library GP: largest deviation 1.1e-15.
- Leave-one-out search over 90 settings on 1000 samples: 2.3 s.
- These checks confirm that the code matches the formulas; they do not show that the GP variance is a calibrated error estimate.

## 10. Design choices for a dynamic process

- The model is dynamic (NARX with 2 past outputs and 2 past inputs), not a static input-output map.
- Anomaly threshold: calibrated to reject 0.5 % of normal samples on start-up data (out-of-sample residual quantile).
- Window of 1000 samples, refitted every 20 accepted samples; the SVR filter is refitted together with it.
- A run of 4 rejections is taken as a process change: the rejected samples are restored to the history and the window; until the filter agrees again, single spikes are caught by a model-free trend test and the models are refitted every 5 accepted samples.
- The supervisor checks the trajectory planned over 20 samples, not only the current point.
- The leave-one-out surface is flat near its minimum; among settings within 2 % of the best, the strongest regularisation and then the narrowest kernel are taken.
- A rejected or missing sample is replaced by the model prediction in the regressor history.

## 11. Limits

- Validated for Tc between 292.0 and 303.0 K at a sampling period of 0.1 min.
- Simulation study on a textbook process; no plant data.
- The supervisor's penalty multiplier is computed but not yet used: there is no controller in this part.
- Anomalies are isolated spikes only; a sensor fault that lasts several samples is treated as a process change.
