# Adaptive kernel model of a CSTR: validation report

Process: continuous stirred-tank reactor, exothermic reaction A -> B. Input: coolant temperature Tc. Output: concentration Ca. All errors are in mol/L.
Data: simulated experiments on the textbook model (Henson & Seborg, 1997) with sensor noise, isolated spikes in 2 % of the samples and a logger gap added on purpose.

The identification system has three levels that share one RBF kernel:

1. an SVR filter rejects anomalous measurements;
2. kernel ridge regression (KRR) on a sliding window of 1000 samples is the process model;
3. a Gaussian-process (GP) supervisor measures how far the operating point is from the training data.

## 1. Data

- Samples: 8000, sampling period 0.1 min; missing values: 8.
- Estimated sensor noise: 0.0021 (standard deviation).
- Input range: 292.0 to 303.0 K, 221 distinct levels; the least-visited tenth of the range holds 6.7 % of samples.
- Start-up (cleaning, tuning, first model): first 5000 samples. Test: last 3000 samples, never used for fitting or tuning.

## 2. Level 1: anomaly filter

A measurement is rejected when it differs from the SVR one-step prediction by more than 3.5 standard deviations of the normal one-step scatter. On the test data the filter rejected 73 samples.

Precision / recall against the known injected spikes, test data:

| Threshold | SVR filter | Rolling median (reference) |
|---|---|---|
| 2.5 | 0.39 / 0.98 (61 of 62 caught, 155 flagged) | 0.64 / 1.00 (62 of 62 caught, 97 flagged) |
| 3.5 | 0.85 / 1.00 (62 of 62 caught, 73 flagged) | 0.95 / 1.00 (62 of 62 caught, 65 flagged) |
| 4.5 | 0.88 / 0.95 (59 of 62 caught, 67 flagged) | 0.98 / 0.98 (61 of 62 caught, 62 flagged) |

The rows other than 3.5 were run with the window frozen.
With the filter the one-step error of the window model is 0.0021; without it, 0.0036 (1.7 times larger).
For isolated spikes on a smooth signal a plain rolling median detects as well as the model-based filter or better; the model-based filter does not rely on the signal being smooth in time.

![filter](fig1_filter.png)

## 3. Level 2: process model

KRR and the linear reference (ARX) predict the next sample of Ca from the last two samples of Ca and Tc. Kernel width 2.00 and regularisation 0.0316 (normalised units) were chosen by exact leave-one-out cross-validation.
Errors are measured against the noise-free simulated output. Free-run: the model is fed its own predictions, as inside a controller.

| Model | One-step RMSE | Free-run RMSE | Free-run max error | Free-run fit |
|---|---|---|---|---|
| ARX (linear) | 0.0020 | 0.0111 | 0.0398 | 67.9 % |
| Kernel ridge | 0.0020 | 0.0057 | 0.0328 | 83.6 % |

The kernel model's free-run error is 2.0 times smaller than the linear model's.
The error is not uniform: where Ca is in its lowest tenth (below 0.848, the hot end, where the reactor gain is steepest) the kernel model's free-run RMSE is 0.0097; elsewhere it is 0.0050.

![free run](fig2_free_run.png)

## 4. Slow drift of the process

Catalyst activity falls linearly to 80 % over 3000 samples. One-step RMSE against the true output, by quarter of the run:

| Strategy | Q1 | Q2 | Q3 | Q4 | Whole run |
|---|---|---|---|---|---|
| Static model, no filter | 0.0048 | 0.0038 | 0.0036 | 0.0040 | 0.0041 |
| Sliding window, no filter | 0.0039 | 0.0040 | 0.0030 | 0.0026 | 0.0034 |
| SVR filter + sliding window | 0.0021 | 0.0016 | 0.0015 | 0.0015 | 0.0017 |

The full system is 2.4 times more accurate than the static model. Filter precision / recall during the drift: 0.87 / 1.00 (59 of 59 caught, 68 flagged).

![drift](fig3_drift.png)

## 5. Level 3: leaving the training range

The coolant temperature moves to 284-290 K, never seen during start-up, and returns. The supervisor is cautious when the GP variance exceeds 0.0316 and conservative above 0.0949; the penalty on control moves is then raised up to twofold.

| | Static model | Sliding window |
|---|---|---|
| GP variance inside the range (median) | 0.00058 | 0.00058 |
| GP variance outside (median) | 0.55983 | 0.00099 |
| Time outside spent cautious or conservative | 100 % | 6 % |
| Time inside spent cautious or conservative | 1.4 % | 1.6 % |
| One-step RMSE inside | 0.0033 | 0.0031 |
| One-step RMSE outside | 0.0419 | 0.0037 |

A frozen model stays uncertain for as long as the process is outside the range. With the sliding window the model relearns the new region and the supervisor returns to normal.

![supervisor](fig4_supervisor.png)

## 6. Numerical self-checks

- Analytical gradient against central differences: largest relative deviation 6.0e-09.
- Closed-form leave-one-out residuals against actual refits: largest deviation 1.8e-14.
- GP variance from the KRR factorisation against a library GP: largest deviation 1.1e-15.
- Leave-one-out search over 90 settings on 1000 samples: 2.3 s.

## 7. Design choices for a dynamic process

- The model is dynamic (NARX with 2 past outputs and 2 past inputs), not a static input-output map.
- Anomaly threshold 3.5 standard deviations: the one-step residual contains model error as well as sensor noise.
- Window of 1000 samples, refitted every 20 accepted samples; the SVR filter is refitted together with it.
- The leave-one-out surface is flat near its minimum; among settings within 2 % of the best, the strongest regularisation and then the narrowest kernel are taken.
- A rejected or missing sample is replaced by the model prediction in the regressor history.

## 8. Limits

- Validated for Tc between 292.0 and 303.0 K at a sampling period of 0.1 min.
- Simulation study on a textbook process; no plant data.
- The supervisor's penalty multiplier is computed but not yet used: there is no controller in this part.
- Anomalies are isolated spikes only; a sensor fault that lasts several samples is treated as a process change.
