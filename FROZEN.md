# Frozen models: written before the test cycle was read

Frozen 13 Nov 2026, 21:27 UTC (14 Nov, 02:57 IST). At this point the
2017-March 2020 HbA1c values had not been read by any code or person. The git
commit that adds this file is the time stamp; `python -m src.test_once` runs
the one-time test only after checking every model file against the hashes
below and the sealed labels against the hash in `PREREG.md`.

## Settings chosen by rolling-origin tuning

| Model | Target | Setting (best mean over the three forward checks) |
| --- | --- | --- |
| L1 logistic | HbA1c 5.7%+ | C = 0.167 |
| L1 logistic | HbA1c 6.5%+ | C = 0.060 |
| LightGBM classifier (A) | HbA1c 5.7%+ | 15 leaves, minimum leaf 50, L2 penalty 5 |
| LightGBM classifier (A) | HbA1c 6.5%+ | 15 leaves, minimum leaf 200, L2 penalty 0 |
| LightGBM value model (B) | log HbA1c | 15 leaves, minimum leaf 100, L2 penalty 5 |

The final models are the fits of the third fold (trained on 2007-2014).
Full grids: `results/validation/tuning_grid.csv`.

## Out-of-time validation (mean of the three checks: 2011-12, 2013-14, 2015-16)

| Method | AUC, HbA1c 5.7%+ | Sensitivity at the ADA test's flag rate | AUC, HbA1c 6.5%+ |
| --- | --- | --- | --- |
| ADA risk test (points) | 0.735 | 0.653 | 0.776 |
| Refitted ADA items | 0.745 | 0.668 | 0.788 |
| L1 logistic | 0.770 | 0.693 | 0.815 |
| EBM | 0.768 | 0.683 | 0.813 |
| EBM with 10 pairs | 0.770 | 0.689 | 0.812 |
| LightGBM classifier | 0.768 | 0.683 | 0.811 |
| LightGBM value model | 0.764 | 0.679 | 0.809 |

LightGBM, EBM and L1 logistic sit within 0.003 AUC of each other, so tuning
stopped here (gate B). Per fold: `results/validation/validation_by_fold.csv`.

## Operating points (set on 2015-2016)

Share of 2015-2016 adults flagged (survey-weighted): ADA risk test 39.5%,
USPSTF 2021 43.8%, ADA 2026 85.4%. Each model's threshold flags the same
share of 2015-2016 and is applied unchanged to the test cycle. Thresholds and
Platt parameters (fitted on 2015-2016 with survey weights) are in
`models/frozen.json`.

## Leakage ladder (2007-2016 only)

| Setting | AUC, HbA1c 5.7%+ |
| --- | --- |
| Honest: train 2007-2014, test 2015-2016 | 0.772 |
| Random 80/20 split of 2007-2016 | 0.759 |
| Diagnosed and treated adults added as positives | 0.816 |
| Add "told prediabetes" and "blood test in the past 3 years" | 0.855 |
| Add HDL cholesterol | 0.858 |

## File hashes (SHA-256, first 16 characters)

| File | Hash |
| --- | --- |
| `models/ada_refit_y57.pkl` | `a9d0563ebaf0685e` |
| `models/l1_y57.pkl` | `0208f2869d21a4e3` |
| `models/ebm_y57.pkl` | `b6753e5f5279553f` |
| `models/ebm_pairs_y57.pkl` | `bacf664e247d827c` |
| `models/lgb_y57.txt` | `915447b988d9f791` |
| `models/ada_refit_y65.pkl` | `bedc090c0de0eb95` |
| `models/l1_y65.pkl` | `1171c3e3f1d4d908` |
| `models/ebm_y65.pkl` | `761834ad5e530711` |
| `models/ebm_pairs_y65.pkl` | `3760ec699e68a06f` |
| `models/lgb_y65.txt` | `fd0f954c966d6059` |
| `models/value_mu.txt` | `988897293d7e8779` |
| `models/value_sigma.txt` | `4bb3cf9b35bb882c` |
| `models/value_z.npy` | `893fe0c84eee3344` |
| `src/models.py` | `78670fd4d670fd28` |
| `src/fit.py` | `d69c9436437e333e` |
| `src/evaluate.py` | `a165ce6d85e532cf` |
| `src/test_once.py` | `d62af0745e36a0f0` |

Full hashes of all model and code files: `models/frozen.json`.

## Deviations from PREREG.md

1. The value model's mean was tuned by squared error of log HbA1c, the natural
   loss for a regression, instead of log-loss.
2. Platt scaling uses the survey weights, so probabilities describe US adults.
3. Smoking enters as two indicators (former, current); "leisure and total
   activity" are `active_rec` and `mvpa_min_wk`.
4. Not run, for lack of time: survey-weighted training, adversarial
   validation, the with-race model, the ADA-items-only LightGBM, the
   fasting-glucose label check, the 2021-2023 stress test and the isotonic
   calibration check.
5. The freeze happened on 14 Nov instead of the planned 6 Nov.
