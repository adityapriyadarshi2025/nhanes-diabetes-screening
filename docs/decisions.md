# Decisions log

Newest first. Each entry: what was decided and why.

## 14 Nov 2026 (week 4, one-time test)

- **Test run once** with `python -m src.test_once` at 03:19 IST, three minutes
  after the freeze commit `b88cb3b` (03:16 IST). Before reading the labels the
  script checked all 13 model files against `models/frozen.json` and the sealed
  labels against the hash in `PREREG.md`. It writes
  `results/test/UNSEALED.txt` and refuses to run a second time.
- **Time stamps.** `models/frozen.json` and `UNSEALED.txt` label their times
  "Asia": the container's time zone was set to Asia/Calcutta without zone
  data, so its clock ran on UTC. Both times are UTC (frozen 21:27 and unsealed
  21:49 on 13 Nov, which is 02:57 and 03:19 IST on 14 Nov).
- **H1 not supported.** LightGBM 70.5% against 69.2% for the ADA test at the
  ADA test's number of tests: +1.3 points (-1.4 to +3.8). EBM +1.4 (-0.9 to
  +3.9); without "borderline" answers +1.4 (-1.4 to +4.0).
- **Independent recheck** before anything was written up: ADA flags rescored
  from the raw inputs (identical for all 6,644 adults); LightGBM
  probabilities, sensitivity and PPV recomputed without `src/evaluate.py`; a
  separate Rao-Wu bootstrap (seed 7) gave -1.2 to +4.0 for H1.
- **Secondary results** (intervals in `results/test/`): PPV +2.4 points (+0.3
  to +4.5) at the ADA test's number of tests; sensitivity +13.9 (+10.0 to
  +17.9) at the USPSTF rule's. H2: no difference. H3: only the step from the
  ADA items to all inputs (+3.0, +0.7 to +5.3). H4: over-prediction before
  recalibration in every cycle; after it, 6.5%+ is under-predicted (1.15, 0.96
  to 1.36) because its share rose from 2.0% to 2.4%. H5: about half of the
  cases among Asian adults are found by every method except ADA 2026.
- **Website text** rewritten for this result: three waffle charts (USPSTF,
  ADA, LightGBM); H3-H5 paragraphs and the conclusion. The yield table no
  longer marks a "best" method. The main table gained the pre-registered
  "threshold re-set on 2017-2020" column. The charts have a phone layout.
  `src/export_web.py` adds the 2007-2014 unweighted and weighted prevalence
  quoted in the H4 text.
- **No test analysis was added or changed after the test.** The only new
  numbers are descriptive: the 2007-2014 prevalence and the frozen LightGBM's
  split-gain shares quoted under "Who the model misses".

## 14 Nov 2026 (weeks 2-3, models and freeze)

- **Ladder fitted** with rolling-origin tuning (`src/fit.py`); final models are
  the third-fold fits, trained on 2007-2014. Settings and validation in
  `FROZEN.md`.
- **Gate B finding**: LightGBM, EBM and L1 logistic are within 0.003 AUC out of
  time, all above the ADA test (0.77 vs 0.735 for HbA1c 5.7%+). Tuning stopped.
- **Leakage ladder**: a random split does not inflate AUC here (0.759 vs 0.772);
  counting diagnosed adults as positives (0.816) and adding "told prediabetes"
  and "blood test in the past 3 years" (0.855) do.
- **Calibration**: Platt scaling with survey weights on 2015-2016. Out of time
  the raw models over-predict (observed/expected about 0.87 for 5.7%+), so
  calibration matters.
- **Value model**: mean tuned by squared error of log HbA1c.
- **Cut for time**: survey-weighted training, adversarial validation, the
  with-race model, the ADA-items-only LightGBM, the fasting-glucose check,
  the 2021-2023 stress test and the isotonic check.
- **Website** moves to `docs/`, so GitHub Pages can serve it from the main
  branch.

## 19 Oct 2026 (week 1, data checks)

- **Download** done from wwwn.cdc.gov: 72 files and their codebook pages,
  checksums in `data/raw/manifest.csv`. Participant counts match CDC's totals
  (2007-08: 10,149 ... 2017-March 2020: 15,560).
- **Cohort**: 29,394 adults; 18,456 train, 4,294 calibrate, 6,644 test.
  The medication rule removed 316 people across all cycles.
- **2007-08 names the pills question `DID070`** (later cycles `DIQ070`);
  handled in `src/harmonise.py`. All other codes checked against the codebooks.
- **Baselines replicate the literature.** ADA score AUC for HbA1c 5.7% or more
  is 0.72-0.73 on 2009-2012 (Poltavskiy et al. 2016, prediabetes: 0.72-0.74). USPSTF 2021
  eligibility in 2017-March 2020 is 44.7% (O'Brien et al. 2023: 43.8%).
- **HbA1c level moves between cycles.** The age-standardised share with
  HbA1c 5.7% or more runs 25.4%, 28.6%, 26.7%, 23.6%, 26.6% over 2007-2016,
  with no clean step at the 2013 lab move. Consequence: calibrating on
  2015-2016 (same lab and analyser as the test cycle) stays, and H4 is a real
  question, not a formality.
- **Missing values are rare** (body and history items 0-5%; income 8-14%).
  Trees use native missing handling; logistic models use median imputation
  plus missing flags.
- **PREREG.md final.** Sealed-label hashes recorded before any model was fitted.

## 19 Oct 2026 (week 1, data)

- **Twelve NHANES components per cycle**: DEMO, GHB, DIQ, BMX, BPQ, MCQ, RHQ,
  PAQ, SMQ, WHQ, GLU, HDL. WHQ (self-reported height and weight) was added so
  the ADA risk test can also be scored the way people fill it in at home.
  GLU (fasting glucose) is used only for a label check; HDL only for the
  leakage ladder.
- **Test labels sealed at build time.** `src/harmonise.py` writes the
  2017-March 2020 HbA1c and fasting glucose to `test_labels.sealed.parquet`;
  every other file for that cycle has inputs only. The file's SHA-256 is printed
  so it can be recorded in `PREREG.md`.
- **Cohort rules** (`src/cohort.py`): aged 20 or over; examined; not pregnant
  (exam status `RIDEXPRG` = 1 or self-report `RHD143` = 1); HbA1c measured;
  answered "no" or "borderline" to "ever told you have diabetes" (`DIQ010`);
  and taking neither insulin (`DIQ050`) nor diabetes pills (`DIQ070`).
  The medication rule is new: people on glucose-lowering drugs are already
  under care and their HbA1c is lowered by treatment, so they are not
  screening candidates.
- **Refused or don't know** answers become missing, never "no".
- **Gestational diabetes**: `RHQ162` = 1 is yes; women never pregnant
  (`RHQ131` = 2) and men are "no". "Borderline" (`RHQ162` = 3) counts as no
  diagnosis; check this code against the codebook pages.
- **Physical activity**: three versions are stored (any of five activity
  types, leisure-time only, and moderate-equivalent minutes a week). Which one
  scores the ADA item is fixed in `scores.py` before any model is trained.
- **Survey design**: strata and PSU ids are prefixed with the cycle, so
  pooled analyses never merge strata from different cycles. Intervals use the
  Rao-Wu bootstrap (draw n-1 of a stratum's n PSUs, rescale weights).
- **Project folder**: `Documents/GitHub/nhanes-diabetes-screening`, next to
  the other GitHub Desktop repositories.
