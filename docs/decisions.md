# Decisions log

Newest first. Each entry: what was decided and why.

## 15 Nov 2026 (after the test: external review)

An external review of the repo and site (14 Nov) found real problems. Every
number below was recomputed here before anything changed. No model,
threshold or confirmatory result changed.

- **PPV claim corrected.** With the frozen thresholds LightGBM tested 1.5
  points fewer adults than the ADA test (-3.0 to 0.0), and at equal volume PPV
  moves with sensitivity, so "wasted fewer tests" was mostly a volume effect.
  `src/review_extras.py` re-sets each threshold on 2017-2020 to the rule's
  share, again in every bootstrap replicate: LightGBM +2.6 (-0.3 to +5.9), PPV
  +1.6 (-0.2 to +3.7); L1 +3.3 (+0.8 to +5.4), EBM +3.0 (+0.5 to +5.6), value
  model +2.9 (+0.9 to +5.4). The review's figures (+2.6, -0.3 to +5.3; PPV
  +1.6, -0.6 to +3.8) differ only in bootstrap draws. This comparison was
  pre-registered without intervals; the intervals are post-hoc and not
  adjusted for six models.
- **Negative results now discussed on the site**: 6.5%+ (ADA test 85.1%,
  every model lower at the frozen thresholds; -6.4 to +0.9 at equal testing,
  all intervals include zero), refitted ADA items -8.2 (-16.1 to -0.2), and
  paired subgroup differences (Mexican American -4.8, -11.6 to +1.9; Black
  -3.4, -6.7 to 0.0; other Hispanic +8.0, -0.1 to +14.9) in
  `results/review/subgroup_differences.csv`.
- **Leakage ladder.** Step 1 reused the frozen model's rounds, early-stopped
  on 2015-2016, the rows step 1 is scored on; step 2 rested on one seed.
  `src/ladder.py` now chooses rounds on a fifth of each step's training rows
  and averages 20 seeds: 0.771, 0.767 (seeds 0.752-0.782), 0.807, 0.843,
  0.846. `--original` reproduces the frozen ladder byte for byte.
- **Weighted training (H4).** `src/dev_checks.py`: survey-weighted LightGBM
  moves uncalibrated observed/expected on the three forward checks from
  0.79-0.92 to 0.87-0.99 (5.7%+) and from 0.64-0.77 to 0.81-1.05 (6.5%+);
  AUC changes by -0.012 to +0.002. Unweighted training explains most of the
  over-prediction; the rest, largest in 2013-2014, is drift.
- **Grid edge.** Settings beyond the edge of the grid, tuned exactly as in
  `src/fit.py` (which reproduces the frozen settings' log-loss to six
  decimals): best improvement 0.00004 (0.01%) for 5.7%+ and 0.00039 (0.32%)
  for 6.5%+. Frozen models kept.
- **Site.** Data notes (the pre-pandemic file and its weights, top-coding,
  `RIDRETH3` from 2011) and the literature checks are on the page; the results
  text is baked into `index.html` by `scripts/prerender.py`, so it reads
  without JavaScript; a contributions section was added.
- **Repo.** The download script verifies the committed manifest (`--verify`,
  `--update-manifest`). The README explains `--force`; a scratch rerun with
  `--force` reproduced all eight `results/test` files byte for byte. PREREG.md
  separates its writing date (19 Oct) from its first commit (28 Oct).
  `scripts/recheck_h1.py` is in the repo, and the LightGBM split-gain shares
  quoted on the site (age 49%, waist-to-height 13%, BMI 7.5%) are exported by
  `src/export_web.py`.
- **Code hashes.** `models/frozen.json` hashes every file in `src/` at the
  freeze. `export_web.py` and `ladder.py` changed since (site export; ladder
  fix). Every file that produces the test (`models.py`, `fit.py`,
  `evaluate.py`, `test_once.py`, `config.py`, `harmonise.py`, `cohort.py`,
  `scores.py`) is unchanged.
- **Gestational diabetes code 3** checked in the codebook pages: "Borderline"
  in `RHQ_E` (21 women) and `P_RHQ` (17 women). Settled.

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
  diagnosis; check this code against the codebook pages. (Checked 15 Nov
  2026: code 3 is "Borderline" in the codebooks.)
- **Physical activity**: three versions are stored (any of five activity
  types, leisure-time only, and moderate-equivalent minutes a week). Which one
  scores the ADA item is fixed in `scores.py` before any model is trained.
- **Survey design**: strata and PSU ids are prefixed with the cycle, so
  pooled analyses never merge strata from different cycles. Intervals use the
  Rao-Wu bootstrap (draw n-1 of a stratum's n PSUs, rescale weights).
- **Project folder**: `Documents/GitHub/nhanes-diabetes-screening`, next to
  the other GitHub Desktop repositories.
