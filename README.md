# Finding undiagnosed diabetes without a blood test

ES 335 (Machine Learning, IIT Gandhinagar) course project, 2026.

**Question.** Using only what an adult can answer and a nurse can measure with a
scale and a tape, which way of choosing people for an HbA1c test finds the most
undiagnosed dysglycaemia per test on a later survey: the ADA risk test, the
USPSTF 2021 rule, the ADA 2026 rule, or a model trained on 2007-2014 NHANES data?

**Data.** NHANES (CDC National Center for Health Statistics), six cycles from
2007-2008 to 2017-March 2020, public domain. Train on 2007-2014, calibrate on
2015-2016, test once on 2017-March 2020.

**Website.** https://adityapriyadarshi2025.github.io/nhanes-diabetes-screening/
(served by GitHub Pages from `docs/`).

## Result

The one-time test on 2017-March 2020 ran on 14 Nov 2026, after the models were
frozen (`FROZEN.md`, commit `b88cb3b`).

- **H1, the pre-registered confirmatory test, is not supported.** At the ADA
  risk test's number of tests, the LightGBM classifier found 70.5% of adults
  with HbA1c 5.7% or more, against 69.2% for the ADA test: +1.3 points (95%
  interval -1.4 to +3.8). The EBM agrees (+1.4, -0.9 to +3.9).
- Each test found slightly more cases: 45.8% of the adults LightGBM sent for a
  test had HbA1c 5.7% or more, against 43.4% (+2.4 points, +0.3 to +4.5).
- At the USPSTF 2021 rule's number of tests, the models found about 14 points
  more cases (LightGBM +13.9, +10.0 to +17.9), and the ADA risk test alone also
  beats the USPSTF rule. The ADA 2026 rule finds 98% of cases by testing 86% of
  adults.
- H2: predicting the HbA1c value made no detectable difference for 6.5%+.
  H3: the only step that clearly helped was giving a linear model all inputs
  instead of the seven ADA items (+3.0, +0.7 to +5.3). H4: before
  recalibration every model over-predicted in every later survey; after it,
  5.7%+ is well calibrated on the test (observed/expected 0.97) and 6.5%+ is
  under-predicted (1.15, 0.96 to 1.36). H5: among Asian adults, the ADA test,
  the USPSTF rule and the models each found about half of the cases.

Tables: `results/test/`. Charts and the full write-up: the website.

## How to run

```
bash scripts/download_nhanes.sh        # 72 files from wwwn.cdc.gov + codebooks, writes a SHA-256 manifest
pip install -r requirements.txt        # Python 3.10 or newer
python scripts/toolchain_check.py      # optional: model export to JavaScript works (needs Node.js)
python -m src.harmonise                # one recoded row per participant; seals the test labels
python -m src.cohort                   # cohort rules, counts per cycle
python -m src.explore_week1            # week-1 data checks -> results/week1/
python -m src.scores                   # baseline flag rates and AUC on 2007-2016
python -m src.fit                      # rolling-origin tuning, final models, calibration -> models/
python -m src.ladder                   # leakage ladder on 2007-2016
python -m src.test_once                # one-time test on 2017-March 2020, after FROZEN.md is committed
python -m src.export_web               # JSON for the website -> docs/data/
```

## Layout

```
scripts/   download_nhanes.sh  toolchain_check.py
src/       config.py  harmonise.py  cohort.py  scores.py  explore_week1.py
           models.py  fit.py  evaluate.py  ladder.py  test_once.py  export_web.py
models/    frozen models and frozen.json (settings, thresholds, hashes)
tests/     test_scores.py (python -m tests.test_scores)
docs/      the website (GitHub Pages) and decisions.md, a dated log of every design decision
results/   week1/, validation/ and test/ tables and figures
data/      raw/ codebooks/ processed/   (not in git; rebuilt by the steps above)
```

The 2017-March 2020 HbA1c values live only in
`data/processed/test_labels.sealed.parquet` and are read once, after the models
are frozen (see `PREREG.md` and `FROZEN.md`).
