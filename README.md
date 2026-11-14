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
- With its frozen threshold LightGBM also tested 1.5 points fewer adults than
  the ADA test (-3.0 to 0.0), so the pre-registered comparison slightly
  favoured the ADA test. Re-setting every threshold on 2017-2020 to test
  exactly the ADA test's share (added after the test, following a review), the
  models find +1.5 to +3.3 points more cases; LightGBM +2.6 (-0.3 to +5.9),
  with intervals above zero for L1, EBM and the value model. The higher PPV
  seen with the frozen thresholds was mostly the lower volume: at equal
  testing LightGBM's PPV gain is +1.6 (-0.2 to +3.7).
- Where the models did worse: for HbA1c 6.5%+ every model found fewer cases
  than the ADA test at its frozen threshold (LightGBM 83.0% vs 85.1%); the
  refitted ADA items were clearly worse (-8.2, -16.1 to -0.2); and among
  Mexican American and Black adults LightGBM found fewer cases (-4.8 and -3.4
  points, intervals including or touching zero).
- At the USPSTF 2021 rule's number of tests, the models found about 14 points
  more cases (LightGBM +13.9, +10.0 to +17.9), and the ADA risk test alone also
  beats the USPSTF rule. The ADA 2026 rule finds 98% of cases by testing 86% of
  adults.
- H2: predicting the HbA1c value made no detectable difference for 6.5%+.
  H3: the only step that clearly helped was giving a linear model all inputs
  instead of the seven ADA items (+3.0, +0.7 to +5.3). H4: before
  recalibration every model over-predicted in every later survey, mostly
  because training was unweighted (a weighted-training check moves
  observed/expected from 0.79-0.92 to 0.87-0.99); after recalibration, 5.7%+
  is well calibrated on the test (0.97) and 6.5%+ is under-predicted (1.15,
  0.96 to 1.36). H5: among Asian adults, the ADA test, the USPSTF rule and the
  models each found about half of the cases.
- Leakage ladder (20 seeds, revised after review): AUC 0.771 honest; a random
  split alone makes no reliable difference (0.767, seeds 0.752-0.782); adding
  diagnosed adults and two leaky questions lifts it to 0.843.

Tables: `results/test/` (the one-time test) and `results/review/` (analyses
added after the test). Charts and the full write-up: the website.

## How to run

```
bash scripts/download_nhanes.sh        # 72 files from wwwn.cdc.gov + codebooks, checked against data/raw/manifest.csv
pip install -r requirements.txt        # Python 3.10 or newer
python scripts/toolchain_check.py      # optional: model export to JavaScript works (needs Node.js)
python -m src.harmonise                # one recoded row per participant; seals the test labels
python -m src.cohort                   # cohort rules, counts per cycle
python -m src.explore_week1            # week-1 data checks -> results/week1/
python -m src.scores                   # baseline flag rates and AUC on 2007-2016
python -m src.fit                      # rolling-origin tuning, final models, calibration -> models/
python -m src.ladder --original        # leakage ladder exactly as frozen (FROZEN.md)
python -m src.ladder                   # leakage ladder over 20 seeds -> results/review/
python -m src.test_once --force        # one-time test on 2017-March 2020 (see below)
python -m src.review_extras            # equal-testing and subgroup intervals, added after the test
python -m src.dev_checks               # weighted training and grid-edge checks on 2007-2016
python scripts/recheck_h1.py           # H1 recomputed without src/evaluate.py
python -m src.export_web               # JSON for the website -> docs/data/
python scripts/prerender.py            # bakes the results text into docs/index.html (needs Playwright)
```

`bash scripts/download_nhanes.sh --verify` checks an existing copy against the
committed manifest without downloading; `--update-manifest` rewrites it.

The test has been run: `results/test/UNSEALED.txt` is committed as the record,
so `src.test_once` refuses to run again on a fresh clone unless given
`--force`. With the same data it reproduces `results/test/` exactly.

## Layout

```
scripts/   download_nhanes.sh  toolchain_check.py  recheck_h1.py  prerender.py
src/       config.py  harmonise.py  cohort.py  scores.py  explore_week1.py
           models.py  fit.py  evaluate.py  ladder.py  test_once.py  export_web.py
           review_extras.py  dev_checks.py   (added after the test)
models/    frozen models and frozen.json (settings, thresholds, hashes)
tests/     test_scores.py (python -m tests.test_scores)
docs/      the website (GitHub Pages) and decisions.md, a dated log of every design decision
results/   week1/, validation/, test/ and review/ tables and figures
data/      raw/ codebooks/ processed/   (not in git; rebuilt by the steps above)
```

The 2017-March 2020 HbA1c values live only in
`data/processed/test_labels.sealed.parquet` and are read once, after the models
are frozen (see `PREREG.md` and `FROZEN.md`).
