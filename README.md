# Finding undiagnosed diabetes without a blood test

ES 335 (Machine Learning, IIT Gandhinagar) course project, 2026, by Aditya
Priyadarshi.

**Website:** https://adityapriyadarshi2025.github.io/nhanes-diabetes-screening/
(served by GitHub Pages from `docs/`). The website presents the reasoning; this
repository lets you check and reproduce it.

**Question.** Using only what an adult can answer and a nurse can measure with a
scale and a tape, which way of choosing people for an HbA1c test finds the most
undiagnosed dysglycaemia per test on a later survey: the ADA risk test, the
USPSTF 2021 rule, the ADA 2026 rule, or a model trained on 2007-2014 NHANES data?

**One row** is one US adult aged 20 or over, examined in NHANES, not pregnant,
with an HbA1c result, never told they have diabetes and taking no diabetes
medicine. **The outcome** is laboratory HbA1c of 5.7% or more (primary) and
6.5% or more.

**Data.** NHANES (CDC National Center for Health Statistics), six cycles from
2007-2008 to 2017-March 2020, public domain: 29,394 adults, of whom 18,456 are
used for training (2007-2014), 4,294 for calibration (2015-2016) and 6,644 for
the one-time test (2017-March 2020).

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

## Setup

Python 3.10 or newer (developed on 3.13), then:

```
pip install -r requirements.txt
```

Optional: Node.js for `scripts/toolchain_check.py`, and Playwright for
`scripts/prerender.py` (`pip install playwright` then
`python -m playwright install chromium`).

## Data access

NHANES is public domain and needs no account. `bash scripts/download_nhanes.sh`
downloads the 72 data files (12 components for each of 6 cycles) from
`https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/<year>/DataFiles/<FILE>.xpt`, plus
each file's codebook page, and checks every file against the committed
SHA-256 manifest `data/raw/manifest.csv`, which lists each file and its URL.
Raw and processed data are not stored in git (`data/` is ignored, except the
manifest).

- `--verify` checks an existing copy without downloading; `--update-manifest`
  rewrites the manifest.
- If the script cannot reach the CDC, download the files listed in the
  manifest by hand into `data/raw/<cycle>/` and run `--verify`.

## How to run

There are no notebooks: every step is a script, run in this order from the
repository root.

```
bash scripts/download_nhanes.sh        # 72 files from wwwn.cdc.gov + codebooks, checked against data/raw/manifest.csv
python scripts/toolchain_check.py      # optional: model export to JavaScript works (needs Node.js)
python -m src.harmonise                # one recoded row per participant; seals the test labels
python -m src.cohort                   # cohort rules, counts per cycle
python -m src.explore_week1            # week-1 data checks -> results/week1/
python -m src.scores                   # baseline flag rates and AUC on 2007-2016
python -m tests.test_scores            # unit tests for the ADA points and both rules
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

The test has been run: `results/test/UNSEALED.txt` is committed as the record,
so `src.test_once` refuses to run again on a fresh clone unless given
`--force`. With the same data it reproduces `results/test/` exactly. The
2017-March 2020 HbA1c values live only in
`data/processed/test_labels.sealed.parquet`. `src.harmonise` writes their
content hash to `results/week1/sealed_labels_hash.txt`, and `src.test_once`
refuses to read them unless that hash matches the one recorded in `PREREG.md`.

## Where to check each claim on the website

| Website section | Files that produce or record it |
| --- | --- |
| Question and hypotheses | `PREREG.md` (fixed before modelling), `FROZEN.md` (fixed before the test) |
| Data, cohort flow, data notes | `scripts/download_nhanes.sh`, `data/raw/manifest.csv`, `src/harmonise.py`, `src/cohort.py`, `results/week1/` |
| Checks against published numbers | `results/week1/ada_auc_dev.csv`, `results/week1/baseline_flag_rates.csv` |
| Preprocessing and inputs | `src/harmonise.py`, `src/models.py` (`prepare`, `FEATURES`), `docs/decisions.md` |
| Baselines and methods | `src/scores.py`, `tests/test_scores.py`, `src/models.py`, `src/fit.py`, `models/frozen.json`, `results/validation/tuning_grid.csv`, `results/review/grid_edge.csv` |
| Splits, metrics, bootstrap | `splits.json`, `src/config.py`, `src/evaluate.py` |
| Out-of-time validation | `results/validation/validation_by_fold.csv`, `results/validation/validation_mean.csv` |
| Leakage ladder | `src/ladder.py`, `results/validation/leakage_ladder.csv` (frozen), `results/review/leakage_ladder_summary.csv` |
| Results, H1-H5, yield slider | `src/test_once.py`, `results/test/`, `scripts/recheck_h1.py` |
| Same number of tests; where the models did worse | `src/review_extras.py`, `results/review/matched_volume.csv`, `results/review/subgroup_differences.csv` |
| H4 weighted-training check | `src/dev_checks.py`, `results/review/weighted_training.csv` |
| Who the model misses | `results/test/missed_profile.csv` |
| Calculator | `docs/data/ebm_y57.json`, `docs/data/ebm_y65.json` (from `src/export_web.py`), `docs/assets/app.js` |

## Layout

```
scripts/   download_nhanes.sh  toolchain_check.py  recheck_h1.py  prerender.py
src/       config.py  harmonise.py  cohort.py  scores.py  explore_week1.py
           models.py  fit.py  evaluate.py  ladder.py  test_once.py  export_web.py
           review_extras.py  dev_checks.py   (added after the test)
models/    frozen models and frozen.json (settings, thresholds, hashes)
tests/     test_scores.py
docs/      the website (GitHub Pages) and decisions.md, a dated log of every design decision
results/   week1/       data checks before any model
           validation/  tuning grid, forward-check validation, frozen leakage ladder
           test/        the one-time test on 2017-March 2020
           review/      analyses added after the test, following a review
data/      raw/ codebooks/ processed/   (not in git, except data/raw/manifest.csv)
```

## AI-use disclosure

This project was built with extensive help from Claude, an AI assistant made by
Anthropic, used through the Claude app from October to November 2026.

- Claude generated and ranked candidate project ideas, and researched the
  screening guidelines, related studies and NHANES documentation for the
  proposal and the plan.
- Claude wrote most of the code in `src/`, `scripts/` and `tests/` and the
  website in `docs/`, and ran the pipeline: the data download (on the author's
  computer, through the Claude desktop app), tuning, the one-time test, the
  post-test analyses and the independent recheck.
- Claude drafted `PREREG.md`, `FROZEN.md`, this README, `docs/decisions.md` and
  the website text, and made the changes after the external review of
  14 November 2026.
- Aditya Priyadarshi chose the project, approved the plan and each stage before
  it ran, made every git commit and push, obtained the external review and
  decided which changes to adopt, and is responsible for everything submitted.

Every number on the website comes from the result files in this repository,
which the code here produces.

## References

Data
- Centers for Disease Control and Prevention, National Center for Health
  Statistics. National Health and Nutrition Examination Survey data, 2007-2008
  to 2017-March 2020. https://wwwn.cdc.gov/nchs/nhanes/
- Stierman B, Afful J, Carroll MD, et al. National Health and Nutrition
  Examination Survey 2017-March 2020 prepandemic data files: development of
  files and prevalence estimates for selected health outcomes. National Health
  Statistics Reports no. 158. NCHS; 2021. doi:10.15620/cdc:106273

Screening rules and the baseline
- US Preventive Services Task Force. Screening for prediabetes and type 2
  diabetes: recommendation statement. JAMA. 2021;326(8):736-743.
  doi:10.1001/jama.2021.12531
- American Diabetes Association Professional Practice Committee. 2. Diagnosis
  and classification of diabetes: Standards of Care in Diabetes 2026. Diabetes
  Care. 2026;49(Suppl 1). doi:10.2337/dc26-S002
- Bang H, Edwards AM, Bomback AS, et al. Development and validation of a
  patient self-assessment score for diabetes risk. Ann Intern Med.
  2009;151(11):775-783.

Studies used as checks or context
- Poltavskiy E, Kim DJ, Bang H. Comparison of screening scores for diabetes and
  prediabetes. Diabetes Res Clin Pract. 2016;118:146-153.
  doi:10.1016/j.diabres.2016.06.022
- O'Brien MJ, Zhang Y, Bailey SC, et al. Screening for prediabetes and
  diabetes: clinical performance and implications for health equity. Am J Prev
  Med. 2023;64(6):814-823. doi:10.1016/j.amepre.2023.01.007
- Bergenstal RM, Gal RL, Connor CG, et al. Racial differences in the
  relationship of glucose concentrations and hemoglobin A1c levels. Ann Intern
  Med. 2017;167(2):95-102.
- Sun et al. 2026. arXiv:2606.16056. https://arxiv.org/abs/2606.16056
- Heshmat Feizabadi N, Tahmasbi V, Yeganeh A, Shongwe SC. Explainable machine
  learning for diabetes screening using NHANES 2021-2023 data: a
  leakage-controlled classification study. Math Comput Appl. 2026;31(5):221.
  doi:10.3390/mca31050221

Methods
- Tibshirani R. Regression shrinkage and selection via the lasso. J R Stat Soc
  B. 1996;58(1):267-288.
- Lou Y, Caruana R, Gehrke J, Hooker G. Accurate intelligible models with
  pairwise interactions. Proceedings of KDD 2013.
- Nori H, Jenkins S, Koch P, Caruana R. InterpretML: a unified framework for
  machine learning interpretability. arXiv:1909.09223; 2019.
- Ke G, Meng Q, Finley T, et al. LightGBM: a highly efficient gradient boosting
  decision tree. Advances in Neural Information Processing Systems 30 (2017).
- Peacock JL, Sauzet O, Ewings SM, Kerry SM. Dichotomising continuous data while
  retaining statistical power using a distributional approach. Stat Med.
  2012;31(26):3089-3103.
- Platt J. Probabilistic outputs for support vector machines and comparisons to
  regularized likelihood methods. In: Advances in Large Margin Classifiers. MIT
  Press; 1999.
- Rao JNK, Wu CFJ. Resampling inference with complex survey data. J Am Stat
  Assoc. 1988;83(401):231-241.
- Pedregosa F, et al. Scikit-learn: machine learning in Python. J Mach Learn
  Res. 2011;12:2825-2830.

Software: pandas, NumPy, SciPy, scikit-learn, LightGBM, interpret-core, PyArrow,
Matplotlib, Plotly.js 2.35.2 and Playwright.
