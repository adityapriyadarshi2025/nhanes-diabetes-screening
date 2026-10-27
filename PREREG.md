# Pre-registration: analysis plan fixed before modelling

Status: **final**, 19 Oct 2026. Written after the week-1 data checks on
2007-2016 and before any model was fitted; the 2017-March 2020 outcomes were
never read. The git commit of this file is the time stamp. Any later change is
listed under "Changes" with its date and reason.

## 1. Question and hypotheses

Using only what an adult can answer and a nurse can measure with a scale and a
tape, which way of choosing people for an HbA1c test finds the most undiagnosed
dysglycaemia per test on a later survey?

| | Claim | Test on 2017-March 2020 |
| --- | --- | --- |
| H1 (primary) | At the ADA risk test's flag rate, the LightGBM classifier finds more adults with HbA1c 5.7% or more | Paired difference in survey-weighted sensitivity; 95% Rao-Wu bootstrap interval excludes 0 |
| H2 | For HbA1c 6.5% or more, the HbA1c value model (B) beats the classifier (A) | Difference in PR-AUC and in sensitivity at the ADA flag rate, with intervals |
| H3 | Any gain can be split into new weights, more inputs, non-linear shapes and interactions | Differences between neighbouring rungs of the methods ladder |
| H4 | Calibration drifts across survey years | Observed/expected ratio and calibration slope per cycle |
| H5 | For Asian adults, BMI-25 rules miss more cases than BMI-23 variants and the models | Subgroup sensitivity with intervals |

Only H1 is confirmatory; the rest are reported with intervals as secondary.

## 2. Data and cohort

NHANES 2007-2008 to 2017-March 2020 (six cycles), files and checksums in
`data/raw/manifest.csv`. One row is one adult who is aged 20 or over, examined,
not pregnant, has an HbA1c result, answered "no" or "borderline" to "ever told
you have diabetes" and takes neither insulin nor diabetes pills
(`src/cohort.py`; counts per step in `results/week1/cohort_flow.csv`).

| | Train 2007-2014 | Calibrate 2015-2016 | Test 2017-March 2020 |
| --- | --- | --- | --- |
| Adults in the cohort | 18,456 | 4,294 | 6,644 |
| HbA1c 5.7% or more | 5,902 | 1,440 | sealed |
| HbA1c 6.5% or more | 627 | 127 | sealed |

Unweighted counts. Per-cycle counts after every step:
`results/week1/cohort_flow.csv`.

## 3. Targets

`y57`: HbA1c 5.7% or more (primary). `y65`: HbA1c 6.5% or more. Model B
predicts log HbA1c; HbA1c is reported to 0.1, so "c or more" is read as an
underlying value of c - 0.05 or more.

## 4. Inputs

Allowed: age, sex, BMI, waist, waist-to-height ratio, height, weight, arm
circumference, upper-leg length, told high blood pressure, blood-pressure
medicine, family history of diabetes, heart disease or stroke, gestational
diabetes, smoking, leisure and total activity, sedentary minutes, education,
income-to-poverty ratio. Race and ethnicity (`race1`) only in the pre-specified
secondary model.

Never inputs: diagnosis answers, told prediabetes (`DIQ160`), blood test in the
past 3 years (`DIQ180`), diabetes medicines, any blood or urine measurement,
measured blood pressure, survey weights, and the cycle itself.

## 5. Baselines (`src/scores.py`, unit tests in `tests/test_scores.py`)

- ADA/CDC risk test, flag at 5 or more points. "Physically active" = any
  leisure-time moderate or vigorous activity (`active_rec`); a missing answer
  adds no points. Variants: BMI 23 for Asian adults; self-reported BMI;
  any activity.
- USPSTF 2021: age 35-70 and BMI 25 or more (variant: 23 for Asian adults).
- ADA 2026 (Table 2.5): age 35+, or BMI 25+ (23+ Asian) with one risk factor
  (family history, any group other than non-Hispanic White, heart disease or
  stroke, told hypertension or on treatment, inactivity, BMI 40+), or past
  gestational diabetes. Lab-based and unrecorded risk factors are left out.

## 6. Models and settings fixed now

| Rung | Model | Inputs |
| --- | --- | --- |
| 2 | Refitted ADA: logistic regression on the 7 ADA items | ADA items |
| 3 | L1 logistic regression, 10 penalties on a log grid; median imputation + missing flags | all allowed |
| 4 | EBM, no interactions | all allowed |
| 5 | EBM, up to 10 pairwise interactions | all allowed |
| 6 | LightGBM classifier (A), one per target | all allowed |
| 7 | LightGBM value model (B): mean and scale of log HbA1c + empirical residual distribution | all allowed |

LightGBM grid (27 settings): leaves 15, 31, 63; minimum leaf size 50, 100,
200; L2 penalty 0, 1, 5; learning rate 0.03; feature fraction 0.8; early
stopping after 100 rounds; risk never decreasing in age, BMI, waist and
waist-to-height (monotone method "intermediate"). EBM: 256 bins, 8 outer bags.
Training is unweighted. Calibration: Platt scaling fitted on 2015-2016
(isotonic as a check).

## 7. Splits (`splits.json`)

Tuning by rolling origin: train 2007-2010 / check 2011-2012; train 2007-2012 /
check 2013-2014; train 2007-2014 / check 2015-2016. The setting with the best
mean log-loss over the three checks wins. Final models train on 2007-2014;
2015-2016 sets calibration and every threshold; 2017-March 2020 is used once.

## 8. Metrics

Primary: survey-weighted sensitivity and PPV at matched flag rates. The share
flagged by the ADA test, the USPSTF rule and the ADA 2026 rule is measured on
2015-2016; each model's threshold flags the same weighted share there and is
applied unchanged to the test cycle. Also reported: matching on the test cycle
itself, yield curves, tests per case found, ROC-AUC, PR-AUC, Brier score,
calibration intercept and slope, observed/expected ratio by cycle.

Intervals: 1,000 Rao-Wu bootstrap replicates resampling PSUs within strata,
the same replicate for both members of a pair. Unweighted figures sit beside
weighted ones.

Subgroups: age 20-34, 35-49, 50-69, 70+; sex; race and ethnicity
(`race3`, incl. Asian adults); BMI under 23, 23-25, 25-30, 30+.

## 9. Secondary analyses (pre-specified)

Leakage ladder on 2007-2016 only; G1-only LightGBM; with-race model; ADA test
variants above; dropping "borderline" answers; survey-weighted training;
adversarial validation for input drift; fasting-glucose label check
(2015-2016 only before the freeze).

## 10. Test-set protocol

The 2017-March 2020 HbA1c and fasting glucose live only in
`data/processed/test_labels.sealed.parquet` (15,560 rows; content SHA-256
`71e17e0dd189a218c4ce038a1799f857d56f9b9c441ac6fcacb935e21a6b4962`, file
SHA-256 `e593a2fe9e66e083c7e08109a56803a9fa4e3fe60a1ed3e7655f3712a3879bd4`;
see `results/week1/sealed_labels_hash.txt`). Before it is
read, `FROZEN.md` records the commit hash, model files and thresholds. The
test is run once; later bug fixes are documented, rerun in full and noted on
the site.

## Changes

None since the final version of 19 Oct 2026.
