# Decisions log

Newest first. Each entry: what was decided and why.

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
