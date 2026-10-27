"""The three baselines, computed from pre-test answers and body measurements only.

ada_points / ada_flag
    ADA/CDC Prediabetes Risk Test (Bang et al., Ann Intern Med 2009, as published
    by ADA and CDC): age 40-49 = 1, 50-59 = 2, 60+ = 3; man = 1; gestational
    diabetes = 1; parent or sibling with diabetes = 1; ever told high blood
    pressure = 1; not physically active = 1; BMI 25-29.9 = 1, 30-39.9 = 2,
    40+ = 3 (the test's weight chart is these BMI cut-offs). Flag: 5 or more.
uspstf_flag
    USPSTF 2021: age 35-70 and BMI 25 or more.
ada2026_flag
    ADA Standards of Care 2026, Table 2.5: everyone aged 35+; younger adults with
    BMI 25+ (23+ for Asian adults) and at least one risk factor; anyone with a
    history of gestational diabetes.

Pre-test only: the ADA 2026 risk factors that need a blood test (HDL,
triglycerides) and those NHANES does not record consistently (PCOS, HIV,
acanthosis nigricans, fatty liver) are left out, as is a known prediabetes
diagnosis (it reveals an earlier blood test). Measured blood pressure is not
used; hypertension means told by a doctor or on treatment.

A missing answer adds no points, as a "don't know" would on the paper test.

Run from the project folder:  python -m src.scores   (flag rates, and the ADA
score's AUC on the development cycles as a sanity check)
"""
import numpy as np
import pandas as pd

# "Physically active" for the ADA item: any leisure-time moderate or vigorous
# activity (PAQ650/PAQ665). Fixed before modelling; "active_any" is the
# sensitivity analysis.
ACTIVE = "active_rec"


def _asian(d):
    return d["race3"] == 6          # non-Hispanic Asian, recorded from 2011 on


def ada_points(d, active=ACTIVE, bmi="bmi", asian_bmi=False):
    """ADA risk test score, 0-11. asian_bmi: 1 point from BMI 23 for Asian adults."""
    age = d["age"]
    pts = np.select([age >= 60, age >= 50, age >= 40], [3, 2, 1], 0)
    pts = pts + (d["male"] == 1) + (d["gdm"] == 1) + (d["famhx"] == 1)
    pts = pts + (d["htn_told"] == 1) + (d[active] == 0)
    first = np.where(asian_bmi & _asian(d), 23.0, 25.0)
    b = d[bmi]
    pts = pts + np.select([b >= 40, b >= 30, b >= first], [3, 2, 1], 0)
    return pd.Series(np.asarray(pts, dtype=int), index=d.index)


def uspstf_flag(d, asian_bmi=False):
    first = np.where(asian_bmi & _asian(d), 23.0, 25.0)
    return (d["age"].between(35, 70) & (d["bmi"] >= first)).astype(int)


def ada2026_flag(d, active=ACTIVE):
    overweight = d["bmi"] >= np.where(_asian(d), 23.0, 25.0)
    high_risk_group = d["race1"] != 3          # everyone except non-Hispanic White
    risk_factor = ((d["famhx"] == 1) | high_risk_group | (d["cvd"] == 1)
                   | (d["htn_told"] == 1) | (d["bp_med"] == 1)
                   | (d[active] == 0) | (d["bmi"] >= 40))
    return ((d["age"] >= 35) | (overweight & risk_factor) | (d["gdm"] == 1)).astype(int)


def add_scores(d):
    """All baselines and their pre-specified variants, as new columns."""
    out = d.copy()
    out["ada_points"] = ada_points(d)
    out["ada_flag"] = (out["ada_points"] >= 5).astype(int)
    out["ada_points_asian23"] = ada_points(d, asian_bmi=True)
    out["ada_flag_asian23"] = (out["ada_points_asian23"] >= 5).astype(int)
    out["ada_points_selfbmi"] = ada_points(d, bmi="sr_bmi")
    out["ada_flag_selfbmi"] = (out["ada_points_selfbmi"] >= 5).astype(int)
    out["ada_points_anyactive"] = ada_points(d, active="active_any")
    out["uspstf_flag"] = uspstf_flag(d)
    out["uspstf_flag_asian23"] = uspstf_flag(d, asian_bmi=True)
    out["ada2026_flag"] = ada2026_flag(d)
    return out


def main():
    from sklearn.metrics import roc_auc_score
    from .config import DEV_CYCLES, PROCESSED, RESULTS

    dev = add_scores(pd.read_parquet(PROCESSED / "cohort_dev.parquet"))
    test = add_scores(pd.read_parquet(PROCESSED / "cohort_test_inputs.parquet"))
    both = pd.concat([dev, test], ignore_index=True)
    flags = ["ada_flag", "ada_flag_asian23", "ada_flag_selfbmi", "uspstf_flag",
             "uspstf_flag_asian23", "ada2026_flag"]
    rates = both.groupby("cycle").apply(
        lambda g: pd.Series({f: 100 * np.average(g[f], weights=g["wt_mec"]) for f in flags}))
    rates.round(1).to_csv(RESULTS / "week1" / "baseline_flag_rates.csv")
    print("Share flagged (% of cohort, survey-weighted):\n", rates.round(1).T.to_string())

    rows = []
    for cyc in DEV_CYCLES:
        g = dev[dev["cycle"] == cyc]
        for y in ("y57", "y65"):
            rows.append({"cycle": cyc, "target": y,
                         "auc_ada_points": roc_auc_score(g[y], g["ada_points"], sample_weight=g["wt_mec"]),
                         "auc_ada_selfbmi": roc_auc_score(g[y], g["ada_points_selfbmi"], sample_weight=g["wt_mec"])})
    auc = pd.DataFrame(rows)
    auc.round(3).to_csv(RESULTS / "week1" / "ada_auc_dev.csv", index=False)
    print("\nADA score AUC by development cycle (survey-weighted):\n", auc.round(3).to_string(index=False))
    print(f"\nItems missing in the cohort (count as 0 points): "
          f"BMI {both['bmi'].isna().sum()}, family history {both['famhx'].isna().sum()}, "
          f"activity {both[ACTIVE].isna().sum()}, gestational diabetes {both['gdm'].isna().sum()}")


if __name__ == "__main__":
    main()
