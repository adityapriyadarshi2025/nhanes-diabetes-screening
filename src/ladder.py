import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import CALIB_CYCLE, DEV_CYCLES, PROCESSED, RESULTS, ROOT, TRAIN_CYCLES
from .evaluate import auc
from .models import FEATURES, LGB_BASE, prepare, split

SEED = 2026


def fit_auc(train, test, features, label, params, rounds):
    b = lgb.train({**LGB_BASE, **params, "objective": "binary",
                   "monotone_constraints": [1 if f in ("age", "bmi", "waist", "whtr") else 0 for f in features]},
                  lgb.Dataset(train[features], train[label]), rounds)
    return float(auc(b.predict(test[features]), test[label].to_numpy(), test["wt_mec"].to_numpy())[0])


def random_split(d):
    rng = np.random.default_rng(SEED)
    mask = rng.random(len(d)) < 0.8
    return d[mask].reset_index(drop=True), d[~mask].reset_index(drop=True)


def main():
    frozen = json.loads((ROOT / "models" / "frozen.json").read_text())
    params = frozen["chosen"]["lgb"]["y57"]
    rounds = lgb.Booster(model_file=str(ROOT / "models" / "lgb_y57.txt")).num_trees()

    cohort = prepare(pd.read_parquet(PROCESSED / "cohort_dev.parquet"))
    adults = prepare(pd.read_parquet(PROCESSED / "examined_adults_dev.parquet"))
    adults["y57"] = (adults["hba1c"] >= 5.7).astype(int)
    adults["known"] = ((adults["dx_answer"] == 1) | (adults["insulin_now"] == 1) | (adults["dm_pills_now"] == 1)).astype(int)
    adults["y57_or_known"] = (adults["y57"] | adults["known"]).astype(int)
    adults = adults[adults["dx_answer"].isin([1, 2, 3])].reset_index(drop=True)

    steps = []
    tr, te = split(cohort, TRAIN_CYCLES), split(cohort, CALIB_CYCLE)
    steps.append(("1. Honest: train 2007-2014, test 2015-2016, undiagnosed adults, no lab values",
                  fit_auc(tr, te, FEATURES, "y57", params, rounds)))
    tr, te = random_split(split(cohort, DEV_CYCLES))
    steps.append(("2. Random 80/20 split of 2007-2016 instead of a later survey",
                  fit_auc(tr, te, FEATURES, "y57", params, rounds)))
    tr, te = random_split(split(adults, DEV_CYCLES))
    steps.append(("3. Diagnosed and treated adults added as positives",
                  fit_auc(tr, te, FEATURES, "y57_or_known", params, rounds)))
    extra = FEATURES + ["told_prediabetes", "blood_test_3y"]
    steps.append(("4. Add 'told prediabetes' and 'blood test in the past 3 years'",
                  fit_auc(tr, te, extra, "y57_or_known", params, rounds)))
    steps.append(("5. Add one blood value (HDL cholesterol)",
                  fit_auc(tr, te, extra + ["hdl"], "y57_or_known", params, rounds)))
    out = pd.DataFrame(steps, columns=["step", "auc"])
    (RESULTS / "validation").mkdir(parents=True, exist_ok=True)
    out.round(4).to_csv(RESULTS / "validation" / "leakage_ladder.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
