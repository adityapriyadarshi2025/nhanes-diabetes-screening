import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import CALIB_CYCLE, DEV_CYCLES, PROCESSED, RESULTS, ROOT, TRAIN_CYCLES
from .evaluate import auc
from .models import FEATURES, LGB_BASE, prepare, split

SEED = 2026
SEEDS = range(1, 21)
MONO = ("age", "bmi", "waist", "whtr")


def lgb_params(features, params, seed):
    return {**LGB_BASE, **params, "objective": "binary", "seed": seed,
            "monotone_constraints": [1 if f in MONO else 0 for f in features]}


def score(b, test, features, label):
    return float(auc(b.predict(test[features]), test[label].to_numpy(), test["wt_mec"].to_numpy())[0])


def fit_fixed(train, test, features, label, params, rounds):
    b = lgb.train(lgb_params(features, params, SEED), lgb.Dataset(train[features], train[label]), rounds)
    return score(b, test, features, label), rounds


def fit_inner(train, test, features, label, params, seed):
    """Rounds come from early stopping on 20% of the training rows, never from the rows being scored."""
    p = lgb_params(features, params, seed)
    keep = np.random.default_rng(seed).random(len(train)) < 0.8
    a, v = train[keep], train[~keep]
    es = lgb.train(p, lgb.Dataset(a[features], a[label]), 5000, valid_sets=[lgb.Dataset(v[features], v[label])],
                   callbacks=[lgb.early_stopping(100, verbose=False)])
    b = lgb.train(p, lgb.Dataset(train[features], train[label]), es.best_iteration)
    return score(b, test, features, label), es.best_iteration


def random_split(d, seed):
    mask = np.random.default_rng(seed).random(len(d)) < 0.8
    return d[mask].reset_index(drop=True), d[~mask].reset_index(drop=True)


def load():
    cohort = prepare(pd.read_parquet(PROCESSED / "cohort_dev.parquet"))
    adults = prepare(pd.read_parquet(PROCESSED / "examined_adults_dev.parquet"))
    adults["y57"] = (adults["hba1c"] >= 5.7).astype(int)
    adults["known"] = ((adults["dx_answer"] == 1) | (adults["insulin_now"] == 1) | (adults["dm_pills_now"] == 1)).astype(int)
    adults["y57_or_known"] = (adults["y57"] | adults["known"]).astype(int)
    return cohort, adults[adults["dx_answer"].isin([1, 2, 3])].reset_index(drop=True)


def steps(cohort, adults, seed):
    extra = FEATURES + ["told_prediabetes", "blood_test_3y"]
    tr, te = random_split(split(adults, DEV_CYCLES), seed)
    return [
        ("1. Honest: train 2007-2014, test 2015-2016, undiagnosed adults, no lab values",
         split(cohort, TRAIN_CYCLES), split(cohort, CALIB_CYCLE), FEATURES, "y57"),
        ("2. Random 80/20 split of 2007-2016 instead of a later survey",
         *random_split(split(cohort, DEV_CYCLES), seed), FEATURES, "y57"),
        ("3. Diagnosed and treated adults added as positives", tr, te, FEATURES, "y57_or_known"),
        ("4. Add 'told prediabetes' and 'blood test in the past 3 years'", tr, te, extra, "y57_or_known"),
        ("5. Add one blood value (HDL cholesterol)", tr, te, extra + ["hdl"], "y57_or_known"),
    ]


def main():
    frozen = json.loads((ROOT / "models" / "frozen.json").read_text())
    params = frozen["chosen"]["lgb"]["y57"]
    cohort, adults = load()
    if "--original" in sys.argv:
        rounds = lgb.Booster(model_file=str(ROOT / "models" / "lgb_y57.txt")).num_trees()
        out = pd.DataFrame([(name, fit_fixed(tr, te, f, y, params, rounds)[0]) for name, tr, te, f, y in steps(cohort, adults, SEED)],
                           columns=["step", "auc"])
        out.round(4).to_csv(RESULTS / "validation" / "leakage_ladder.csv", index=False)
        print(out.round(3).to_string(index=False))
        return
    rows = []
    for seed in SEEDS:
        for name, tr, te, f, y in steps(cohort, adults, seed):
            a, rounds = fit_inner(tr, te, f, y, params, seed)
            rows.append({"seed": seed, "step": name, "auc": a, "rounds": rounds})
        print("seed", seed, "done", flush=True)
    per_seed = pd.DataFrame(rows)
    summary = per_seed.groupby("step", sort=False)["auc"].agg(["mean", "std", "min", "max"]).reset_index()
    out = RESULTS / "review"
    out.mkdir(parents=True, exist_ok=True)
    per_seed.round(4).to_csv(out / "leakage_ladder_seeds.csv", index=False)
    summary.round(4).to_csv(out / "leakage_ladder_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
