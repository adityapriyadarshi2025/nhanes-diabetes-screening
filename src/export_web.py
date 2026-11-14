import json
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import RESULTS, ROOT, TRAIN_CYCLES
from .fit import validation_table
from .models import CUTS, load_dev, split

SITE = ROOT / "docs" / "data"
MODELS = ROOT / "models"


def ebm_json(model, platt):
    ebm = model.m
    out = {"intercept": float(np.ravel(ebm.intercept_)[0]), "features": list(ebm.feature_names_in_),
           "platt": platt, "terms": []}
    for feats, scores in zip(ebm.term_features_, ebm.term_scores_):
        cuts = []
        for f in feats:
            levels = ebm.bins_[f]
            cuts.append(np.asarray(levels[min(len(levels), len(feats)) - 1]).round(6).tolist())
        out["terms"].append({"name": " x ".join(ebm.feature_names_in_[f] for f in feats),
                             "features": [int(f) for f in feats], "cuts": cuts,
                             "scores": np.asarray(scores).round(6).tolist()})
    return out


def fill_rules(dev):
    tr = split(dev, TRAIN_CYCLES)
    rules = {}
    for target, cols in {"waist": ["bmi", "male", "age"], "arm": ["bmi", "male", "age"], "leg": ["height", "male", "age"]}.items():
        d = tr[[target] + cols].dropna()
        X = np.c_[np.ones(len(d)), d[cols].to_numpy()]
        beta = np.linalg.lstsq(X, d[target].to_numpy(), rcond=None)[0]
        rules[target] = {"intercept": float(beta[0]), "coef": dict(zip(cols, map(float, beta[1:])))}
    medians = {c: float(tr[c].median()) for c in ("educ", "pir", "mvpa_min_wk", "sedentary_min")}
    return rules, medians


def train_prevalence(dev):
    tr = split(dev, TRAIN_CYCLES)
    return {t: {"unweighted": float((tr["hba1c"] >= c).mean()),
                "weighted": float(np.average(tr["hba1c"] >= c, weights=tr["wt_mec"]))} for t, c in CUTS.items()}


def lgb_gain():
    b = lgb.Booster(model_file=str(MODELS / "lgb_y57.txt"))
    gain = b.feature_importance("gain")
    return dict(sorted(((n, float(g / gain.sum())) for n, g in zip(b.feature_name(), gain)), key=lambda t: -t[1]))


def records(path):
    return json.loads(pd.read_csv(path).to_json(orient="records")) if path.exists() else None


def main():
    SITE.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((MODELS / "frozen.json").read_text())
    dev = load_dev()
    rules, medians = fill_rules(dev)
    for t in ("y57", "y65"):
        m = pickle.loads((MODELS / f"ebm_{t}.pkl").read_bytes())
        out = ebm_json(m, frozen["calibration"]["platt"][f"ebm|{t}"])
        out.update({"fill": rules, "median": medians})
        (SITE / f"ebm_{t}.json").write_text(json.dumps(out))
    preds = pd.read_parquet(ROOT / "data" / "processed" / "fold_predictions.parquet")
    val = validation_table(preds, dev)
    val_mean = val.groupby(["target", "model"])[["auc", "pr_auc", "log_loss", "obs_exp", "sens_at_ada_rate"]].mean().reset_index()
    flow = pd.read_csv(RESULTS / "week1" / "cohort_flow.csv", index_col=0)
    test = RESULTS / "test"
    data = {
        "flow": {"steps": list(flow.index), "cycles": list(flow.columns),
                 "counts": flow.fillna(0).astype(int).values.tolist()},
        "outcome_by_cycle_dev": records(RESULTS / "week1" / "outcome_by_cycle_dev.csv"),
        "validation": json.loads(val.to_json(orient="records")),
        "validation_mean": json.loads(val_mean.to_json(orient="records")),
        "leakage_ladder": records(RESULTS / "validation" / "leakage_ladder.csv"),
        "train_prevalence": train_prevalence(dev),
        "frozen": {"chosen": frozen["chosen"], "rule_rates": frozen["calibration"]["rule_rates_2015_16"],
                   "thresholds": frozen["calibration"]["thresholds"], "frozen_at": frozen["frozen_at"]},
        "main_table": records(test / "main_table.csv"),
        "discrimination": records(test / "discrimination.csv"),
        "subgroups": records(test / "subgroups.csv"),
        "calibration_by_cycle": records(test / "calibration_by_cycle.csv"),
        "hba1c_by_cycle": records(test / "hba1c_by_cycle.csv"),
        "missed_profile": records(test / "missed_profile.csv"),
        "hypotheses": json.loads((test / "hypotheses.json").read_text()) if (test / "hypotheses.json").exists() else None,
        "unsealed": (test / "UNSEALED.txt").read_text().strip() if (test / "UNSEALED.txt").exists() else None,
        "lgb_gain": lgb_gain(),
        "review": {name: records(RESULTS / "review" / f"{name}.csv") for name in
                   ("matched_volume", "subgroup_differences", "leakage_ladder_summary", "weighted_training", "grid_edge")},
    }
    (SITE / "results.json").write_text(json.dumps(data))
    if (test / "yield_curves.json").exists():
        (SITE / "yield_curves.json").write_text((test / "yield_curves.json").read_text())
    print("wrote", sorted(p.name for p in SITE.iterdir()))


if __name__ == "__main__":
    main()
