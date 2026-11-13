import json
import pickle

import numpy as np
import pandas as pd

from .config import RESULTS, ROOT

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


def records(path):
    return json.loads(pd.read_csv(path).to_json(orient="records")) if path.exists() else None


def main():
    SITE.mkdir(parents=True, exist_ok=True)
    frozen = json.loads((MODELS / "frozen.json").read_text())
    for t in ("y57", "y65"):
        m = pickle.loads((MODELS / f"ebm_{t}.pkl").read_bytes())
        (SITE / f"ebm_{t}.json").write_text(json.dumps(ebm_json(m, frozen["calibration"]["platt"][f"ebm|{t}"])))
    flow = pd.read_csv(RESULTS / "week1" / "cohort_flow.csv", index_col=0)
    test = RESULTS / "test"
    data = {
        "flow": {"steps": list(flow.index), "cycles": list(flow.columns),
                 "counts": flow.fillna(0).astype(int).values.tolist()},
        "outcome_by_cycle_dev": records(RESULTS / "week1" / "outcome_by_cycle_dev.csv"),
        "validation": records(RESULTS / "validation" / "validation_by_fold.csv"),
        "validation_mean": records(RESULTS / "validation" / "validation_mean.csv"),
        "leakage_ladder": records(RESULTS / "validation" / "leakage_ladder.csv"),
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
    }
    (SITE / "results.json").write_text(json.dumps(data))
    if (test / "yield_curves.json").exists():
        (SITE / "yield_curves.json").write_text((test / "yield_curves.json").read_text())
    print("wrote", sorted(p.name for p in SITE.iterdir()))


if __name__ == "__main__":
    main()
