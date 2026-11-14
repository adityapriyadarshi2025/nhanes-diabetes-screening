import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import RESULTS, ROOT
from .evaluate import auc
from .models import CUTS, FEATURES, FOLDS, LGB_BASE, LGBClassifier, load_dev, log_loss, split

OUT = RESULTS / "review"
SEED = 2026
EDGE = {"y57": [dict(num_leaves=a, min_data_in_leaf=b, lambda_l2=c) for a in (4, 7, 15) for b in (25, 50) for c in (5, 10, 20)],
        "y65": [dict(num_leaves=a, min_data_in_leaf=b, lambda_l2=0) for a in (4, 7, 15) for b in (200, 400, 800)]}


def weighted_training(dev, chosen):
    """Same LightGBM settings with and without survey weights; observed/expected before any recalibration."""
    rows = []
    for t in CUTS:
        p = {**LGB_BASE, **chosen["lgb"][t], "objective": "binary"}
        for train, check in FOLDS:
            tr, ck = split(dev, train), split(dev, check)
            keep = np.random.default_rng(SEED).random(len(tr)) < 0.8
            for weighted in (False, True):
                wt = (tr["wt_mec"] / tr["wt_mec"].mean()).to_numpy() if weighted else np.ones(len(tr))
                a = lgb.Dataset(tr.loc[keep, FEATURES], tr.loc[keep, t], weight=wt[keep])
                v = lgb.Dataset(tr.loc[~keep, FEATURES], tr.loc[~keep, t], weight=wt[~keep])
                es = lgb.train(p, a, 5000, valid_sets=[v], callbacks=[lgb.early_stopping(100, verbose=False)])
                b = lgb.train(p, lgb.Dataset(tr[FEATURES], tr[t], weight=wt), es.best_iteration)
                pred, y, w = b.predict(ck[FEATURES]), ck[t].to_numpy(), ck["wt_mec"].to_numpy()
                rows.append({"target": t, "check": check, "weighted_training": weighted, "rounds": es.best_iteration,
                             "obs_exp": float((w * y).sum() / (w * pred).sum()), "auc": float(auc(pred, y, w)[0])})
    return pd.DataFrame(rows)


def grid_edge(dev, chosen):
    """Settings beyond the edge of the pre-registered grid, tuned exactly as in src/fit.py."""
    rows = []
    for t, grid in EDGE.items():
        for params in grid:
            losses = []
            for train, check in FOLDS:
                tr, ck = split(dev, train), split(dev, check)
                m = LGBClassifier(params).fit(tr, tr[t], valid=(ck, ck[t]))
                losses.append(log_loss(ck[t].values, m.prob(ck)))
            rows.append({"target": t, **params, "mean_log_loss": float(np.mean(losses)),
                         "frozen_setting": all(chosen["lgb"][t][k] == v for k, v in params.items())})
    return pd.DataFrame(rows)


def main():
    chosen = json.loads((ROOT / "models" / "frozen.json").read_text())["chosen"]
    dev = load_dev()
    OUT.mkdir(parents=True, exist_ok=True)
    wt = weighted_training(dev, chosen)
    wt.round(4).to_csv(OUT / "weighted_training.csv", index=False)
    print(wt.round(3).to_string(index=False), flush=True)
    ge = grid_edge(dev, chosen)
    ge.round(6).to_csv(OUT / "grid_edge.csv", index=False)
    print(ge.sort_values(["target", "mean_log_loss"]).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
