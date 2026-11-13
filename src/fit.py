import hashlib
import json
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import CALIB_CYCLE, RESULTS, ROOT
from .evaluate import auc, average_precision, calibration, logistic_fit, logit, observed_expected, threshold_for_rate
from .models import (C_GRID, CUTS, FEATURES, FOLDS, LGB_BASE, LGB_GRID, AdaRefit, EBM, L1Logistic, LGBClassifier,
                     ValueModel, load_dev, log_loss, split)

OUT = RESULTS / "validation"
MODELS = ROOT / "models"
RULES = {"ada": "ada_flag", "uspstf": "uspstf_flag", "ada2026": "ada2026_flag"}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def tune_l1(dev, target):
    rows = []
    for C in C_GRID:
        losses = []
        for train, check in FOLDS:
            tr, ck = split(dev, train), split(dev, check)
            losses.append(log_loss(ck[target].values, L1Logistic(C).fit(tr, tr[target]).prob(ck)))
        rows.append({"target": target, "C": C, "mean_log_loss": np.mean(losses)})
    return pd.DataFrame(rows)


def tune_lgb(dev, target):
    rows = []
    for params in LGB_GRID:
        losses, rounds = [], []
        for train, check in FOLDS:
            tr, ck = split(dev, train), split(dev, check)
            m = LGBClassifier(params).fit(tr, tr[target], valid=(ck, ck[target]))
            losses.append(log_loss(ck[target].values, m.prob(ck)))
            rounds.append(m.rounds)
        rows.append({"target": target, **params, "mean_log_loss": np.mean(losses), "rounds": rounds})
    return pd.DataFrame(rows)


def tune_value(dev):
    rows = []
    for params in LGB_GRID:
        errors = []
        for train, check in FOLDS:
            tr, ck = split(dev, train), split(dev, check)
            p = {**LGB_BASE, **params, "objective": "regression"}
            b = lgb.train(p, lgb.Dataset(tr[FEATURES], np.log(tr["hba1c"])), 5000,
                          valid_sets=[lgb.Dataset(ck[FEATURES], np.log(ck["hba1c"]))],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            pred = b.predict(ck[FEATURES], num_iteration=b.best_iteration)
            errors.append(np.mean((np.log(ck["hba1c"]) - pred) ** 2))
        rows.append({**params, "mean_mse": np.mean(errors)})
    return pd.DataFrame(rows)


def best(table, col):
    row = table.sort_values(col).iloc[0]
    return {k: (int(row[k]) if k != "lambda_l2" else float(row[k])) for k in ("num_leaves", "min_data_in_leaf", "lambda_l2")}


def builders(chosen):
    return {
        "ada_refit": lambda t: AdaRefit(),
        "l1": lambda t: L1Logistic(chosen["l1_C"][t]),
        "ebm": lambda t: EBM(0),
        "ebm_pairs": lambda t: EBM(10),
        "lgb": lambda t: LGBClassifier(chosen["lgb"][t]),
    }


def fold_models(dev, chosen):
    preds, final = [], {}
    for i, (train, check) in enumerate(FOLDS):
        tr, ck = split(dev, train), split(dev, check)
        base = ck[["SEQN", "cycle", "wt_mec", "strata", "psu", "y57", "y65", "ada_points"]].copy()
        for target in CUTS:
            for name, make in builders(chosen).items():
                t0 = time.time()
                m = make(target).fit(tr, tr[target], valid=(ck, ck[target]))
                preds.append(base.assign(fold=i + 1, model=name, target=target, p=m.prob(ck)))
                if i == len(FOLDS) - 1:
                    final[(name, target)] = m
                log(f"fold {i + 1} {target} {name} {time.time() - t0:.0f}s")
        vm = ValueModel(chosen["value"]).fit(tr, valid=(ck, None))
        for target in CUTS:
            preds.append(base.assign(fold=i + 1, model="lgb_value", target=target, p=vm.prob(ck, target)))
            if i == len(FOLDS) - 1:
                final[("lgb_value", target)] = vm
        log(f"fold {i + 1} value model")
    return pd.concat(preds, ignore_index=True), final


def validation_table(preds, dev):
    rows = []
    for (fold, model, target), g in preds.groupby(["fold", "model", "target"]):
        y, p, w = g[target].to_numpy(), g["p"].to_numpy(), g["wt_mec"].to_numpy()
        ck = dev[dev["SEQN"].isin(g["SEQN"])].set_index("SEQN").loc[g["SEQN"]]
        rate = np.average(ck["ada_flag"], weights=w)
        thr = threshold_for_rate(p, w, rate)
        sens = np.sum(w * y * (p >= thr)) / np.sum(w * y)
        rows.append({"fold": fold, "check_cycle": g["cycle"].iloc[0], "model": model, "target": target,
                     "auc": auc(p, y, w)[0], "pr_auc": average_precision(p, y, w)[0], "log_loss": log_loss(y, p),
                     "obs_exp": observed_expected(p, y, w)[0], "sens_at_ada_rate": sens})
    for (fold, target), g in preds[preds["model"] == "lgb"].groupby(["fold", "target"]):
        y, w = g[target].to_numpy(), g["wt_mec"].to_numpy()
        ck = dev[dev["SEQN"].isin(g["SEQN"])].set_index("SEQN").loc[g["SEQN"]]
        flag = ck["ada_flag"].to_numpy()
        rows.append({"fold": fold, "check_cycle": g["cycle"].iloc[0], "model": "ada_points", "target": target,
                     "auc": auc(ck["ada_points"].to_numpy(float), y, w)[0],
                     "pr_auc": average_precision(ck["ada_points"].to_numpy(float), y, w)[0],
                     "sens_at_ada_rate": np.sum(w * y * flag) / np.sum(w * y)})
    return pd.DataFrame(rows)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(final, chosen, calib, dev):
    MODELS.mkdir(exist_ok=True)
    files = {}
    for (name, target), m in final.items():
        if name == "lgb":
            path = MODELS / f"lgb_{target}.txt"
            m.b.save_model(str(path), num_iteration=m.rounds)
        elif name == "lgb_value":
            if target != "y57":
                continue
            m.mu.save_model(str(MODELS / "value_mu.txt"))
            m.sigma.save_model(str(MODELS / "value_sigma.txt"))
            np.save(MODELS / "value_z.npy", m.z)
            path = MODELS / "value_z.npy"
            files["value_mu.txt"] = sha256(MODELS / "value_mu.txt")
            files["value_sigma.txt"] = sha256(MODELS / "value_sigma.txt")
        else:
            path = MODELS / f"{name}_{target}.pkl"
            path.write_bytes(pickle.dumps(m))
        files[path.name] = sha256(path)
    frozen = {"features": FEATURES, "chosen": chosen, "calibration": calib, "model_files": files,
              "code_files": {p.name: sha256(p) for p in sorted((ROOT / "src").glob("*.py"))},
              "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (MODELS / "frozen.json").write_text(json.dumps(frozen, indent=2))
    return frozen


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dev = load_dev()
    chosen = {"l1_C": {}, "lgb": {}}
    grids = []
    for target in CUTS:
        t = tune_l1(dev, target)
        chosen["l1_C"][target] = float(t.sort_values("mean_log_loss").iloc[0]["C"])
        grids.append(t.assign(model="l1"))
        log(f"l1 {target} C={chosen['l1_C'][target]:.4g}")
        t = tune_lgb(dev, target)
        chosen["lgb"][target] = best(t, "mean_log_loss")
        grids.append(t.assign(model="lgb"))
        log(f"lgb {target} {chosen['lgb'][target]}")
    t = tune_value(dev)
    chosen["value"] = best(t, "mean_mse")
    grids.append(t.assign(model="lgb_value"))
    log(f"value {chosen['value']}")
    pd.concat(grids, ignore_index=True).to_csv(OUT / "tuning_grid.csv", index=False)

    preds, final = fold_models(dev, chosen)
    preds.to_parquet(MODELS.parent / "data" / "processed" / "fold_predictions.parquet", index=False)
    table = validation_table(preds, dev)
    table.round(4).to_csv(OUT / "validation_by_fold.csv", index=False)
    summary = table.groupby(["target", "model"])[["auc", "pr_auc", "log_loss", "obs_exp", "sens_at_ada_rate"]].mean()
    summary.round(4).to_csv(OUT / "validation_mean.csv")
    print(summary.round(3).to_string())

    cal = split(dev, CALIB_CYCLE)
    last = preds[preds["fold"] == len(FOLDS)]
    rates = {k: float(np.average(cal[v], weights=cal["wt_mec"])) for k, v in RULES.items()}
    calib = {"rule_rates_2015_16": rates, "platt": {}, "thresholds": {}}
    for (model, target), g in last.groupby(["model", "target"]):
        y, p, w = g[target].to_numpy(float), g["p"].to_numpy(), g["wt_mec"].to_numpy()
        a, b = logistic_fit(np.c_[np.ones(len(p)), logit(p)], y, w)
        pc = 1 / (1 + np.exp(-(a + b * logit(p))))
        key = f"{model}|{target}"
        calib["platt"][key] = [float(a), float(b)]
        calib["thresholds"][key] = {k: float(threshold_for_rate(pc, w, r)) for k, r in rates.items()}
    frozen = save(final, chosen, calib, dev)
    log(f"saved {len(frozen['model_files'])} model files")


if __name__ == "__main__":
    main()
