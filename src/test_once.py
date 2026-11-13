import hashlib
import json
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import DEV_CYCLES, PROCESSED, RESULTS, ROOT
from .evaluate import (auc, average_precision, brier, calibration, calibration_reps, ci, flag_stats, logit,
                       observed_expected, replicate_weights, threshold_for_rate)
from .models import CUTS, FEATURES, ValueModel, prepare

OUT = RESULTS / "test"
MODELS = ROOT / "models"
PREREG_CONTENT_HASH = "71e17e0dd189a218c4ce038a1799f857d56f9b9c441ac6fcacb935e21a6b4962"
NAMES = ["ada_refit", "l1", "ebm", "ebm_pairs", "lgb", "lgb_value"]
OPS = {"ada": "ada_flag", "uspstf": "uspstf_flag", "ada2026": "ada2026_flag"}
RULES = ["ada_flag", "ada_flag_asian23", "ada_flag_selfbmi", "uspstf_flag", "uspstf_flag_asian23", "ada2026_flag"]
LADDER = ["ada_refit", "l1", "ebm", "ebm_pairs", "lgb"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_models(frozen):
    for name, h in frozen["model_files"].items():
        if sha256(MODELS / name) != h:
            sys.exit(f"{name} changed after the freeze")
    vm = ValueModel(frozen["chosen"]["value"])
    vm.mu = lgb.Booster(model_file=str(MODELS / "value_mu.txt"))
    vm.sigma = lgb.Booster(model_file=str(MODELS / "value_sigma.txt"))
    vm.z = np.load(MODELS / "value_z.npy")
    models = {("lgb_value", t): vm for t in CUTS}
    for t in CUTS:
        for name in ["ada_refit", "l1", "ebm", "ebm_pairs"]:
            models[(name, t)] = pickle.loads((MODELS / f"{name}_{t}.pkl").read_bytes())
        models[("lgb", t)] = lgb.Booster(model_file=str(MODELS / f"lgb_{t}.txt"))
    return models


def unseal():
    labels = pd.read_parquet(PROCESSED / "test_labels.sealed.parquet").sort_values("SEQN")
    content = hashlib.sha256(labels.to_csv(index=False, float_format="%.6g").encode()).hexdigest()
    if content != PREREG_CONTENT_HASH:
        sys.exit("sealed labels do not match the hash in PREREG.md")
    return labels


def predict(models, frozen, d):
    raw, cal = {}, {}
    for (name, t), m in models.items():
        if name == "lgb":
            p = m.predict(d[FEATURES])
        elif name == "lgb_value":
            p = m.prob(d, t)
        else:
            p = m.prob(d)
        a, b = frozen["calibration"]["platt"][f"{name}|{t}"]
        raw[(name, t)], cal[(name, t)] = p, sigmoid(a + b * logit(p))
    return raw, cal


def stat_row(y, flag, w, W):
    point, reps = flag_stats(y, flag, w), flag_stats(y, flag, W)
    row = {}
    for k in ("flag_rate", "sens", "ppv"):
        row.update({k: point[k][0], f"{k}_lo": np.percentile(reps[k], 2.5), f"{k}_hi": np.percentile(reps[k], 97.5)})
    row["tests_per_case"] = 1 / point["ppv"][0]
    row["n_flagged"], row["cases_found"], row["cases"] = int(flag.sum()), int((flag * y).sum()), int(y.sum())
    return row, reps


def diff_cols(prefix, a, b, pa, pb):
    d = a - b
    return {f"{prefix}": pa - pb, f"{prefix}_lo": np.percentile(d, 2.5), f"{prefix}_hi": np.percentile(d, 97.5)}


def main_table(d, cal, frozen, w, W):
    rows = []
    for t in CUTS:
        y = d[t].to_numpy()
        for op, col in OPS.items():
            rule_flag = d[col].to_numpy()
            rrow, rreps = stat_row(y, rule_flag, w, W)
            rows.append({"target": t, "operating_point": op, "method": col, **rrow})
            test_rate = rrow["flag_rate"]
            for name in NAMES:
                p = cal[(name, t)]
                flag = (p >= frozen["calibration"]["thresholds"][f"{name}|{t}"][op]).astype(int)
                mrow, mreps = stat_row(y, flag, w, W)
                mrow.update(diff_cols("sens_diff", mreps["sens"], rreps["sens"], mrow["sens"], rrow["sens"]))
                mrow.update(diff_cols("ppv_diff", mreps["ppv"], rreps["ppv"], mrow["ppv"], rrow["ppv"]))
                same = (p >= threshold_for_rate(p, w, test_rate)).astype(int)
                mrow["sens_matched_on_test"] = flag_stats(y, same, w)["sens"][0]
                rows.append({"target": t, "operating_point": op, "method": name, **mrow})
    return pd.DataFrame(rows)


def discrimination(d, cal, w, W):
    rows = []
    for t in CUTS:
        y = d[t].to_numpy()
        scores = {name: cal[(name, t)] for name in NAMES}
        scores["ada_points"] = d["ada_points"].to_numpy(float)
        for name, s in scores.items():
            a, ap = auc(s, y, W), average_precision(s, y, W)
            row = {"target": t, "method": name,
                   "auc": auc(s, y, w)[0], "auc_lo": np.percentile(a, 2.5), "auc_hi": np.percentile(a, 97.5),
                   "pr_auc": average_precision(s, y, w)[0], "pr_auc_lo": np.percentile(ap, 2.5),
                   "pr_auc_hi": np.percentile(ap, 97.5)}
            if name != "ada_points":
                b, oe = brier(s, y, W), observed_expected(s, y, W)
                ci_, cs = calibration(s, y, w)
                creps = calibration_reps(s, y, W, n=200)
                row.update({"brier": brier(s, y, w)[0], "brier_lo": np.percentile(b, 2.5), "brier_hi": np.percentile(b, 97.5),
                            "obs_exp": observed_expected(s, y, w)[0], "obs_exp_lo": np.percentile(oe, 2.5),
                            "obs_exp_hi": np.percentile(oe, 97.5), "cal_intercept": ci_, "cal_slope": cs,
                            "cal_slope_lo": np.percentile(creps[:, 1], 2.5), "cal_slope_hi": np.percentile(creps[:, 1], 97.5)})
            rows.append(row)
    return pd.DataFrame(rows)


def hypotheses(d, cal, frozen, w, W, table):
    out = {}
    y = d["y57"].to_numpy()
    pick = lambda t, op, m: table[(table.target == t) & (table.operating_point == op) & (table.method == m)].iloc[0]
    for m in ("lgb", "ebm", "ebm_pairs"):
        r = pick("y57", "ada", m)
        out[f"H1_{m}"] = {"sens_model": r.sens, "sens_ada": pick("y57", "ada", "ada_flag").sens,
                          "diff": r.sens_diff, "lo": r.sens_diff_lo, "hi": r.sens_diff_hi,
                          "supported": bool(r.sens_diff_lo > 0)}
    y65 = d["y65"].to_numpy()
    pa = average_precision(cal[("lgb_value", "y65")], y65, W) - average_precision(cal[("lgb", "y65")], y65, W)
    point = (average_precision(cal[("lgb_value", "y65")], y65, w) - average_precision(cal[("lgb", "y65")], y65, w))[0]
    sens = {m: flag_stats(y65, (cal[(m, "y65")] >= frozen["calibration"]["thresholds"][f"{m}|y65"]["ada"]).astype(int), W)["sens"]
            for m in ("lgb_value", "lgb")}
    out["H2"] = {"pr_auc_diff": point, "pr_auc_lo": np.percentile(pa, 2.5), "pr_auc_hi": np.percentile(pa, 97.5),
                 "sens_diff": pick("y65", "ada", "lgb_value").sens - pick("y65", "ada", "lgb").sens,
                 "sens_lo": np.percentile(sens["lgb_value"] - sens["lgb"], 2.5),
                 "sens_hi": np.percentile(sens["lgb_value"] - sens["lgb"], 97.5)}
    steps = ["ada_flag"] + LADDER
    flags = {"ada_flag": d["ada_flag"].to_numpy()}
    for m in LADDER:
        flags[m] = (cal[(m, "y57")] >= frozen["calibration"]["thresholds"][f"{m}|y57"]["ada"]).astype(int)
    reps = {m: flag_stats(y, flags[m], W)["sens"] for m in steps}
    pts = {m: flag_stats(y, flags[m], w)["sens"][0] for m in steps}
    out["H3"] = [{"from": a, "to": b, "gain": pts[b] - pts[a], "lo": np.percentile(reps[b] - reps[a], 2.5),
                  "hi": np.percentile(reps[b] - reps[a], 97.5)} for a, b in zip(steps[:-1], steps[1:])]
    return out


def subgroups(d, cal, frozen, w, W):
    d = d.assign(age_band=pd.cut(d["age"], [20, 35, 50, 70, 200], right=False, labels=["20-34", "35-49", "50-69", "70+"]),
                 sex=np.where(d["male"] == 1, "Men", "Women"),
                 race=d["race3"].map({1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
                                      4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other or multiracial"}),
                 bmi_band=pd.cut(d["bmi"], [0, 23, 25, 30, 200], right=False, labels=["under 23", "23-25", "25-30", "30+"]))
    methods = {c: d[c].to_numpy() for c in RULES}
    for m in ("lgb", "ebm"):
        methods[m] = (cal[(m, "y57")] >= frozen["calibration"]["thresholds"][f"{m}|y57"]["ada"]).astype(int)
    rows = []
    y = d["y57"].to_numpy()
    for dim in ("age_band", "sex", "race", "bmi_band"):
        for level in d[dim].dropna().unique():
            mask = (d[dim] == level).to_numpy()
            for name, flag in methods.items():
                ym, fm = y[mask], flag[mask]
                point = flag_stats(ym, fm, w[mask])
                reps = flag_stats(ym, fm, W[:, mask])
                rows.append({"dimension": dim, "group": str(level), "method": name, "n": int(mask.sum()),
                             "cases": int(ym.sum()), "flag_rate": point["flag_rate"][0], "sens": point["sens"][0],
                             "sens_lo": np.percentile(reps["sens"], 2.5), "sens_hi": np.percentile(reps["sens"], 97.5)})
    return pd.DataFrame(rows)


def yield_curves(d, cal, w):
    out = {}
    for t in CUTS:
        y = d[t].to_numpy()
        curves = {}
        scores = {name: cal[(name, t)] for name in NAMES}
        scores["ada_points"] = d["ada_points"].to_numpy(float)
        for name, s in scores.items():
            pts = []
            for r in np.round(np.arange(0.01, 0.91, 0.01), 2):
                st = flag_stats(y, (s >= threshold_for_rate(s, w, r)).astype(int), w)
                pts.append([float(r), float(st["flag_rate"][0]), float(st["sens"][0]), float(st["ppv"][0])])
            curves[name] = pts
        rules = {}
        for c in RULES:
            st = flag_stats(y, d[c].to_numpy(), w)
            rules[c] = [float(st["flag_rate"][0]), float(st["sens"][0]), float(st["ppv"][0])]
        out[t] = {"curves": curves, "rules": rules, "prevalence": float(np.average(y, weights=w))}
    return out


def by_cycle(d, raw, cal):
    preds = pd.read_parquet(PROCESSED / "fold_predictions.parquet")
    rows = []
    for (cycle, model, t), g in preds.groupby(["cycle", "model", "target"]):
        y, p, w = g[t].to_numpy(), g["p"].to_numpy(), g["wt_mec"].to_numpy()
        i, s = calibration(p, y, w)
        rows.append({"cycle": cycle, "model": model, "target": t, "kind": "uncalibrated, out of time",
                     "obs_exp": observed_expected(p, y, w)[0], "cal_slope": s})
    w = d["wt_mec"].to_numpy()
    for (model, t), p in raw.items():
        for kind, q in (("uncalibrated, out of time", p), ("calibrated on 2015-2016", cal[(model, t)])):
            y = d[t].to_numpy()
            i, s = calibration(q, y, w)
            rows.append({"cycle": "2017-2020", "model": model, "target": t, "kind": kind,
                         "obs_exp": observed_expected(q, y, w)[0], "cal_slope": s})
    return pd.DataFrame(rows)


def hba1c_by_cycle(d):
    dev = pd.read_parquet(PROCESSED / "cohort_dev.parquet")
    both = pd.concat([dev[["cycle", "hba1c", "wt_mec"]], d[["cycle", "hba1c", "wt_mec"]]])
    rows = []
    for cycle, g in both.groupby("cycle"):
        rows.append({"cycle": cycle, "n": len(g), "mean_hba1c": np.average(g["hba1c"], weights=g["wt_mec"]),
                     "pct_57": 100 * np.average(g["hba1c"] >= 5.7, weights=g["wt_mec"]),
                     "pct_65": 100 * np.average(g["hba1c"] >= 6.5, weights=g["wt_mec"])})
    return pd.DataFrame(rows)


def missed_profile(d, cal, frozen):
    rows = []
    for t in CUTS:
        flag = cal[("lgb", t)] >= frozen["calibration"]["thresholds"][f"lgb|{t}"]["ada"]
        cases = d[d[t] == 1].assign(found=flag[d[t] == 1])
        for found, g in cases.groupby("found"):
            w = g["wt_mec"]
            avg = lambda c: float(np.average(g[c], weights=w)) if g[c].notna().any() else None
            rows.append({"target": t, "group": "found" if found else "missed", "n": len(g),
                         "age": avg("age"), "bmi": float(np.average(g["bmi"].dropna(), weights=w[g["bmi"].notna()])),
                         "men_pct": 100 * avg("male"), "hba1c": avg("hba1c"),
                         "family_history_pct": 100 * float(np.average(g["famhx"].fillna(0), weights=w)),
                         "told_high_bp_pct": 100 * float(np.average(g["htn_told"].fillna(0), weights=w)),
                         "asian_pct": 100 * float(np.average(g["race3"] == 6, weights=w)),
                         "black_pct": 100 * float(np.average(g["race3"] == 4, weights=w))})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    marker = OUT / "UNSEALED.txt"
    if marker.exists() and "--force" not in sys.argv:
        sys.exit("The test cycle was already unsealed; rerun only to fix a documented bug, with --force.")
    frozen = json.loads((MODELS / "frozen.json").read_text())
    models = load_models(frozen)
    labels = unseal()
    marker.write_text(f"Test labels unsealed {time.strftime('%Y-%m-%d %H:%M:%S %Z')}; models frozen {frozen['frozen_at']}\n")

    d = prepare(pd.read_parquet(PROCESSED / "cohort_test_inputs.parquet"))
    d = d.merge(labels[["SEQN", "hba1c"]], on="SEQN", how="left", validate="one_to_one")
    for t, c in CUTS.items():
        d[t] = (d["hba1c"] >= c).astype(int)
    w = d["wt_mec"].to_numpy(float)
    W = replicate_weights(d, 1000)
    raw, cal = predict(models, frozen, d)

    table = main_table(d, cal, frozen, w, W)
    table.round(5).to_csv(OUT / "main_table.csv", index=False)
    disc = discrimination(d, cal, w, W)
    disc.round(5).to_csv(OUT / "discrimination.csv", index=False)
    hyp = hypotheses(d, cal, frozen, w, W, table)
    nb = d[d["borderline"] == 0].reset_index(drop=True)
    nb_w, nb_W = nb["wt_mec"].to_numpy(float), W[:, (d["borderline"] == 0).to_numpy()]
    nb_raw, nb_cal = predict(models, frozen, nb)
    nb_table = main_table(nb, nb_cal, frozen, nb_w, nb_W)
    r = nb_table[(nb_table.target == "y57") & (nb_table.operating_point == "ada") & (nb_table.method == "lgb")].iloc[0]
    hyp["H1_lgb_without_borderline"] = {"diff": r.sens_diff, "lo": r.sens_diff_lo, "hi": r.sens_diff_hi}
    (OUT / "hypotheses.json").write_text(json.dumps(hyp, indent=2, default=float))
    subgroups(d, cal, frozen, w, W).round(5).to_csv(OUT / "subgroups.csv", index=False)
    (OUT / "yield_curves.json").write_text(json.dumps(yield_curves(d, cal, w)))
    by_cycle(d, raw, cal).round(4).to_csv(OUT / "calibration_by_cycle.csv", index=False)
    hba1c_by_cycle(d).round(4).to_csv(OUT / "hba1c_by_cycle.csv", index=False)
    missed_profile(d, cal, frozen).round(3).to_csv(OUT / "missed_profile.csv", index=False)
    print(json.dumps(hyp, indent=1, default=float))


if __name__ == "__main__":
    main()
