import json
import sys

import numpy as np
import pandas as pd

from .config import PROCESSED, RESULTS, ROOT
from .evaluate import flag_stats, replicate_weights, threshold_for_rate
from .models import CUTS, prepare
from .test_once import NAMES, OPS, load_models, predict, unseal

OUT = RESULTS / "review"


def add(row, key, point, reps):
    row.update({key: point, f"{key}_lo": np.percentile(reps, 2.5), f"{key}_hi": np.percentile(reps, 97.5)})


def matched(p, y, rule_rate, rule_rate_reps, w, W):
    """Threshold re-set on 2017-2020 so the model tests the rule's share; re-set again in every replicate."""
    point = flag_stats(y, (p >= threshold_for_rate(p, w, rule_rate)).astype(int), w)
    sens, ppv = np.empty(len(W)), np.empty(len(W))
    for r in range(len(W)):
        s = flag_stats(y, (p >= threshold_for_rate(p, W[r], rule_rate_reps[r])).astype(int), W[r])
        sens[r], ppv[r] = s["sens"][0], s["ppv"][0]
    return point, {"sens": sens, "ppv": ppv}


def matched_volume(d, cal, frozen, w, W):
    rows = []
    for t in CUTS:
        y = d[t].to_numpy()
        for op, col in OPS.items():
            rule = d[col].to_numpy()
            rp, rr = flag_stats(y, rule, w), flag_stats(y, rule, W)
            for name in NAMES:
                p = cal[(name, t)]
                flag = (p >= frozen["calibration"]["thresholds"][f"{name}|{t}"][op]).astype(int)
                fp, fr = flag_stats(y, flag, w), flag_stats(y, flag, W)
                mp, mr = matched(p, y, rp["flag_rate"][0], rr["flag_rate"], w, W)
                row = {"target": t, "operating_point": op, "method": name, "rule_tested": rp["flag_rate"][0],
                       "model_tested_frozen": fp["flag_rate"][0]}
                add(row, "tested_diff_frozen", fp["flag_rate"][0] - rp["flag_rate"][0], fr["flag_rate"] - rr["flag_rate"])
                add(row, "sens_diff_frozen", fp["sens"][0] - rp["sens"][0], fr["sens"] - rr["sens"])
                add(row, "sens_matched", mp["sens"][0], mr["sens"])
                add(row, "sens_diff_matched", mp["sens"][0] - rp["sens"][0], mr["sens"] - rr["sens"])
                add(row, "ppv_matched", mp["ppv"][0], mr["ppv"])
                add(row, "ppv_diff_matched", mp["ppv"][0] - rp["ppv"][0], mr["ppv"] - rr["ppv"])
                rows.append(row)
    return pd.DataFrame(rows)


def subgroup_differences(d, cal, frozen, w, W):
    d = d.assign(age_band=pd.cut(d["age"], [20, 35, 50, 70, 200], right=False, labels=["20-34", "35-49", "50-69", "70+"]),
                 sex=np.where(d["male"] == 1, "Men", "Women"),
                 race=d["race3"].map({1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
                                      4: "Non-Hispanic Black", 6: "Non-Hispanic Asian", 7: "Other or multiracial"}),
                 bmi_band=pd.cut(d["bmi"], [0, 23, 25, 30, 200], right=False, labels=["under 23", "23-25", "25-30", "30+"]))
    y, ada = d["y57"].to_numpy(), d["ada_flag"].to_numpy()
    rows = []
    for dim in ("age_band", "sex", "race", "bmi_band"):
        for level in d[dim].dropna().unique():
            mask = (d[dim] == level).to_numpy()
            ap, ar = flag_stats(y[mask], ada[mask], w[mask]), flag_stats(y[mask], ada[mask], W[:, mask])
            for m in ("lgb", "ebm"):
                flag = (cal[(m, "y57")] >= frozen["calibration"]["thresholds"][f"{m}|y57"]["ada"]).astype(int)[mask]
                mp, mr = flag_stats(y[mask], flag, w[mask]), flag_stats(y[mask], flag, W[:, mask])
                row = {"dimension": dim, "group": str(level), "method": m, "cases": int(y[mask].sum()),
                       "ada_sens": ap["sens"][0], "model_sens": mp["sens"][0]}
                add(row, "sens_diff", mp["sens"][0] - ap["sens"][0], mr["sens"] - ar["sens"])
                add(row, "tested_diff", mp["flag_rate"][0] - ap["flag_rate"][0], mr["flag_rate"] - ar["flag_rate"])
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    if not (RESULTS / "test" / "UNSEALED.txt").exists():
        sys.exit("Run src.test_once first: this script only adds analyses after the one-time test.")
    frozen = json.loads((ROOT / "models" / "frozen.json").read_text())
    models = load_models(frozen)
    labels = unseal()
    d = prepare(pd.read_parquet(PROCESSED / "cohort_test_inputs.parquet"))
    d = d.merge(labels[["SEQN", "hba1c"]], on="SEQN", how="left", validate="one_to_one")
    for t, c in CUTS.items():
        d[t] = (d["hba1c"] >= c).astype(int)
    w = d["wt_mec"].to_numpy(float)
    W = replicate_weights(d, 1000)
    _, cal = predict(models, frozen, d)
    OUT.mkdir(parents=True, exist_ok=True)
    mv = matched_volume(d, cal, frozen, w, W)
    mv.round(5).to_csv(OUT / "matched_volume.csv", index=False)
    subgroup_differences(d, cal, frozen, w, W).round(5).to_csv(OUT / "subgroup_differences.csv", index=False)
    r = mv[(mv.target == "y57") & (mv.operating_point == "ada") & (mv.method == "lgb")].iloc[0]
    print(r.drop(["target", "operating_point", "method"]).astype(float).round(4).to_string())


if __name__ == "__main__":
    main()
