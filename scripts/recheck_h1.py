"""Independent recheck of H1 that shares no evaluation code with src/evaluate.py.

Rescores the ADA test from the raw inputs, runs the frozen LightGBM directly and
draws its own Rao-Wu bootstrap (seed 7). Run after src.test_once:
    python scripts/recheck_h1.py
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import FEATURES, prepare  # noqa: E402

if not (ROOT / "results" / "test" / "UNSEALED.txt").exists():
    sys.exit("Run src.test_once first.")

d = prepare(pd.read_parquet(ROOT / "data" / "processed" / "cohort_test_inputs.parquet"))
d = d.merge(pd.read_parquet(ROOT / "data" / "processed" / "test_labels.sealed.parquet")[["SEQN", "hba1c"]], on="SEQN")
y, w = (d["hba1c"] >= 5.7).to_numpy(int), d["wt_mec"].to_numpy(float)
age, bmi = d["age"].to_numpy(), d["bmi"].to_numpy()
points = (np.select([age >= 60, age >= 50, age >= 40], [3, 2, 1], 0) + (d["male"] == 1) + (d["gdm"] == 1)
          + (d["famhx"] == 1) + (d["htn_told"] == 1) + (d["active_rec"] == 0)
          + np.select([bmi >= 40, bmi >= 30, bmi >= 25], [3, 2, 1], 0))
ada = (points >= 5).to_numpy(int)
print("ADA flags identical to the pipeline:", bool((ada == d["ada_flag"].to_numpy()).all()))

frozen = json.loads((ROOT / "models" / "frozen.json").read_text())
p = lgb.Booster(model_file=str(ROOT / "models" / "lgb_y57.txt")).predict(d[FEATURES])
a, b = frozen["calibration"]["platt"]["lgb|y57"]
q = 1 / (1 + np.exp(-(a + b * np.log(p / (1 - p)))))
model = (q >= frozen["calibration"]["thresholds"]["lgb|y57"]["ada"]).astype(int)


def sens(flag, ww):
    return (ww * y * flag).sum() / (ww * y).sum()


def ppv(flag, ww):
    return (ww * y * flag).sum() / (ww * flag).sum()


print(f"ADA test: tests {np.average(ada, weights=w):.4f}, sensitivity {sens(ada, w):.4f}, PPV {ppv(ada, w):.4f}")
print(f"LightGBM: tests {np.average(model, weights=w):.4f}, sensitivity {sens(model, w):.4f}, PPV {ppv(model, w):.4f}")

units = {}
for i, (s, u) in enumerate(zip(d["strata"], d["psu"])):
    units.setdefault(s, {}).setdefault(u, []).append(i)
rng = np.random.default_rng(7)
diffs = []
for _ in range(1000):
    ww = np.zeros_like(w)
    for psus in units.values():
        keys = list(psus)
        n = len(keys)
        for k in rng.integers(0, n, n - 1):
            idx = psus[keys[k]]
            ww[idx] += w[idx] * n / (n - 1)
    diffs.append(sens(model, ww) - sens(ada, ww))
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"H1 sensitivity difference {100 * (sens(model, w) - sens(ada, w)):+.2f} points, 95% interval {100 * lo:+.2f} to {100 * hi:+.2f}")
