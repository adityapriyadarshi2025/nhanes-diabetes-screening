import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import PROCESSED, ROOT
from .scores import add_scores

SEED = 2026
FEATURES = ["age", "male", "bmi", "waist", "whtr", "height", "weight", "arm", "leg", "htn_told", "bp_med",
            "famhx", "cvd", "gdm", "smoke_former", "smoke_current", "active_rec", "mvpa_min_wk",
            "sedentary_min", "educ", "pir"]
MONOTONE = {"age", "bmi", "waist", "whtr"}
CUTS = {"y57": 5.7, "y65": 6.5}
SPLITS = json.loads((ROOT / "splits.json").read_text())
FOLDS = [(f["train"], f["check"]) for f in SPLITS["tuning_folds"]]
C_GRID = list(np.logspace(-3, 1, 10))
LGB_GRID = [dict(num_leaves=a, min_data_in_leaf=b, lambda_l2=c)
            for a in (15, 31, 63) for b in (50, 100, 200) for c in (0.0, 1.0, 5.0)]
LGB_BASE = dict(learning_rate=0.03, feature_fraction=0.8, seed=SEED, deterministic=True, force_col_wise=True,
                num_threads=2, verbose=-1, monotone_constraints_method="intermediate",
                monotone_constraints=[1 if f in MONOTONE else 0 for f in FEATURES])


def load_dev():
    return prepare(pd.read_parquet(PROCESSED / "cohort_dev.parquet"))


def prepare(d):
    d = add_scores(d)
    known = d["smoking"].notna()
    d["smoke_former"] = np.where(known, (d["smoking"] == 1).astype(float), np.nan)
    d["smoke_current"] = np.where(known, (d["smoking"] == 2).astype(float), np.nan)
    return d.reset_index(drop=True)


def ada_items(d):
    a, b = d["age"], d["bmi"]
    return pd.DataFrame({
        "age_40_49": a.between(40, 50, inclusive="left"), "age_50_59": a.between(50, 60, inclusive="left"),
        "age_60_plus": a >= 60, "man": d["male"] == 1, "gestational": d["gdm"] == 1, "family": d["famhx"] == 1,
        "high_bp": d["htn_told"] == 1, "inactive": d["active_rec"] == 0,
        "bmi_25_29": b.between(25, 30, inclusive="left"), "bmi_30_39": b.between(30, 40, inclusive="left"),
        "bmi_40_plus": b >= 40}).astype(float)


class AdaRefit:
    def fit(self, d, y, valid=None):
        self.m = LogisticRegression(penalty=None, max_iter=5000).fit(ada_items(d), y)
        return self

    def prob(self, d, target=None):
        return self.m.predict_proba(ada_items(d))[:, 1]


class L1Logistic:
    def __init__(self, C):
        self.C = C

    def fit(self, d, y, valid=None):
        self.m = make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                               LogisticRegression(penalty="l1", C=self.C, solver="liblinear", max_iter=5000))
        self.m.fit(d[FEATURES], y)
        return self

    def prob(self, d, target=None):
        return self.m.predict_proba(d[FEATURES])[:, 1]


class EBM:
    def __init__(self, interactions):
        self.interactions = interactions

    def fit(self, d, y, valid=None):
        self.m = ExplainableBoostingClassifier(
            feature_names=FEATURES, feature_types=["continuous"] * len(FEATURES), interactions=self.interactions,
            max_bins=256, outer_bags=8, random_state=SEED, n_jobs=-1)
        self.m.fit(d[FEATURES], y)
        return self

    def prob(self, d, target=None):
        return self.m.predict_proba(d[FEATURES])[:, 1]


class LGBClassifier:
    def __init__(self, params, rounds=None):
        self.params, self.rounds = params, rounds

    def fit(self, d, y, valid=None):
        p = {**LGB_BASE, **self.params, "objective": "binary"}
        train = lgb.Dataset(d[FEATURES], y)
        if valid is not None:
            dv, yv = valid
            self.b = lgb.train(p, train, 5000, valid_sets=[lgb.Dataset(dv[FEATURES], yv)],
                               callbacks=[lgb.early_stopping(100, verbose=False)])
            self.rounds = self.b.best_iteration
        else:
            self.b = lgb.train(p, train, self.rounds)
        return self

    def prob(self, d, target=None):
        return self.b.predict(d[FEATURES], num_iteration=self.rounds)


class ValueModel:
    """Model B: log HbA1c = mu(x) + sigma(x) * z, z from the training residuals."""

    def __init__(self, params, rounds=None):
        self.params, self.rounds = params, rounds

    def fit(self, d, y=None, valid=None):
        p = {**LGB_BASE, **self.params, "objective": "regression"}
        target = np.log(d["hba1c"].to_numpy())
        if valid is not None:
            dv = valid[0]
            b = lgb.train(p, lgb.Dataset(d[FEATURES], target), 5000,
                          valid_sets=[lgb.Dataset(dv[FEATURES], np.log(dv["hba1c"].to_numpy()))],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            self.rounds = b.best_iteration
        self.mu = lgb.train(p, lgb.Dataset(d[FEATURES], target), self.rounds)
        oof = np.zeros(len(d))
        for a, b in KFold(5, shuffle=True, random_state=SEED).split(d):
            m = lgb.train(p, lgb.Dataset(d[FEATURES].iloc[a], target[a]), self.rounds)
            oof[b] = m.predict(d[FEATURES].iloc[b])
        resid = target - oof
        sp = {**LGB_BASE, "objective": "regression", "num_leaves": 15, "min_data_in_leaf": 200,
              "monotone_constraints": [0] * len(FEATURES)}
        self.sigma = lgb.train(sp, lgb.Dataset(d[FEATURES], np.abs(resid)), 300)
        self.z = np.sort(resid / self.scale(d))
        return self

    def scale(self, d):
        return np.clip(self.sigma.predict(d[FEATURES]), 1e-3, None)

    def median_and_range(self, d, q=(0.1, 0.5, 0.9)):
        mu, s = self.mu.predict(d[FEATURES]), self.scale(d)
        return np.exp(mu[:, None] + s[:, None] * np.quantile(self.z, q)[None, :])

    def prob(self, d, target):
        z = (np.log(CUTS[target] - 0.05) - self.mu.predict(d[FEATURES])) / self.scale(d)
        return 1 - np.searchsorted(self.z, z, side="left") / len(self.z)


def split(d, cycles):
    return d[d["cycle"].isin(cycles if isinstance(cycles, list) else [cycles])].reset_index(drop=True)


def log_loss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
