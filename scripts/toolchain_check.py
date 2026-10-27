"""Toolchain smoke test for the ES 335 diabetes project (synthetic data only).

Checks, on fake NHANES-shaped data with missing values:
 1. LightGBM with monotone constraints really is monotone.
 2. LightGBM -> JavaScript: m2cgen export and a hand-written evaluator of
    booster.dump_model() both reproduce Python probabilities, NaNs included.
 3. EBM -> JSON -> JavaScript reproduces EBM probabilities and per-term
    contributions exactly (what the website calculator needs).
 4. HbA1c value model (mean + scale + empirical residual CDF) runs and gives
    P(HbA1c >= cut-off); NGBoost timing.
 5. Matched-flag-rate metrics and a PSU-within-strata bootstrap are fast enough.
The numbers printed are NOT findings: the data are simulated.

Run from the project folder:
      pip install -r requirements.txt          (ngboost optional)
      python scripts/toolchain_check.py        # needs Node.js on PATH for the JavaScript checks
Pass: [1] shows 0 violations, [2] and [3] show differences around 1e-16.
Outputs go to .cache/toolchain_check/ (not committed).
"""
import json, subprocess, time, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, lightgbm as lgb, m2cgen as m2c
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "toolchain_check"); os.makedirs(HERE, exist_ok=True)
rng = np.random.default_rng(42)

# ---------------- synthetic cohort ----------------
n = 30000
cycle = rng.integers(0, 6, n)                      # 0-3 train, 4 val, 5 test
age = rng.uniform(20, 80, n)
male = rng.integers(0, 2, n).astype(float)
race = rng.integers(1, 6, n)
bmi = np.exp(rng.normal(np.log(28) + 0.01 * cycle, 0.2, n))
height = np.where(male == 1, rng.normal(175, 7, n), rng.normal(162, 7, n))
waist = 3.0 * bmi + 8 * male + 10 + rng.normal(0, 6, n)
famhx = (rng.random(n) < 0.33).astype(float)
htn = (rng.random(n) < 1 / (1 + np.exp(-(-4 + 0.05 * age + 0.06 * (bmi - 27))))).astype(float)
active = (rng.random(n) < 0.6).astype(float)
inc = np.clip(rng.gamma(2, 1.2, n), 0, 5)
lin = (np.log(5.3) + 0.0025 * (age - 45) + 0.004 * (bmi - 27) + 0.03 * (race == 2)
       + 0.012 * famhx + 0.008 * htn - 0.006 * active)
sd = 0.045 + 0.0008 * (age - 20)
tail = (rng.random(n) < 1 / (1 + np.exp(-(-5.0 + 0.04 * age + 0.08 * (bmi - 27))))) * rng.exponential(0.18, n)
hba1c = np.round(np.exp(lin + rng.normal(0, sd) + tail), 1)
waist[rng.random(n) < 0.05] = np.nan
famhx[rng.random(n) < 0.02] = np.nan
inc[rng.random(n) < 0.08] = np.nan

X = pd.DataFrame(dict(age=age, male=male, bmi=bmi, waist=waist, whtr=waist / height,
                      famhx=famhx, htn=htn, active=active, inc=inc))
for r in range(2, 6):                                  # one-hot race (ref = 1)
    X[f"race{r}"] = (race == r).astype(float)
y57, y65 = (hba1c >= 5.7).astype(int), (hba1c >= 6.5).astype(int)
tr, va, te = cycle <= 3, cycle == 4, cycle == 5
print(f"rows {n}; prevalence >=5.7: {y57.mean():.3f}, >=6.5: {y65.mean():.3f}")

feat = list(X.columns)
mono = [1 if f in ("age", "bmi", "waist", "whtr") else 0 for f in feat]

# ---------------- 1. monotone LightGBM ----------------
# NOTE: use method "intermediate" (or "basic"). In a test with LightGBM 4.7.0 on data
# with missing values, "advanced" let predicted risk fall by up to ~0.01 as age/BMI rose.
params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=100,
              feature_fraction=0.9, lambda_l2=1.0, monotone_constraints=mono,
              monotone_constraints_method="intermediate", verbose=-1, seed=1)
dtr, dva = lgb.Dataset(X[tr], y57[tr]), lgb.Dataset(X[va], y57[va])
t0 = time.time()
bst = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
print(f"[1] LightGBM fit {time.time()-t0:.1f}s, best iter {bst.best_iteration}, "
      f"val AUC {roc_auc_score(y57[va], bst.predict(X[va], num_iteration=bst.best_iteration)):.3f}")
probe = X[te].sample(300, random_state=0).copy()
viol = 0
for f in ("age", "bmi", "waist"):
    grid = np.linspace(np.nanpercentile(X[f], 2), np.nanpercentile(X[f], 98), 25)
    preds = []
    for g in grid:
        p = probe.copy(); p[f] = g
        preds.append(bst.predict(p, num_iteration=bst.best_iteration))
    viol += int((np.diff(np.vstack(preds), axis=0) < -1e-12).sum())
print(f"[1] monotonicity violations over age/bmi/waist grids: {viol}")

# ---------------- 2. LightGBM -> JS ----------------
# keep only best_iteration trees for export
bst_final = lgb.train({**params}, dtr, bst.best_iteration)
test_rows = X[te].head(2000).copy()
test_rows.iloc[:200, test_rows.columns.get_loc("waist")] = np.nan      # force NaNs
test_rows.iloc[100:300, test_rows.columns.get_loc("famhx")] = np.nan
p_py = bst_final.predict(test_rows)
rows_js = [[None if (isinstance(v, float) and np.isnan(v)) else float(v) for v in r] for r in test_rows.to_numpy()]
json.dump(rows_js, open(f"{HERE}/rows.json", "w"))

clf = lgb.LGBMClassifier(**{k: v for k, v in params.items() if k not in ("objective",)},
                         n_estimators=bst.best_iteration)
clf.fit(X[tr], y57[tr])
p_py_clf = clf.predict_proba(test_rows)[:, 1]
open(f"{HERE}/lgb_m2cgen.js", "w").write(m2c.export_to_javascript(clf) +
    "\nmodule.exports = score;\n")
json.dump(bst_final.dump_model(), open(f"{HERE}/lgb_dump.json", "w"))

lgb_js_runner = r"""
const fs = require('fs');
const rows = JSON.parse(fs.readFileSync(process.argv[2] + '/rows.json'));
// (a) m2cgen export: NaN passed as NaN
const score = require(process.argv[2] + '/lgb_m2cgen.js');
const outA = rows.map(r => score(r.map(v => v === null ? NaN : v))[1]);
// (b) hand-written evaluator of booster.dump_model()
const dump = JSON.parse(fs.readFileSync(process.argv[2] + '/lgb_dump.json'));
function decide(node, x) {
  let v = x[node.split_feature];
  if (v === null || Number.isNaN(v)) v = NaN;
  const mt = node.missing_type;
  if (Number.isNaN(v) && mt !== 'NaN') v = 0.0;
  if ((mt === 'Zero' && Math.abs(v) <= 1e-35) || (mt === 'NaN' && Number.isNaN(v)))
    return node.default_left ? node.left_child : node.right_child;
  return v <= node.threshold ? node.left_child : node.right_child;
}
function lgbProb(x) {
  let s = 0;
  for (const t of dump.tree_info) {
    let node = t.tree_structure;
    while (node.leaf_value === undefined) node = decide(node, x);
    s += node.leaf_value;
  }
  return 1 / (1 + Math.exp(-s));
}
const outB = rows.map(lgbProb);
fs.writeFileSync(process.argv[2] + '/lgb_js_out.json', JSON.stringify({m2cgen: outA, dump: outB}));
"""
open(f"{HERE}/run_lgb.js", "w").write(lgb_js_runner)
subprocess.run(["node", f"{HERE}/run_lgb.js", HERE], check=True)
js = json.load(open(f"{HERE}/lgb_js_out.json"))
nan_rows = np.isnan(test_rows["waist"].to_numpy()) | np.isnan(test_rows["famhx"].to_numpy())
for k, ref in (("m2cgen", p_py_clf), ("dump", p_py)):
    d = np.abs(np.array(js[k], dtype=float) - ref)
    print(f"[2] LightGBM->JS via {k:7s}: max |diff| all rows {np.nanmax(d):.2e}; "
          f"rows with NaN {np.nanmax(d[nan_rows]):.2e}; NaN outputs {int(np.isnan(np.array(js[k], dtype=float)).sum())}")

# ---------------- 3. EBM -> JSON -> JS ----------------
t0 = time.time()
ebm = ExplainableBoostingClassifier(interactions=5, outer_bags=8, n_jobs=-1, random_state=1)
ebm.fit(X[tr], y57[tr])
print(f"[3] EBM fit {time.time()-t0:.1f}s; val AUC {roc_auc_score(y57[va], ebm.predict_proba(X[va])[:,1]):.3f}; "
      f"LightGBM val AUC {roc_auc_score(y57[va], bst.predict(X[va], num_iteration=bst.best_iteration)):.3f}")
has_monotonize = hasattr(ebm, "monotonize")
if has_monotonize:
    ebm.monotonize(feat.index("age"), increasing=True)
    ebm.monotonize(feat.index("bmi"), increasing=True)
print(f"[3] EBM.monotonize available: {has_monotonize}")

def ebm_to_json(ebm):
    out = {"intercept": float(np.ravel(ebm.intercept_)[0]), "features": list(ebm.feature_names_in_),
           "types": list(ebm.feature_types_in_), "terms": []}
    for feats, scores in zip(ebm.term_features_, ebm.term_scores_):
        term = {"features": list(map(int, feats)), "scores": np.asarray(scores).tolist(), "cuts": []}
        for f in feats:
            levels = ebm.bins_[f]
            b = levels[min(len(levels), len(feats)) - 1]
            term["cuts"].append(np.asarray(b).tolist())     # continuous: sorted cut points
        out["terms"].append(term)
    return out

model = ebm_to_json(ebm)
json.dump(model, open(f"{HERE}/ebm.json", "w"))
ebm_runner = r"""
const fs = require('fs');
const m = JSON.parse(fs.readFileSync(process.argv[2] + '/ebm.json'));
const rows = JSON.parse(fs.readFileSync(process.argv[2] + '/rows.json'));
function binIndex(cuts, v) {           // 0 = missing, 1..len(cuts)+1 = bins
  if (v === null || Number.isNaN(v)) return 0;
  let lo = 0, hi = cuts.length;        // count cuts <= v  (bisect_right)
  while (lo < hi) { const mid = (lo + hi) >> 1; if (cuts[mid] <= v) lo = mid + 1; else hi = mid; }
  return lo + 1;
}
function ebmScore(x) {
  let s = m.intercept; const contrib = [];
  for (const t of m.terms) {
    let sc = t.scores;
    t.features.forEach((f, j) => { sc = sc[binIndex(t.cuts[j], x[f])]; });
    contrib.push(sc); s += sc;
  }
  return {p: 1 / (1 + Math.exp(-s)), contrib};
}
const out = rows.map(ebmScore);
fs.writeFileSync(process.argv[2] + '/ebm_js_out.json', JSON.stringify(out));
"""
open(f"{HERE}/run_ebm.js", "w").write(ebm_runner)
subprocess.run(["node", f"{HERE}/run_ebm.js", HERE], check=True)
ejs = json.load(open(f"{HERE}/ebm_js_out.json"))
p_ebm_py = ebm.predict_proba(test_rows)[:, 1]
contrib_py = ebm.eval_terms(test_rows)
p_ebm_js = np.array([o["p"] for o in ejs])
c_js = np.array([o["contrib"] for o in ejs])
print(f"[3] EBM->JS: max |diff| prob {np.abs(p_ebm_js-p_ebm_py).max():.2e}; "
      f"per-term contributions {np.abs(c_js-contrib_py).max():.2e}; NaN rows prob diff "
      f"{np.abs(p_ebm_js-p_ebm_py)[nan_rows].max():.2e}; JSON size {os.path.getsize(f'{HERE}/ebm.json')/1024:.0f} KB")

# ---------------- 4. HbA1c value model ----------------
logy = np.log(hba1c)
Xtr, ytr = X[tr].reset_index(drop=True), logy[tr]
rparams = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_child_samples=100,
               monotone_constraints=mono, monotone_constraints_method="intermediate", verbose=-1, seed=1)
t0 = time.time()
oof = np.zeros(len(ytr))
for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(Xtr)):
    mdl = lgb.train(rparams, lgb.Dataset(Xtr.iloc[a], ytr[a]), 600)
    oof[b] = mdl.predict(Xtr.iloc[b])
mu_model = lgb.train(rparams, lgb.Dataset(Xtr, ytr), 600)
res = ytr - oof
sparams = dict(objective="regression", learning_rate=0.03, num_leaves=15, min_child_samples=200,
               verbose=-1, seed=1)
sc_model = lgb.train(sparams, lgb.Dataset(Xtr, np.abs(res)), 400)
sig_tr = np.clip(sc_model.predict(Xtr), 1e-3, None)
z_sorted = np.sort(res / sig_tr)                      # empirical standardized residuals
def p_exceed(Xn, cut):
    mu, sig = mu_model.predict(Xn), np.clip(sc_model.predict(Xn), 1e-3, None)
    z = (np.log(cut - 0.05) - mu) / sig               # continuity: values reported to 0.1
    return 1 - np.searchsorted(z_sorted, z, side="left") / len(z_sorted)
print(f"[4] value model (5-fold OOF mean + scale) fit {time.time()-t0:.1f}s")
clf65 = lgb.train({**params, "scale_pos_weight": 1.0}, lgb.Dataset(X[tr], y65[tr]), 2000,
                  valid_sets=[lgb.Dataset(X[va], y65[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
for name, s in (("classifier A", clf65.predict(X[te], num_iteration=clf65.best_iteration)),
                ("value model B", p_exceed(X[te], 6.5))):
    print(f"[4] >=6.5 on synthetic test: {name:13s} AUC {roc_auc_score(y65[te], s):.3f}  "
          f"PR-AUC {average_precision_score(y65[te], s):.3f}")
try:
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    t0 = time.time()
    Xi = Xtr.fillna(Xtr.median())
    ng = NGBRegressor(Dist=Normal, n_estimators=300, learning_rate=0.05, verbose=False).fit(Xi, ytr)
    print(f"[4] NGBoost (Normal, 300 iters) fit {time.time()-t0:.1f}s on {len(ytr)} rows")
except Exception as e:
    print("[4] NGBoost failed:", repr(e)[:120])

# ---------------- 5. matched flag rate + PSU bootstrap ----------------
def flag_threshold(score, rate):
    """Threshold that flags `rate` of rows (top scores)."""
    return np.quantile(score, 1 - rate)
def sens_ppv(y, flagged, w=None):
    w = np.ones_like(y, dtype=float) if w is None else w
    tp = (w * (flagged & (y == 1))).sum()
    return tp / (w * (y == 1)).sum(), tp / max((w * flagged).sum(), 1e-12)
ada_like = (age >= 40).astype(int) + (age >= 50) + (age >= 60) + male + np.nan_to_num(famhx) + htn + (1 - active) \
           + (bmi >= 25) + (bmi >= 30) + (bmi >= 40)
flag_rate_val = (ada_like[va] >= 5).mean()
model_score = bst.predict(X, num_iteration=bst.best_iteration)
thr = flag_threshold(model_score[va], flag_rate_val)          # set on validation, used on test
strata = rng.integers(0, 15, n); psu = rng.integers(0, 2, n); wt = rng.uniform(0.5, 3, n)
yt, wt_t, st_t, ps_t = y57[te], wt[te], strata[te], psu[te]
fa, fm = ada_like[te] >= 5, model_score[te] >= thr
print(f"[5] flag rate val {flag_rate_val:.3f}; on test ADA {fa.mean():.3f} vs model {fm.mean():.3f}")
def psu_bootstrap(B=1000, seed=0):
    g = np.random.default_rng(seed)
    clusters = {}
    for i, (s, p) in enumerate(zip(st_t, ps_t)):
        clusters.setdefault(s, {}).setdefault(p, []).append(i)
    out = []
    for _ in range(B):
        idx = []
        for s, ps in clusters.items():                       # resample PSUs within each stratum
            keys = list(ps)
            for k in g.choice(keys, size=len(keys), replace=True):
                idx.extend(ps[k])
        idx = np.array(idx)
        sa, _ = sens_ppv(yt[idx], fa[idx], wt_t[idx]); sm, _ = sens_ppv(yt[idx], fm[idx], wt_t[idx])
        out.append(sm - sa)
    return np.percentile(out, [2.5, 50, 97.5])
t0 = time.time()
ci = psu_bootstrap()
print(f"[5] paired PSU bootstrap of sensitivity difference (1000 reps) {time.time()-t0:.1f}s: "
      f"median {ci[1]:+.3f}, 95% CI [{ci[0]:+.3f}, {ci[2]:+.3f}]")
