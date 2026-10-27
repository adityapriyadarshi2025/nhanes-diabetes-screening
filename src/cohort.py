"""Apply the cohort rules and count who is left after each step, per cycle.

Cohort: adults aged 20 or over, examined, not pregnant, with an HbA1c result,
who answered "no" or "borderline" to "ever told you have diabetes" and take
neither insulin nor diabetes pills.

Writes
  data/processed/cohort_dev.parquet          2007-2016 cohort with targets
  data/processed/cohort_test_inputs.parquet  2017-March 2020 cohort, inputs only
  data/processed/examined_adults_dev.parquet 2007-2016 adults incl. diagnosed (leakage ladder)
  results/week1/cohort_flow.csv              counts after each step, per cycle

Run from the project folder:  python -m src.cohort
"""
import pandas as pd

from .config import PROCESSED, RESULTS, SEALED_COLS, TEST_CYCLE

STEPS = [
    ("All participants", lambda d: d["age"].notna()),
    ("Aged 20 or over", lambda d: d["age"] >= 20),
    ("Examined", lambda d: d["examined"] == 1),
    ("Not pregnant", lambda d: d["pregnant"] == 0),
    ("HbA1c measured", lambda d: d["has_hba1c"] == 1),
    ("Never told they have diabetes (answered no or borderline)", lambda d: d["dx_answer"].isin([2, 3])),
    ("Not taking insulin or diabetes pills", lambda d: (d["insulin_now"] != 1) & (d["dm_pills_now"] != 1)),
]


def apply_steps(d):
    keep = pd.Series(True, index=d.index)
    counts = []
    for name, rule in STEPS:
        keep &= rule(d)
        counts.append((name, d.loc[keep].groupby("cycle").size()))
    return keep, counts


def main():
    dev = pd.read_parquet(PROCESSED / "persons_dev.parquet")
    test = pd.read_parquet(PROCESSED / "persons_test_inputs.parquet")
    both = pd.concat([dev, test], ignore_index=True)

    keep, counts = apply_steps(both)
    flow = pd.DataFrame({name: c for name, c in counts}).T.fillna(0).astype(int)
    flow["Total"] = flow.sum(axis=1)
    # why people left at the diagnosis step
    reached = both[apply_steps_upto(both, 5)]
    flow.loc["  of which: told they have diabetes"] = _by_cycle(reached["dx_answer"] == 1, reached)
    flow.loc["  of which: answer refused, unknown or missing"] = _by_cycle(~reached["dx_answer"].isin([1, 2, 3]), reached)
    (RESULTS / "week1").mkdir(parents=True, exist_ok=True)
    flow.to_csv(RESULTS / "week1" / "cohort_flow.csv")
    print(flow.to_string())

    cohort = both[keep].copy()
    cohort["borderline"] = (cohort["dx_answer"] == 3).astype(int)
    c_dev = cohort[cohort["cycle"] != TEST_CYCLE].copy()
    c_dev["y57"] = (c_dev["hba1c"] >= 5.7).astype(int)
    c_dev["y65"] = (c_dev["hba1c"] >= 6.5).astype(int)
    c_dev.to_parquet(PROCESSED / "cohort_dev.parquet", index=False)
    c_test = cohort[cohort["cycle"] == TEST_CYCLE].drop(columns=SEALED_COLS)   # all empty here
    c_test.to_parquet(PROCESSED / "cohort_test_inputs.parquet", index=False)

    adults = dev[apply_steps_upto(dev, 5)].copy()      # before the diagnosis step
    adults.to_parquet(PROCESSED / "examined_adults_dev.parquet", index=False)
    print(f"\ncohort: dev {len(c_dev)}, test {len(c_test)}")


def apply_steps_upto(d, n):
    keep = pd.Series(True, index=d.index)
    for _, rule in STEPS[:n]:
        keep &= rule(d)
    return keep


def _by_cycle(mask, frame):
    s = mask.groupby(frame["cycle"]).sum().astype(int)
    s["Total"] = int(mask.sum())
    return s


if __name__ == "__main__":
    main()
