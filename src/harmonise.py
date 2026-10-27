"""Read the raw NHANES files and build one harmonised row per participant.

Each cycle's files are merged on SEQN (the participant id) and recoded into the
same variables with the same meaning in every cycle. Codebook answers such as
7/9, 77/99 or 7777/9999 (refused, don't know) become missing.

The test cycle's lab outcomes are written to a separate "sealed" file, so no
working file contains them before the models are frozen:

  data/processed/persons_dev.parquet          everyone in 2007-2016, all columns
  data/processed/persons_test_inputs.parquet  everyone in 2017-March 2020, no outcome values
  data/processed/test_labels.sealed.parquet   2017-March 2020 HbA1c and fasting glucose

Run from the project folder:  python -m src.harmonise
"""
import hashlib

import numpy as np
import pandas as pd

from .config import COMPONENTS, CYCLES, PROCESSED, RAW, RESULTS, SEALED_COLS, TEST_CYCLE


# ---------------------------------------------------------------- reading
def read_xpt(path):
    df = pd.read_sas(path, format="xport")
    df.columns = [c.upper() for c in df.columns]      # 2017-2020 uses e.g. MCQ300c
    num = df.select_dtypes("number").columns
    # pandas reads a stored zero as about 5.4e-79 in some NHANES files
    df[num] = df[num].mask(df[num].abs() < 1e-30, 0.0)
    df["SEQN"] = df["SEQN"].astype("int64")
    if df["SEQN"].duplicated().any():
        raise ValueError(f"duplicate SEQN in {path.name}")
    return df


def load_cycle(label):
    pattern = CYCLES[label]
    merged = None
    for comp in COMPONENTS:
        df = read_xpt(RAW / label / f"{pattern.format(comp)}.xpt")
        if merged is None:                      # DEMO first: one row per participant
            merged = df
        else:
            clash = [c for c in df.columns if c in merged.columns and c != "SEQN"]
            merged = merged.merge(df.drop(columns=clash), on="SEQN", how="left",
                                  validate="one_to_one")
    return merged


# ---------------------------------------------------------------- recoding helpers
def col(raw, name):
    """The column if this cycle has it, else all missing."""
    if name in raw.columns:
        return raw[name].astype(float)
    return pd.Series(np.nan, index=raw.index)


def yes_no(s):
    """1 = yes -> 1, 2 = no -> 0; refused, don't know, not asked -> missing."""
    return s.map({1: 1.0, 2: 0.0})


def in_range(s, lo, hi):
    """Keep values in [lo, hi]; special codes (7777, 9999 ...) -> missing."""
    return s.where((s >= lo) & (s <= hi))


def any_all(frame):
    """1 if any column is 1; 0 if every column is 0; else missing."""
    return pd.Series(np.where(frame.eq(1).any(axis=1), 1.0,
                              np.where(frame.eq(0).all(axis=1), 0.0, np.nan)),
                     index=frame.index)


# ---------------------------------------------------------------- recode one cycle
def recode(raw, label):
    test = label == TEST_CYCLE
    d = pd.DataFrame({"SEQN": raw["SEQN"].values}, index=raw.index)
    d["cycle"] = label

    # survey design
    d["wt_mec"] = col(raw, "WTMECPRP" if test else "WTMEC2YR")
    d["wt_fast"] = col(raw, "WTSAFPRP" if test else "WTSAF2YR")
    d["strata"] = label + "_" + raw["SDMVSTRA"].astype(int).astype(str)
    d["psu"] = d["strata"] + "_" + raw["SDMVPSU"].astype(int).astype(str)

    # who is in the sample
    d["examined"] = (col(raw, "RIDSTATR") == 2).astype(int)
    d["age"] = col(raw, "RIDAGEYR")                    # 80 means 80 or older
    d["male"] = (col(raw, "RIAGENDR") == 1).astype(int)
    d["race1"] = col(raw, "RIDRETH1")   # 1 Mexican Am., 2 other Hispanic, 3 NH White, 4 NH Black, 5 other
    d["race3"] = col(raw, "RIDRETH3")   # as race1 but 6 = NH Asian, 7 = other; 2011 onward only
    d["pregnant"] = ((col(raw, "RIDEXPRG") == 1) | (col(raw, "RHD143") == 1)).astype(int)
    d["educ"] = in_range(col(raw, "DMDEDUC2"), 1, 5)   # 1 < 9th grade ... 5 college graduate
    d["pir"] = col(raw, "INDFMPIR")                     # income / poverty line, capped at 5

    # body measurements (measured at the exam)
    d["weight"] = col(raw, "BMXWT")
    d["height"] = col(raw, "BMXHT")
    d["bmi"] = col(raw, "BMXBMI")
    d["waist"] = col(raw, "BMXWAIST")
    d["whtr"] = d["waist"] / d["height"]
    d["arm"] = col(raw, "BMXARMC")
    d["leg"] = col(raw, "BMXLEG")
    sr_height = in_range(col(raw, "WHD010"), 48, 90)     # self-reported, inches
    sr_weight = in_range(col(raw, "WHD020"), 50, 1000)   # self-reported, pounds
    d["sr_bmi"] = 703.0 * sr_weight / sr_height ** 2

    # diabetes questions: cohort rules and the leakage ladder only, never model inputs
    d["dx_answer"] = col(raw, "DIQ010")    # 1 yes, 2 no, 3 borderline, 7/9 unknown
    d["insulin_now"] = yes_no(col(raw, "DIQ050"))
    d["dm_pills_now"] = yes_no(col(raw, "DIQ070" if "DIQ070" in raw.columns else "DID070"))  # DID070 in 2007-08
    d["told_prediabetes"] = yes_no(col(raw, "DIQ160"))
    d["blood_test_3y"] = yes_no(col(raw, "DIQ180"))

    # risk-factor answers
    d["htn_told"] = yes_no(col(raw, "BPQ020"))
    bp_now = yes_no(col(raw, "BPQ050A"))
    never_told_meds = (col(raw, "BPQ040A") == 2) | (d["htn_told"] == 0)
    d["bp_med"] = bp_now.where(bp_now.notna(), np.where(never_told_meds, 0.0, np.nan))
    d["famhx"] = yes_no(col(raw, "MCQ300C"))
    d["cvd"] = any_all(pd.concat([yes_no(col(raw, f"MCQ160{x}")) for x in "BCDEF"], axis=1))

    # gestational diabetes: women ever told during a pregnancy; never pregnant -> no; men -> no
    gdm = col(raw, "RHQ162").map({1: 1.0, 2: 0.0, 3: 0.0})   # 3 = borderline, not a diagnosis
    never_pregnant = yes_no(col(raw, "RHQ131")) == 0
    gdm = gdm.where(gdm.notna(), np.where(never_pregnant, 0.0, np.nan))
    d["gdm"] = np.where(d["male"] == 1, 0.0, gdm)

    # smoking: 0 never, 1 former, 2 current
    ever = yes_no(col(raw, "SMQ020"))
    now = col(raw, "SMQ040")                 # 1 every day, 2 some days, 3 not at all
    d["smoking"] = np.select([ever == 0, (ever == 1) & (now == 3), (ever == 1) & now.isin([1, 2])],
                             [0.0, 1.0, 2.0], default=np.nan)

    # physical activity (same questionnaire 2007 to March 2020)
    def weekly(flag, days, minutes):
        f = yes_no(col(raw, flag))
        total = in_range(col(raw, days), 1, 7) * in_range(col(raw, minutes), 1, 1440)
        return np.where(f == 0, 0.0, np.where(f == 1, total, np.nan))
    vigorous = weekly("PAQ605", "PAQ610", "PAD615") + weekly("PAQ650", "PAQ655", "PAD660")
    moderate = (weekly("PAQ620", "PAQ625", "PAD630") + weekly("PAQ635", "PAQ640", "PAD645")
                + weekly("PAQ665", "PAQ670", "PAD675"))
    d["mvpa_min_wk"] = moderate + 2 * vigorous          # moderate-equivalent minutes a week
    flags = pd.concat({v: yes_no(col(raw, v)) for v in
                       ["PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665"]}, axis=1)
    d["active_any"] = any_all(flags)                    # any of the five activity types
    d["active_rec"] = any_all(flags[["PAQ650", "PAQ665"]])   # leisure-time activity
    d["sedentary_min"] = in_range(col(raw, "PAD680"), 0, 1440)

    # labs: outcome (HbA1c), fasting glucose (check only), HDL (leakage ladder only)
    d["hba1c"] = col(raw, "LBXGH").round(1)
    d["has_hba1c"] = d["hba1c"].notna().astype(int)
    d["fpg"] = col(raw, "LBXGLU")
    d["hdl"] = col(raw, "LBDHDD")
    return d


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    frames = []
    for label in CYCLES:
        raw = load_cycle(label)
        d = recode(raw, label)
        print(f"{label}: {len(d):>6} participants, {raw.shape[1]:>4} raw columns")
        frames.append(d)
    persons = pd.concat(frames, ignore_index=True)

    dev = persons[persons["cycle"] != TEST_CYCLE]
    test = persons[persons["cycle"] == TEST_CYCLE]
    dev.to_parquet(PROCESSED / "persons_dev.parquet", index=False)
    test.drop(columns=SEALED_COLS).to_parquet(PROCESSED / "persons_test_inputs.parquet", index=False)
    sealed = PROCESSED / "test_labels.sealed.parquet"
    labels = test[["SEQN"] + SEALED_COLS].sort_values("SEQN")
    labels.to_parquet(sealed, index=False)
    # The file hash depends on the pyarrow version; the content hash does not.
    content = hashlib.sha256(labels.to_csv(index=False, float_format="%.6g").encode()).hexdigest()
    note = RESULTS / "week1" / "sealed_labels_hash.txt"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(f"file sha256    {sha256(sealed)}\ncontent sha256 {content}\nrows           {len(labels)}\n")
    print(f"dev rows {len(dev)}, test rows {len(test)}; sealed labels content sha256 {content[:16]}...")


if __name__ == "__main__":
    main()
