"""Hand-checked rows for the three baselines.  Run: python -m tests.test_scores"""
import numpy as np
import pandas as pd

from src.scores import ada2026_flag, ada_points, uspstf_flag

BASE = dict(age=45, male=0, gdm=0, famhx=0, htn_told=0, bp_med=0, cvd=0,
            active_rec=1, active_any=1, bmi=22.0, sr_bmi=22.0, race1=3, race3=3)


def rows(*changes):
    return pd.DataFrame([{**BASE, **c} for c in changes])


def test_ada_points():
    d = rows(
        dict(age=45, male=1, bmi=27.0),                                    # 1 + 1 + 1 = 3
        dict(age=62, bmi=31.0, gdm=1, famhx=1, active_rec=0),             # 3 + 2 + 1 + 1 + 1 = 8
        dict(age=39, male=1, bmi=41.0, htn_told=1, active_rec=0),         # 0 + 1 + 3 + 1 + 1 = 6
        dict(age=50, famhx=np.nan, active_rec=np.nan, bmi=np.nan),         # unknowns add nothing: 2
        dict(age=40, bmi=29.99), dict(age=40, bmi=30.0), dict(age=60, bmi=39.9),
    )
    assert ada_points(d).tolist() == [3, 8, 6, 2, 2, 3, 5]


def test_ada_asian_bmi():
    d = rows(dict(age=50, bmi=24.0, race1=5, race3=6), dict(age=50, bmi=24.0, race1=5, race3=7))
    assert ada_points(d).tolist() == [2, 2]
    assert ada_points(d, asian_bmi=True).tolist() == [3, 2]


def test_uspstf():
    d = rows(dict(age=35, bmi=25.0), dict(age=70, bmi=30.0), dict(age=71, bmi=30.0),
             dict(age=34, bmi=30.0), dict(age=40, bmi=24.9),
             dict(age=40, bmi=24.0, race1=5, race3=6))
    assert uspstf_flag(d).tolist() == [1, 1, 0, 0, 0, 0]
    assert uspstf_flag(d, asian_bmi=True).tolist() == [1, 1, 0, 0, 0, 1]


def test_ada2026():
    d = rows(
        dict(age=36),                                        # everyone from 35
        dict(age=30, bmi=26.0, famhx=1),                     # overweight + family history
        dict(age=30, bmi=26.0),                              # overweight, no risk factor
        dict(age=30, bmi=24.0, race1=5, race3=6),            # Asian, BMI 23+, high-risk group
        dict(age=30, bmi=24.0, race1=4, race3=4),            # BMI under 25, not Asian
        dict(age=25, bmi=21.0, gdm=1),                       # gestational diabetes at any BMI
        dict(age=30, bmi=26.0, active_rec=0),                # overweight + inactive
        dict(age=30, bmi=26.0, race1=1, race3=1),            # overweight + Hispanic
    )
    assert ada2026_flag(d).tolist() == [1, 1, 0, 1, 0, 1, 1, 1]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed", name)
