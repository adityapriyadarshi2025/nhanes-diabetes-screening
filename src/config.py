"""Shared settings: folders, survey cycles and the role of each cycle."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
CODEBOOKS = ROOT / "data" / "codebooks"
RESULTS = ROOT / "results"

# cycle label -> file-name pattern on the CDC site ({} = component code)
CYCLES = {
    "2007-2008": "{}_E",
    "2009-2010": "{}_F",
    "2011-2012": "{}_G",
    "2013-2014": "{}_H",
    "2015-2016": "{}_I",
    "2017-2020": "P_{}",
}
COMPONENTS = ["DEMO", "GHB", "DIQ", "BMX", "BPQ", "MCQ", "RHQ", "PAQ",
              "SMQ", "WHQ", "GLU", "HDL"]

# Roles fixed in PREREG.md: train 2007-2014, calibrate and set thresholds on
# 2015-2016, test once on 2017-March 2020.
DEV_CYCLES = ["2007-2008", "2009-2010", "2011-2012", "2013-2014", "2015-2016"]
TRAIN_CYCLES = DEV_CYCLES[:4]
CALIB_CYCLE = "2015-2016"
TEST_CYCLE = "2017-2020"
# lab outcomes of the test cycle, kept in a separate sealed file until the freeze
SEALED_COLS = ["hba1c", "fpg", "wt_fast"]

# HbA1c laboratory by cycle (CDC lab documentation)
HBA1C_LAB = {
    "2007-2008": "Minnesota",
    "2009-2010": "Minnesota",
    "2011-2012": "Minnesota",
    "2013-2014": "Missouri",
    "2015-2016": "Missouri",
    "2017-2020": "Missouri",
}
