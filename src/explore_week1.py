"""Week-1 data checks.

Development cycles (2007-2016): cohort size, HbA1c and the two targets by
cycle, crude and age-standardised, with survey-design 95% intervals.
All cycles: missing values, input means and screening-rule flag rates.
The test cycle (2017-March 2020) is described from inputs only: its HbA1c
values stay sealed until the models are frozen.

Writes CSV tables and one figure to results/week1/.
Run from the project folder:  python -m src.explore_week1
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import DEV_CYCLES, HBA1C_LAB, PROCESSED, RESULTS, TEST_CYCLE  # noqa: E402

OUT = RESULTS / "week1"
AGE_EDGES = [20, 35, 50, 65, 200]
AGE_LABELS = ["20-34", "35-49", "50-64", "65+"]
INPUTS = ["age", "male", "bmi", "waist", "whtr", "arm", "leg", "sr_bmi", "htn_told", "bp_med",
          "famhx", "cvd", "gdm", "smoking", "active_any", "active_rec", "mvpa_min_wk",
          "sedentary_min", "educ", "pir", "race1", "race3"]
N_BOOT = 500


# ------------------------------------------------------------ survey helpers
def band_index(age):
    return np.searchsorted(AGE_EDGES, age.to_numpy(), side="right") - 1


def rao_wu(groups, w, rng):
    """One Rao-Wu bootstrap replicate: in each stratum draw n-1 of its n PSUs with
    replacement and rescale weights by n/(n-1) x times drawn."""
    out = np.zeros_like(w)
    for psus in groups.values():
        n = len(psus)
        picks = rng.integers(0, n, size=n - 1)
        for j, times in zip(*np.unique(picks, return_counts=True)):
            rows = psus[j]
            out[rows] = w[rows] * times * n / (n - 1)
    return out


def design(frame):
    """{stratum: [row indices of each PSU]} for one cycle."""
    groups = {}
    for (s, p), rows in frame.groupby(["strata", "psu"]).indices.items():
        groups.setdefault(s, []).append(rows)
    return groups


def age_std(values, w, band, std):
    """Directly age-standardised weighted mean of `values`."""
    num = np.bincount(band, weights=w * values, minlength=len(std))
    den = np.bincount(band, weights=w, minlength=len(std))
    return float(np.sum(std * num / den))


# ------------------------------------------------------------ outcome by cycle (dev only)
def outcome_by_cycle(dev, rng):
    std_w = dev["wt_mec"].to_numpy()
    std = np.bincount(band_index(dev["age"]), weights=std_w, minlength=4) / std_w.sum()
    rows, curves = [], []
    for cyc in DEV_CYCLES:
        f = dev[dev["cycle"] == cyc].reset_index(drop=True)
        w, band = f["wt_mec"].to_numpy(), band_index(f["age"])
        a1c, y57, y65 = (f[c].to_numpy(dtype=float) for c in ("hba1c", "y57", "y65"))
        stats = lambda ww: [age_std(a1c, ww, band, std), 100 * age_std(y57, ww, band, std),  # noqa: E731
                            100 * age_std(y65, ww, band, std)]
        point = stats(w)
        groups = design(f)
        boots = np.array([stats(rao_wu(groups, w, rng)) for _ in range(N_BOOT)])
        lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
        rows.append({
            "cycle": cyc, "lab": HBA1C_LAB[cyc], "n": len(f),
            "n_hba1c_5.7_plus": int(y57.sum()), "n_hba1c_6.5_plus": int(y65.sum()),
            "pct_5.7_plus_crude": 100 * np.average(y57, weights=w),
            "pct_6.5_plus_crude": 100 * np.average(y65, weights=w),
            "mean_hba1c_crude": np.average(a1c, weights=w),
            "mean_hba1c_agestd": point[0], "mean_lo": lo[0], "mean_hi": hi[0],
            "pct_5.7_plus_agestd": point[1], "pct57_lo": lo[1], "pct57_hi": hi[1],
            "pct_6.5_plus_agestd": point[2], "pct65_lo": lo[2], "pct65_hi": hi[2],
        })
        values = np.round(np.arange(4.0, 7.51, 0.1), 1)
        share = [100 * w[np.isclose(a1c, v)].sum() / w.sum() for v in values]
        curves.append(pd.Series(share, index=values, name=cyc))
    return pd.DataFrame(rows), pd.concat(curves, axis=1)


# ------------------------------------------------------------ inputs, all cycles
def by_cycle(frame, fn):
    return pd.DataFrame({c: fn(g) for c, g in frame.groupby("cycle")})


def weighted_means(g):
    w = g["wt_mec"]
    out = {}
    for v in INPUTS:
        if v in ("race1", "race3"):
            continue
        m = g[v].notna()
        out[v] = np.average(g.loc[m, v], weights=w[m]) if m.any() else np.nan
    out["smoking_current"] = np.average(g["smoking"].eq(2)[g["smoking"].notna()],
                                        weights=w[g["smoking"].notna()])
    return pd.Series(out)


def rule_rates(g):
    w = g["wt_mec"]
    age_ok = g["age"].between(35, 70)
    asian = g["race3"] == 6
    rules = {
        "USPSTF 2021: age 35-70 and BMI 25+": age_ok & (g["bmi"] >= 25),
        "USPSTF with BMI 23+ for Asian adults": age_ok & (g["bmi"] >= np.where(asian, 23, 25)),
        "Age 35-70 alone": age_ok,
    }
    out = {k: 100 * np.average(v, weights=w) for k, v in rules.items()}
    out["share Asian (race3 = 6)"] = 100 * np.average(asian, weights=w) if g["race3"].notna().any() else np.nan
    out["BMI missing"] = 100 * g["bmi"].isna().mean()
    return pd.Series(out)


# ------------------------------------------------------------ figure
def figure(table, path):
    ink, ink2, muted, grid, blue, surface = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#2a78d6", "#fcfcfb"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=surface)
    x = np.arange(len(table))
    panels = [("mean_hba1c_agestd", "mean_lo", "mean_hi", "Mean HbA1c (%)", "{:.2f}"),
              ("pct_5.7_plus_agestd", "pct57_lo", "pct57_hi", "Share with HbA1c 5.7% or more (%)", "{:.1f}")]
    for ax, (mid, lo, hi, title, fmt) in zip(axes, panels):
        ax.set_facecolor(surface)
        ax.axvspan(2.5, 4.5, color="#f0efec", zorder=0)
        ax.errorbar(x, table[mid], yerr=[table[mid] - table[lo], table[hi] - table[mid]],
                    fmt="-o", color=blue, lw=2, ms=7, mec=surface, mew=2, ecolor=blue, elinewidth=1.5,
                    capsize=0, zorder=3)
        for xi, v in zip(x, table[mid]):
            ax.annotate(fmt.format(v), (xi, v), textcoords="offset points", xytext=(9, -3),
                        fontsize=9, color=ink2)
        ax.set_xticks(x, [c.replace("-20", "-") for c in table["cycle"]], color=ink2)
        ax.set_title(title, loc="left", fontsize=11, color=ink, pad=24)
        ax.grid(axis="y", color=grid, lw=1)
        ax.tick_params(colors=muted, length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        top = ax.get_xaxis_transform()          # x in data units, y in axes fraction
        ax.text(1.0, 1.01, "Minnesota lab", transform=top, ha="center", va="bottom", fontsize=8.5, color=muted)
        ax.text(3.5, 1.01, "Missouri lab", transform=top, ha="center", va="bottom", fontsize=8.5, color=muted)
    fig.suptitle("HbA1c by survey cycle, adults without diagnosed diabetes (age-standardised, 95% CI)",
                 x=0.01, ha="left", fontsize=12, color=ink)
    fig.text(0.01, 0.01, "NHANES 2007-2016, survey-weighted. 2017-March 2020 is sealed until the models are frozen.",
             fontsize=8.5, color=muted)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor=surface)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    dev = pd.read_parquet(PROCESSED / "cohort_dev.parquet")
    test = pd.read_parquet(PROCESSED / "cohort_test_inputs.parquet")
    assert "hba1c" not in test.columns, "test cycle outcomes must stay sealed"
    both = pd.concat([dev, test], ignore_index=True)

    table, dist = outcome_by_cycle(dev, rng)
    table.round(3).to_csv(OUT / "outcome_by_cycle_dev.csv", index=False)
    dist.round(3).to_csv(OUT / "hba1c_distribution_dev.csv", index_label="hba1c")
    figure(table, OUT / "fig_hba1c_by_cycle.png")

    missing = by_cycle(both, lambda g: 100 * g[INPUTS].isna().mean()).round(1)
    missing.to_csv(OUT / "missing_pct_by_cycle.csv", index_label="input")
    means = by_cycle(both, weighted_means).round(3)
    means.to_csv(OUT / "input_means_by_cycle.csv", index_label="input")
    rules = by_cycle(both, rule_rates).round(1)
    rules.to_csv(OUT / "rule_flag_rates_by_cycle.csv", index_label="rule")

    pd.set_option("display.width", 200)
    cols = ["cycle", "lab", "n", "n_hba1c_5.7_plus", "n_hba1c_6.5_plus", "pct_5.7_plus_crude",
            "pct_6.5_plus_crude", "mean_hba1c_agestd", "pct_5.7_plus_agestd", "pct57_lo", "pct57_hi"]
    print(table[cols].round(2).to_string(index=False))
    print("\nScreening-rule flag rates (% of cohort, weighted):\n", rules.to_string())
    print("\nMissing (% of cohort):\n", missing.to_string())


if __name__ == "__main__":
    main()
