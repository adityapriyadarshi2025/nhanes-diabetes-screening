import numpy as np


def design(frame):
    groups = {}
    for (s, _), rows in frame.reset_index(drop=True).groupby(["strata", "psu"]).indices.items():
        groups.setdefault(s, []).append(rows)
    return list(groups.values())


def replicate_weights(frame, n_rep=1000, seed=2026):
    """Rao-Wu bootstrap: per stratum draw n-1 of its n PSUs with replacement, rescale by n/(n-1)."""
    rng = np.random.default_rng(seed)
    groups = design(frame)
    w = frame["wt_mec"].to_numpy(float)
    reps = np.zeros((n_rep, len(w)))
    for r in range(n_rep):
        for psus in groups:
            n = len(psus)
            picks, times = np.unique(rng.integers(0, n, size=n - 1), return_counts=True)
            for j, k in zip(picks, times):
                reps[r, psus[j]] = w[psus[j]] * k * n / (n - 1)
    return reps


def threshold_for_rate(score, w, rate):
    order = np.argsort(-score, kind="stable")
    share = np.cumsum(w[order]) / w.sum()
    return score[order][min(np.searchsorted(share, rate), len(order) - 1)]


def flag_stats(y, flag, W):
    W = np.atleast_2d(W)
    y, flag = y.astype(float), flag.astype(float)
    tp, pos, flagged, total = W @ (y * flag), W @ y, W @ flag, W.sum(1)
    return {"sens": tp / pos, "ppv": tp / np.maximum(flagged, 1e-12), "flag_rate": flagged / total}


def _groups(score):
    order = np.argsort(-score, kind="stable")
    s = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(s) != 0) + 1]
    return order, starts


def auc(score, y, W):
    W = np.atleast_2d(W)
    order, starts = _groups(score)
    yo = y[order].astype(float)
    pos = np.add.reduceat(W[:, order] * yo, starts, axis=1)
    neg = np.add.reduceat(W[:, order] * (1 - yo), starts, axis=1)
    neg_above = np.cumsum(neg, axis=1) - neg
    P, N = pos.sum(1), neg.sum(1)
    return (pos * (N[:, None] - neg_above - neg) + 0.5 * pos * neg).sum(1) / (P * N)


def average_precision(score, y, W):
    W = np.atleast_2d(W)
    order, starts = _groups(score)
    yo = y[order].astype(float)
    tp = np.cumsum(np.add.reduceat(W[:, order] * yo, starts, axis=1), axis=1)
    fp = np.cumsum(np.add.reduceat(W[:, order] * (1 - yo), starts, axis=1), axis=1)
    recall = tp / tp[:, -1:]
    precision = tp / np.maximum(tp + fp, 1e-12)
    step = np.diff(np.c_[np.zeros(len(W)), recall], axis=1)
    return (step * precision).sum(1)


def brier(p, y, W):
    W = np.atleast_2d(W)
    return (W @ (y - p) ** 2) / W.sum(1)


def observed_expected(p, y, W):
    W = np.atleast_2d(W)
    return (W @ y.astype(float)) / (W @ p)


def logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def logistic_fit(X, y, w, offset=0.0, iters=100):
    w = w / w.mean()
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(offset + X @ beta)))
        g = X.T @ (w * (y - p))
        H = (X * (w * p * (1 - p))[:, None]).T @ X
        step = np.linalg.solve(H, g)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def calibration(p, y, w):
    lp = logit(p)
    slope = logistic_fit(np.c_[np.ones_like(lp), lp], y.astype(float), w)[1]
    intercept = logistic_fit(np.ones((len(lp), 1)), y.astype(float), w, offset=lp)[0]
    return intercept, slope


def calibration_reps(p, y, W, n=None):
    W = np.atleast_2d(W)[:n]
    return np.array([calibration(p, y, w) for w in W])


def ci(values, point=None):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return {"est": float(np.median(values) if point is None else point), "lo": float(lo), "hi": float(hi)}
