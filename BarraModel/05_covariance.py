"""
Step 5: Factor covariance matrix F and specific variances Delta.

F is built USE4-style: volatilities and correlations get different half-lives
because correlations are noisier and need a longer memory. Both get a
Newey-West adjustment, because daily factor returns are autocorrelated and
ignoring that understates risk once you scale to a monthly or annual horizon.

Specific variances are EWMA on the regression residuals, then shrunk toward
the average of similar-sized stocks. The shrinkage matters more here than in
a commercial model: with 395 names some residual series are short or noisy,
and an unshrunk estimate hands the portfolio report a confidently wrong
number for whichever stock happened to be quiet last quarter.

Not implemented, deliberately, and both would RAISE the risk forecast:
  - Eigenfactor risk adjustment. Sample covariance matrices systematically
    understate the risk of optimized portfolios.
  - Volatility regime adjustment. Scales the forecast by how recent realized
    risk compared to prediction.
Neither changes the risk *decomposition*, which is what the report is for.

Outputs to ./cache/covariance.pkl
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

HL_VOL = 84        # USE4 short-horizon volatility half-life, days
HL_CORR = 504      # correlations are noisier, longer memory
HL_SPEC = 84
NW_LAGS = 2
WINDOW = 756       # 3 years of daily data per estimate
MIN_OBS = 252
ANNUALIZE = 252

SHRINK_Q = 0.1     # USE4 Bayesian shrinkage intensity
N_SIZE_GROUPS = 10


def ewma_weights(n, halflife):
    w = 0.5 ** (np.arange(n - 1, -1, -1) / halflife)
    return w / w.sum()


def ewma_nw_cov(R, halflife, lags=NW_LAGS):
    """
    Exponentially weighted covariance with a Newey-West serial-correlation
    adjustment. R is T x K, most recent observation last.
    """
    T, K = R.shape
    w = ewma_weights(T, halflife)
    mu = (w[:, None] * R).sum(axis=0)
    X = R - mu

    C = (X * w[:, None]).T @ X

    for d in range(1, lags + 1):
        if T - d < 2:
            break
        wd = ewma_weights(T - d, halflife)
        Cd = (X[:-d] * wd[:, None]).T @ X[d:]
        C = C + (1.0 - d / (lags + 1.0)) * (Cd + Cd.T)
    return C


def cov_to_corr(C):
    s = np.sqrt(np.clip(np.diag(C), 1e-16, None))
    R = C / np.outer(s, s)
    np.fill_diagonal(R, 1.0)
    return np.clip(R, -1.0, 1.0)


def nearest_psd(C, floor=1e-12):
    """Clip negative eigenvalues. Newey-West does not guarantee PSD."""
    vals, vecs = np.linalg.eigh((C + C.T) / 2.0)
    if (vals > 0).all():
        return C
    vals = np.clip(vals, floor, None)
    return vecs @ np.diag(vals) @ vecs.T


def build_factor_cov(F_hist):
    """
    Volatilities at the short half-life, correlations at the long one,
    recombined. This is what USE4 does and it matters: using one half-life
    for both either makes correlations jumpy or makes volatility stale.
    """
    R = F_hist.values
    C_vol = ewma_nw_cov(R, HL_VOL)
    C_cor = ewma_nw_cov(R, HL_CORR)

    vol = np.sqrt(np.clip(np.diag(C_vol), 1e-16, None))
    corr = cov_to_corr(nearest_psd(C_cor))

    C = np.outer(vol, vol) * corr
    return nearest_psd(C) * ANNUALIZE


def build_specific_var(U_hist, mcap_now):
    """
    EWMA specific variance with Newey-West, then Bayesian shrinkage toward
    the mean of the stock's size decile, per USE4:

        v = q|s - s_bar| / (delta_bar + q|s - s_bar|)
        s_shrunk = v * s_bar + (1 - v) * s

    An estimate far from its size peers gets pulled harder, which is the
    right behaviour: extreme specific-risk estimates are usually noise.
    """
    T = len(U_hist)
    w = ewma_weights(T, HL_SPEC)

    out = {}
    for col in U_hist.columns:
        u = U_hist[col]
        m = u.notna().values
        if m.sum() < MIN_OBS:
            continue
        uu = u.values[m]
        ww = ewma_weights(m.sum(), HL_SPEC)
        mu = (ww * uu).sum()
        x = uu - mu
        v = (ww * x * x).sum()
        for d in range(1, NW_LAGS + 1):
            if len(x) - d < 2:
                break
            wd = ewma_weights(len(x) - d, HL_SPEC)
            cd = (wd * x[:-d] * x[d:]).sum()
            v += 2.0 * (1.0 - d / (NW_LAGS + 1.0)) * cd
        out[col] = max(v, 1e-10)

    s = pd.Series(out) ** 0.5          # daily specific vol
    cap = mcap_now.reindex(s.index)
    ok = cap.notna() & (cap > 0)
    s, cap = s[ok], cap[ok]

    groups = pd.qcut(np.log(cap), N_SIZE_GROUPS, labels=False,
                     duplicates="drop")
    s_bar = s.groupby(groups).transform("mean")
    delta_bar = s.groupby(groups).transform("std")

    gap = (s - s_bar).abs()
    v = (SHRINK_Q * gap) / (delta_bar + SHRINK_Q * gap)
    v = v.fillna(0.0).clip(0.0, 1.0)
    s_shrunk = v * s_bar + (1.0 - v) * s

    return (s_shrunk ** 2) * ANNUALIZE, v


def main():
    panel = pd.read_pickle(os.path.join(CACHE, "panel.pkl"))
    fr = pd.read_pickle(os.path.join(CACHE, "factor_returns.pkl"))

    F = fr["factor_returns"]
    U = fr["specific_returns"]
    factors = list(F.columns)
    mcap = panel["mcap"]

    dates = F.index
    last = dates[-1]
    hist = F.loc[:last].tail(WINDOW)
    print(f"Estimating covariance as of {last.date()} "
          f"from {len(hist)} trading days")

    Fcov = build_factor_cov(hist)
    Fcov = pd.DataFrame(Fcov, index=factors, columns=factors)

    U_hist = U.loc[:last].tail(WINDOW)
    spec_var, shrink = build_specific_var(U_hist, mcap.loc[last])

    out = {
        "factor_cov": Fcov,
        "specific_var": spec_var,
        "shrinkage": shrink,
        "as_of": last,
        "factors": factors,
    }
    path = os.path.join(CACHE, "covariance.pkl")
    pd.to_pickle(out, path)

    # --- diagnostics ---
    vols = pd.Series(np.sqrt(np.diag(Fcov)), index=factors)
    print("\n--- annualized factor volatility ---")
    print(vols.sort_values(ascending=False).round(4).to_string())

    ev = np.linalg.eigvalsh(Fcov.values)
    print(f"\n  smallest eigenvalue: {ev.min():.3e}  "
          f"(positive definite: {ev.min() > 0})")
    print(f"  condition number: {ev.max() / max(ev.min(), 1e-300):.1f}")

    ss = np.sqrt(spec_var)
    print(f"\n--- specific volatility, annualized, {len(ss)} stocks ---")
    print(f"  median {ss.median():.2%}   "
          f"10th pct {ss.quantile(0.1):.2%}   "
          f"90th pct {ss.quantile(0.9):.2%}")
    print(f"  mean shrinkage applied: {shrink.mean():.3f}")

    print("\n--- factor correlation, selected ---")
    corr = pd.DataFrame(cov_to_corr(Fcov.values), index=factors,
                        columns=factors)
    sel = ["Country", "Technology", "Energy", "Beta", "Momentum", "Size",
           "BookToPrice"]
    sel = [s for s in sel if s in corr.columns]
    print(corr.loc[sel, sel].round(2).to_string())

    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
