"""
Step 3: Barra USE4 style factor exposures.

Builds the raw descriptors, cleans and scales them the USE4 way, combines
them into 12 style factors, and orthogonalizes the ones USE4 orthogonalizes.

Scaling, applied cross-sectionally on every date:
  1. Winsorize at +/- 3 standard deviations.
  2. Standardize to cap-weighted mean 0, equal-weighted sd 1. This is why a
     cap-weighted benchmark has zero exposure to every style, and why VOO
     drops out of the active exposure calculation almost entirely.
  3. Fill gaps with the sector mean, then 0 (the cap-weighted mean).

Three factors depart from published USE4 because the data cannot support
the original descriptors. See 00_PLAN.md section 2.3 and 2.5:
  Earnings Yield  loses its forecast E/P, weights renormalized onto trailing.
  Growth          loses both forecast descriptors, historical only.
  Leverage        no preferred equity, no long/short debt split.

Outputs to ./cache/exposures.pkl
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

TRADING_DAYS = 252
MONTH = 21
LAG_DAYS = 63          # must match 01_data.py

# USE4 half-lives, in trading days
HL_BETA = 63
HL_MOMENTUM = 126
HL_DASTD = 42
HL_HSIGMA = 63

WIN_BETA = 252
WIN_MOMENTUM = 504
MOMENTUM_SKIP = 21

STYLE_WEIGHTS = {
    "Beta":          {"BETA": 1.0},
    "Momentum":      {"RSTR": 1.0},
    "Size":          {"LNCAP": 1.0},
    "ResidVol":      {"DASTD": 0.74, "CMRA": 0.16, "HSIGMA": 0.10},
    "BookToPrice":   {"BTOP": 1.0},
    "EarningsYield": {"CETOP": 0.66, "ETOP": 0.34},
    "Growth":        {"SGRO": 0.66, "EGRO": 0.34},
    "Leverage":      {"MLEV": 0.38, "DTOA": 0.35, "BLEV": 0.27},
    "Liquidity":     {"STOM": 0.35, "STOQ": 0.35, "STOA": 0.30},
    "DividendYield": {"DTOP": 1.0},
}

# Styles built by cubing another style rather than from raw descriptors.
# The cube MUST be taken on the standardized parent, not the raw descriptor.
# Log market cap runs 9.0 to 15.4, so it sits far from zero and cubing it is
# 98.8% linear: there is essentially no non-linear content to isolate, and
# orthogonalizing against Size leaves noise that reintroduces a -0.46
# correlation once rescaled. Cubing the standardized parent, which is
# centered at 0 with sd 1, gives a genuine 0.70 correlation with real
# curvature to keep. Same argument for Beta, which centers near 1.
DERIVED_STYLES = {
    "NonLinearSize": "Size",
    "NonLinearBeta": "Beta",
}

# factor -> factors it is orthogonalized against, in USE4
ORTHOGONALIZE = {
    "NonLinearSize": ["Size"],
    "NonLinearBeta": ["Beta"],
    "ResidVol": ["Beta", "Size"],
    "Liquidity": ["Size"],
}


# --- scaling helpers --------------------------------------------------------

def winsorize(df, n_sd=3.0):
    """Clip each cross-section at +/- n_sd about its own mean."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1)
    lo = (mu - n_sd * sd)
    hi = (mu + n_sd * sd)
    return df.clip(lower=lo, upper=hi, axis=0)


def standardize(df, mcap):
    """Cap-weighted mean 0, equal-weighted sd 1, cross-sectionally."""
    w = mcap.where(df.notna())
    w = w.div(w.sum(axis=1), axis=0)
    mu = (df * w).sum(axis=1, min_count=1)
    out = df.sub(mu, axis=0)
    sd = out.std(axis=1).replace(0.0, np.nan)
    return out.div(sd, axis=0)


def fill_by_sector(df, sector_map):
    """USE4 fills missing exposures from peers. Sector mean, then 0."""
    out = df.copy()
    for sec, members in sector_map.groupby(sector_map).groups.items():
        cols = [c for c in members if c in out.columns]
        if not cols:
            continue
        sec_mean = out[cols].mean(axis=1)
        out[cols] = out[cols].apply(lambda s: s.fillna(sec_mean))
    return out.fillna(0.0)


def scale(df, mcap, sector_map):
    return fill_by_sector(standardize(winsorize(df), mcap), sector_map)


def orthogonalize(target, others, weights):
    """
    Weighted cross-sectional regression of target on others; keep the
    residual. Prevents Residual Volatility from double-counting Beta and
    Size, and keeps the non-linear factors from restating their parents.

    `weights` must be sqrt(market cap), matching the factor-return
    regression in step 4, NOT market cap itself. Orthogonality only holds in
    the metric you impose it in, and cap weight here is savagely
    concentrated: the top 10 names are 44.5% of the universe. Using raw cap
    weight makes NonLinearSize exactly orthogonal to Size among the
    mega-caps while leaving a 0.72 correlation for the typical stock, which
    is a near-collinear column in the very regression this feeds. Weighting
    by sqrt(cap) instead puts the residual at right angles to Size in the
    metric step 4 actually solves in, which is what conditioning depends on.
    """
    resid = pd.DataFrame(np.nan, index=target.index, columns=target.columns)
    X_all = [o for o in others]
    for dt in target.index:
        y = target.loc[dt]
        X = pd.concat([o.loc[dt] for o in X_all], axis=1)
        w = weights.loc[dt]
        m = y.notna() & X.notna().all(axis=1) & w.notna() & (w > 0)
        if m.sum() < 30:
            continue
        Xm = np.column_stack([np.ones(m.sum()), X[m].values])
        sw = np.sqrt(w[m].values)
        b = np.linalg.lstsq(Xm * sw[:, None], y[m].values * sw, rcond=None)[0]
        resid.loc[dt, y[m].index] = y[m].values - Xm @ b
    return resid


# --- descriptor builders ----------------------------------------------------

def beta_and_residuals(excess, mkt_excess):
    """
    EWMA market model. Returns beta, and the residual series needed for
    HSIGMA. Vectorized across all names: a 395-name rolling OLS loop over
    4,300 dates is far too slow, and the EWMA covariance ratio is what USE4
    specifies anyway.
    """
    kw = dict(halflife=HL_BETA, min_periods=WIN_BETA)
    m = mkt_excess
    m_mean = m.ewm(**kw).mean()
    m_var = m.ewm(**kw).var()

    r_mean = excess.ewm(**kw).mean()
    cov = excess.mul(m, axis=0).ewm(**kw).mean().sub(r_mean.mul(m_mean, axis=0))

    beta = cov.div(m_var.replace(0.0, np.nan), axis=0)
    alpha = r_mean.sub(beta.mul(m_mean, axis=0))
    resid = excess.sub(alpha).sub(beta.mul(m, axis=0))
    return beta, resid


def cmra(excess):
    """
    Cumulative range: the spread between the best and worst point of the
    trailing 12-month cumulative log return path. Sampled monthly.
    """
    logret = np.log1p(excess.clip(lower=-0.99))
    monthly = logret.rolling(MONTH).sum().iloc[::1]
    cum = [monthly.shift(MONTH * k) for k in range(12)]
    stack = pd.concat(cum, axis=0, keys=range(12))
    zmax = stack.groupby(level=1).max()
    zmin = stack.groupby(level=1).min()
    return zmax - zmin


def annual_growth(level, n_years=5):
    """
    USE4 growth: slope of the annual series regressed on time, scaled by the
    mean level. Handles sign changes and zero crossings better than a CAGR,
    which is essential for earnings.
    """
    pts = [level.shift(TRADING_DAYS * k) for k in range(n_years)]
    x = np.arange(n_years - 1, -1, -1, dtype=float)
    x = x - x.mean()
    num = sum(x[k] * pts[k] for k in range(n_years))
    slope = num / (x ** 2).sum()
    mean_level = sum(pts) / n_years
    return slope / mean_level.abs().replace(0.0, np.nan)


def ttm(raw, calendar):
    """
    Trailing-twelve-month sum of a quarterly flow, built from the raw stamps
    rather than the daily forward-filled panel, then re-filled and lagged.
    Summing the daily panel directly would double-count every quarter.
    """
    out = {}
    for col in raw.columns:
        s = raw[col].dropna()
        if s.empty:
            continue
        # collapse repeated stamps of the same quarterly figure
        q = s[s != s.shift(1)]
        out[col] = q.rolling(4).sum()
    if not out:
        return pd.DataFrame(index=calendar)
    df = pd.DataFrame(out).reindex(calendar.union(
        pd.DatetimeIndex(sorted({d for v in out.values() for d in v.index}))
    )).ffill().reindex(calendar)
    return df.shift(LAG_DAYS)


def main():
    panel = pd.read_pickle(os.path.join(CACHE, "panel.pkl"))
    cal = panel["calendar"]
    stocks = panel["stocks"]

    sectors = pd.read_csv(os.path.join(CACHE, "sectors.csv"), index_col=0)
    sector_map = sectors["sector"].reindex(stocks)

    mcap = panel["mcap"][stocks]
    px = panel["px"][stocks]
    excess = panel["excess"][stocks]
    fund = {k: v[stocks] for k, v in panel["fundamentals"].items()}
    raw = {k: v[stocks] for k, v in panel["fundamentals_raw"].items()}

    print(f"Building descriptors: {len(stocks)} stocks, {len(cal)} days")

    d = {}

    # --- price-based ---
    print("  beta, residual volatility")
    beta, resid = beta_and_residuals(excess, panel["mkt_excess"])
    d["BETA"] = beta
    d["HSIGMA"] = resid.ewm(halflife=HL_HSIGMA, min_periods=WIN_BETA).std()
    d["DASTD"] = excess.ewm(halflife=HL_DASTD, min_periods=WIN_BETA).std()
    d["CMRA"] = cmra(excess)

    print("  momentum")
    logret = np.log1p(excess.clip(lower=-0.99))
    d["RSTR"] = (logret.shift(MOMENTUM_SKIP)
                 .ewm(halflife=HL_MOMENTUM, min_periods=WIN_MOMENTUM - MOMENTUM_SKIP)
                 .mean())

    print("  size")
    d["LNCAP"] = np.log(mcap.replace(0.0, np.nan))

    print("  liquidity")
    shares = panel["shares"][stocks]
    turnover = (panel["volume"][stocks] / shares.replace(0.0, np.nan))
    for name, months in [("STOM", 1), ("STOQ", 3), ("STOA", 12)]:
        win = MONTH * months
        d[name] = np.log(turnover.rolling(win, min_periods=win // 2)
                         .mean().replace(0.0, np.nan))

    # --- fundamental ---
    print("  value, earnings, growth, leverage, yield")
    book = fund["TOTAL_EQUITY"]
    debt = fund["SHORT_AND_LONG_TERM_DEBT"]
    assets = fund["BS_TOT_ASSET"]

    d["BTOP"] = book / mcap

    ni_ttm = ttm(raw["NET_INCOME"], cal)
    cfo_ttm = ttm(raw["CF_CASH_FROM_OPER"], cal)
    rev_ttm = ttm(raw["SALES_REV_TURN"], cal)
    dps_ttm = ttm(raw["IS_DIV_PER_SHR"], cal)

    d["ETOP"] = ni_ttm / mcap
    d["CETOP"] = cfo_ttm / mcap
    d["DTOP"] = dps_ttm / px

    d["EGRO"] = annual_growth(ni_ttm)
    d["SGRO"] = annual_growth(rev_ttm)

    d["MLEV"] = (mcap + debt) / mcap
    d["DTOA"] = debt / assets.replace(0.0, np.nan)
    d["BLEV"] = (book + debt) / book.replace(0.0, np.nan)

    # --- scale descriptors, combine into styles ---
    print("  scaling and combining")
    d = {k: v.replace([np.inf, -np.inf], np.nan).reindex(
        index=cal, columns=stocks) for k, v in d.items()}
    scaled = {k: scale(v, mcap, sector_map) for k, v in d.items()}

    styles = {}
    for style, weights in STYLE_WEIGHTS.items():
        combo = sum(w * scaled[desc] for desc, w in weights.items())
        styles[style] = scale(combo, mcap, sector_map)

    # Derived styles cube the already-standardized parent. See DERIVED_STYLES.
    for style, parent in DERIVED_STYLES.items():
        styles[style] = scale(styles[parent] ** 3, mcap, sector_map)

    # --- orthogonalize, in dependency order ---
    # Rescaling afterwards deliberately skips the winsorize step. Clipping the
    # tails of a residual is not sign-neutral and tilts it back towards the
    # variable it was just orthogonalized against, which is enough to leak a
    # visible correlation back in. The inputs were already winsorized, so the
    # residual has no room to blow up.
    print("  orthogonalizing")
    reg_weights = np.sqrt(mcap)       # same weights as the step 4 regression
    for style, against in ORTHOGONALIZE.items():
        others = [styles[a] for a in against]
        resid = orthogonalize(styles[style], others, reg_weights)
        styles[style] = fill_by_sector(
            standardize(resid, mcap), sector_map)
        print(f"    {style} vs {', '.join(against)}")

    out = {
        "styles": styles,
        "descriptors": scaled,
        "sector_map": sector_map,
        "stocks": stocks,
        "calendar": cal,
    }
    path = os.path.join(CACHE, "exposures.pkl")
    pd.to_pickle(out, path)

    # --- diagnostics ---
    print("\n--- exposure checks (2015 onward) ---")
    w = mcap.div(mcap.sum(axis=1), axis=0)
    rows = []
    for s, X in styles.items():
        Xs = X.loc["2015":]
        cw = (Xs * w.loc["2015":]).sum(axis=1)
        rows.append((s, Xs.std(axis=1).mean(), cw.mean(),
                     Xs.notna().mean(axis=1).mean()))
    rep = pd.DataFrame(rows, columns=["style", "xs_sd", "capwtd_mean", "coverage"])
    print(rep.round(4).to_string(index=False))
    print("\n  xs_sd should be ~1.00, capwtd_mean ~0.00")

    print("\n--- orthogonality, in the weighting step 4 solves in ---")
    last = cal[-1]
    sw = np.sqrt(mcap.loc[last])
    for style, against in ORTHOGONALIZE.items():
        for a in against:
            y, x = styles[style].loc[last], styles[a].loc[last]
            m = y.notna() & x.notna() & sw.notna()
            wv = sw[m].values
            yc = y[m].values - np.average(y[m].values, weights=wv)
            xc = x[m].values - np.average(x[m].values, weights=wv)
            r = (np.average(yc * xc, weights=wv)
                 / np.sqrt(np.average(yc ** 2, weights=wv)
                           * np.average(xc ** 2, weights=wv)))
            ew = np.corrcoef(y[m].values, x[m].values)[0, 1]
            print(f"  {style:14s} vs {a:6s}  sqrt-cap-wtd {r:+.3f}   "
                  f"equal-wtd {ew:+.3f}")

    print("\n--- exposure matrix conditioning, last date ---")
    order = list(styles.keys())
    Xm = pd.DataFrame({s: styles[s].loc[last] for s in order})
    wv = np.sqrt(mcap.loc[last])
    keep = Xm.notna().all(axis=1) & wv.notna() & (wv > 0)
    A = Xm[keep].values * np.sqrt(wv[keep].values)[:, None]
    print(f"  {keep.sum()} usable names")
    print(f"  condition number of weighted style block: "
          f"{np.linalg.cond(A):.1f}   [under ~30 is comfortable]")

    print("\n--- style correlations, last date ---")
    C = pd.DataFrame({s: X.loc[last] for s, X in styles.items()}).corr()
    print(C.round(2).to_string())

    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
