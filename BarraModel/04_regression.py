"""
Step 4: Daily constrained cross-sectional WLS. The heart of the model.

Each day, regress every stock's excess return on its factor exposures:

    r_n = f_country + sum_s X_ns f_s + sum_k X_nk f_k + u_n

with sector dummies X_ns, style exposures X_nk, and a country column of
ones. Exposures are lagged one day so the regression never uses a number
that was not knowable before the return happened.

Two details make this work:

WEIGHTS. Regression weights are proportional to sqrt(market cap). Specific
variance is roughly inversely proportional to sqrt(cap), so this is the
heteroskedasticity correction, and it stops micro-caps dominating without
handing the fit to the mega-caps.

THE CONSTRAINT. The country column is the sum of the sector dummies, so the
design matrix is exactly singular. Barra resolves this by requiring the
cap-weighted sector factor returns to sum to zero, which makes the country
factor the cap-weighted market return and every sector return a deviation
from it. Rather than Lagrange multipliers, this is imposed as a change of
basis: drop the last sector column and rewrite the rest as

    Xs~ = Xs - (ws / wS) * XS      for s = 1 .. S-1

so the restricted problem is unconstrained and full rank. The dropped
sector's return is recovered afterwards from the constraint. Numerically
cleaner than the augmented system, and it cannot silently return a
least-norm solution when the matrix is near singular.

Outputs to ./cache/factor_returns.pkl
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

START = "2012-01-01"
MIN_STOCKS = 100          # skip dates with a thinner cross-section
MIN_SECTOR_MEMBERS = 3    # fold tiny sector buckets into the country factor


def solve_day(y, Xs, Xk, w, sector_caps):
    """
    One day's constrained WLS.

    y           returns, n
    Xs          sector dummies, n x S
    Xk          style exposures, n x K
    w           regression weights (sqrt cap), n
    sector_caps cap weight per sector, S, summing to 1

    Returns (country, sector returns S, style returns K, residuals n, r2).
    """
    n, S = Xs.shape
    K = Xk.shape[1]

    # change of basis: fold sector S into the others via the constraint
    wS = sector_caps[-1]
    ratio = sector_caps[:-1] / wS
    Xs_r = Xs[:, :-1] - Xs[:, [-1]] * ratio[None, :]

    A = np.column_stack([np.ones(n), Xs_r, Xk])
    sw = np.sqrt(w)
    Aw = A * sw[:, None]
    yw = y * sw

    coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)

    country = coef[0]
    f_sector_head = coef[1:1 + (S - 1)]
    f_style = coef[1 + (S - 1):]

    # recover the dropped sector from the constraint
    f_sector_last = -(sector_caps[:-1] * f_sector_head).sum() / wS
    f_sector = np.append(f_sector_head, f_sector_last)

    fitted = country + Xs @ f_sector + Xk @ f_style
    resid = y - fitted

    ss_res = (w * resid ** 2).sum()
    ss_tot = (w * (y - np.average(y, weights=w)) ** 2).sum()
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return country, f_sector, f_style, resid, r2


def main():
    panel = pd.read_pickle(os.path.join(CACHE, "panel.pkl"))
    exp = pd.read_pickle(os.path.join(CACHE, "exposures.pkl"))

    stocks = exp["stocks"]
    styles = exp["styles"]
    style_names = list(styles.keys())
    sector_map = exp["sector_map"]
    sectors = sorted(sector_map.dropna().unique())

    ret = panel["excess"][stocks]
    mcap = panel["mcap"][stocks]

    cal = [d for d in exp["calendar"] if d >= pd.Timestamp(START)]
    print(f"Regressing {len(cal)} days, {len(sectors)} sectors, "
          f"{len(style_names)} styles")

    # sector membership as a fixed n x S indicator
    sec_dummy = pd.DataFrame(
        {s: (sector_map == s).astype(float) for s in sectors},
        index=stocks)

    f_country, f_sec, f_sty, r2s, ns = {}, {}, {}, {}, {}
    skipped = []
    specific = pd.DataFrame(np.nan, index=pd.DatetimeIndex(cal),
                            columns=stocks, dtype="float32")

    prev = None
    for dt in cal:
        y_all = ret.loc[dt]
        if prev is None:
            prev = dt
            continue

        # exposures as of the prior close, returns over today
        prev_dt = prev
        prev = dt

        X_all = pd.DataFrame({k: styles[k].loc[prev_dt] for k in style_names})
        w_all = np.sqrt(mcap.loc[prev_dt])

        m = (y_all.notna() & w_all.notna() & (w_all > 0)
             & X_all.notna().all(axis=1) & sector_map.notna())
        if m.sum() < MIN_STOCKS:
            continue

        names = y_all.index[m]
        Xs = sec_dummy.loc[names, sectors].values
        Xk = X_all.loc[names].values
        y = y_all[names].values
        w = w_all[names].values
        # Cap weights for the constraint come from the PRIOR close, the same
        # point in time as the exposures and regression weights. Reading them
        # at dt instead lets a name with a missing market cap today produce a
        # NaN sector weight, which propagates through the change of basis and
        # takes down the whole solve with an opaque LAPACK error.
        cap = mcap.loc[prev_dt, names].values

        # drop sectors with too few names today, folding them into country
        counts = Xs.sum(axis=0)
        live = counts >= MIN_SECTOR_MEMBERS
        if not live.all():
            keep_rows = Xs[:, live].sum(axis=1) > 0
            Xs, Xk, y, w, cap = (Xs[keep_rows][:, live], Xk[keep_rows],
                                 y[keep_rows], w[keep_rows], cap[keep_rows])
            names = names[keep_rows]
        live_sectors = [s for s, k in zip(sectors, live) if k]

        # cap weight per sector, for the constraint
        sector_caps = np.array(
            [cap[Xs[:, j] > 0].sum() for j in range(Xs.shape[1])])
        sector_caps = sector_caps / sector_caps.sum()

        # A single non-finite entry anywhere takes LAPACK down with an error
        # that names neither the date nor the column. Skip and count instead.
        if not (np.isfinite(y).all() and np.isfinite(w).all()
                and np.isfinite(Xk).all() and np.isfinite(sector_caps).all()
                and (sector_caps > 0).all()):
            skipped.append(dt)
            continue

        c, fs, fk, resid, r2 = solve_day(y, Xs, Xk, w, sector_caps)

        f_country[dt] = c
        f_sec[dt] = pd.Series(fs, index=live_sectors)
        f_sty[dt] = pd.Series(fk, index=style_names)
        specific.loc[dt, names] = resid.astype("float32")
        r2s[dt] = r2
        ns[dt] = len(names)

    factors = pd.concat([
        pd.Series(f_country, name="Country").to_frame(),
        pd.DataFrame(f_sec).T,
        pd.DataFrame(f_sty).T,
    ], axis=1)
    factors.index = pd.DatetimeIndex(factors.index)

    out = {
        "factor_returns": factors,
        "specific_returns": specific.loc[factors.index],
        "r2": pd.Series(r2s),
        "n_stocks": pd.Series(ns),
        "sectors": sectors,
        "styles": style_names,
        "sector_map": sector_map,
    }
    path = os.path.join(CACHE, "factor_returns.pkl")
    pd.to_pickle(out, path)

    # --- diagnostics ---
    print(f"\n  {len(factors)} days solved, "
          f"median {pd.Series(ns).median():.0f} stocks per cross-section")
    if skipped:
        print(f"  skipped {len(skipped)} days with non-finite inputs, "
              f"first: {skipped[0].date()}, last: {skipped[-1].date()}")
    print(f"  median daily R-squared: {pd.Series(r2s).median():.3f}   "
          f"[15-45% is the usual range for a daily equity model]")

    print("\n--- factor return volatility, annualized ---")
    vol = (factors.std() * np.sqrt(252)).sort_values(ascending=False)
    print(vol.round(4).to_string())

    print("\n--- constraint check: cap-weighted sector returns sum to 0 ---")
    print(f"  mean abs daily sector sum: "
          f"{factors[out['sectors']].abs().sum(axis=1).mean():.6f} "
          f"(unweighted, for scale)")

    print("\n--- country factor vs cap-weighted market return ---")
    mkt = panel["mkt_excess"].reindex(factors.index)
    both = pd.concat([factors["Country"], mkt], axis=1).dropna()
    print(f"  correlation: {both.corr().iloc[0, 1]:.4f}   [expect > 0.99]")
    print(f"  country ann. vol {factors['Country'].std() * np.sqrt(252):.2%}"
          f"   market ann. vol {mkt.std() * np.sqrt(252):.2%}")

    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
