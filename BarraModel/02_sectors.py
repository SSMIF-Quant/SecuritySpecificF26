"""
Step 2: GICS sector map for the 397-stock estimation universe.

Primary source is yfinance, whose 11-sector taxonomy maps 1:1 onto GICS.
That map is then audited against a multivariate regression of each stock's
market-relative return on the 11 sector-ETF market-relative returns. In
testing on 38 hand-known tickers the ETF classifier agreed 86.8% of the
time, so it is a gross-error detector, not the source of truth: it flags
roughly 50 names worth eyeballing rather than all 397.

Outputs:
  cache/sectors.csv          ticker, sector, etf, etf_implied, agrees
  cache/sector_review.csv    the disagreements, for manual review
"""

import os
import time
import numpy as np
import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

# yfinance sector name -> the sector ETF that tracks it
SECTOR_TO_ETF = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Healthcare": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Energy": "XLE",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}
ETF_TO_SECTOR = {v: k for k, v in SECTOR_TO_ETF.items()}

# Audit window. Uses only the most recent years so the ETF-implied
# classification reflects today's business mix, matching yfinance.
AUDIT_START = "2024-01-01"

# Names yfinance cannot resolve, filled by hand. ANSS and DFS were acquired
# during the sample (Ansys by Synopsys in 2025, Discover by Capital One in
# 2025) so their quotes no longer exist; the rest are live tickers that
# yfinance intermittently 404s under rate limiting. FI and DFS sit in
# Financial Services rather than Technology because GICS moved payment
# processors and transaction services into Financials in March 2023.
OVERRIDES = {
    "ANSS": "Technology",
    "DFS": "Financial Services",
    "BK": "Financial Services",
    "CTRA": "Energy",
    "HES": "Energy",
    "K": "Consumer Defensive",
}

# yfinance sector labels that are simply wrong or stale, found by working
# through the ETF audit's disagreement list by hand. Most trace to the March
# 2023 GICS restructure, which yfinance has not fully absorbed:
#   - Payment processors and transaction services moved to Financials.
#   - HR, payroll and data processing services moved to Industrials
#     (Professional Services).
# The rest are plain misclassifications. The ETF classifier flagged all of
# these, which is what it is for; it also flagged ~75 names where yfinance
# was right, so these were picked out by hand rather than applied wholesale.
GICS_CORRECTIONS = {
    # Tech -> Industrials (Professional Services / Transportation)
    "ADP": "Industrials",     # payroll and HR services
    "PAYX": "Industrials",    # payroll and HR services
    "BR": "Industrials",      # Broadridge, investor communications
    "UBER": "Industrials",    # GICS Ground Transportation, not software
    "FTV": "Industrials",     # Fortive, industrial instruments
    "LDOS": "Industrials",    # Leidos, government services
    "TDY": "Industrials",     # Teledyne, instruments
    # Tech -> Financials (payments)
    "FIS": "Financial Services",
    "CPAY": "Financial Services",   # Corpay, formerly FLEETCOR
    # Consumer Cyclical -> Basic Materials (Containers and Packaging)
    "IP": "Basic Materials",        # International Paper
    "PKG": "Basic Materials",       # Packaging Corp of America
    "SW": "Basic Materials",        # Smurfit Westrock
}


def fetch_sectors(tickers, cache_path):
    """One yfinance pass, cached. Re-run only fills gaps."""
    known = {}
    if os.path.exists(cache_path):
        prev = pd.read_csv(cache_path)
        known = dict(zip(prev["ticker"], prev["sector"]))

    def missing():
        return [t for t in tickers
                if t not in known or pd.isna(known.get(t))]

    todo = missing()
    print(f"  {len(known)} cached, {len(todo)} to fetch")

    for i, t in enumerate(todo, 1):
        try:
            known[t] = yf.Ticker(t).info.get("sector", np.nan)
        except Exception:
            known[t] = np.nan
        if i % 25 == 0:
            print(f"    {i}/{len(todo)}")
            time.sleep(1)

    # Most first-pass failures are rate limiting, not bad tickers. Retry the
    # stragglers slowly before falling back to the hand-coded overrides.
    retry = missing()
    if retry:
        print(f"  retrying {len(retry)} slowly: {retry}")
        for t in retry:
            time.sleep(2)
            try:
                known[t] = yf.Ticker(t).info.get("sector", np.nan)
            except Exception:
                known[t] = np.nan

    still = missing()
    for t in still:
        if t in OVERRIDES:
            known[t] = OVERRIDES[t]
    if still:
        print(f"  applied overrides for {[t for t in still if t in OVERRIDES]}")
        unresolved = [t for t in still if t not in OVERRIDES]
        if unresolved:
            print(f"  STILL UNRESOLVED (add to OVERRIDES): {unresolved}")

    return pd.Series(known, name="sector")


def etf_implied_sector(panel, tickers):
    """
    Regress each stock's market-relative return on the 11 sector-ETF
    market-relative returns; the largest coefficient names the sector.
    Multivariate, not pairwise: pairwise correlation is confounded by a
    defensiveness / rate-sensitivity factor that drags JNJ and PFE onto
    XLP and XLU, T and VZ onto XLU, and V and MA onto XLC.
    """
    ret = panel["ret"].loc[AUDIT_START:]
    etfs = [e for e in panel["etfs"] if e in ret.columns]

    S = ret[etfs]
    # Equal-weighted mean of the 11 sector ETFs, NOT the cap-weighted market
    # return. The cap-weighted market is dominated by mega-cap Technology, so
    # subtracting it leaves almost no tech signal in MSFT or GOOGL and the
    # regression fits noise: using it drops agreement from 87% to 79% and
    # misclassifies MSFT as XLP. An equal-weighted sector average is the
    # neutral benchmark this classification needs.
    mkt = S.mean(axis=1)
    Sd = S.sub(mkt, axis=0).dropna()

    out = {}
    for t in tickers:
        if t not in ret.columns:
            continue
        y = (ret[t] - mkt).reindex(Sd.index)  # noqa: E501  (same benchmark as Sd)
        m = y.notna()
        if m.sum() < 250:
            continue
        X = np.column_stack([np.ones(m.sum()), Sd[m].values])
        beta = np.linalg.lstsq(X, y[m].values, rcond=None)[0][1:]
        out[t] = etfs[int(np.argmax(beta))]
    return pd.Series(out, name="etf_implied")


def main():
    panel = pd.read_pickle(os.path.join(CACHE, "panel.pkl"))
    stocks = panel["stocks"]
    print(f"Mapping {len(stocks)} stocks")

    path = os.path.join(CACHE, "sectors.csv")
    sectors = fetch_sectors(stocks, path)

    df = pd.DataFrame({"sector": sectors}).reindex(stocks)

    applied = {t: s for t, s in GICS_CORRECTIONS.items() if t in df.index}
    for t, s in applied.items():
        df.loc[t, "sector"] = s
    print(f"  applied {len(applied)} hand GICS corrections")

    df["etf"] = df["sector"].map(SECTOR_TO_ETF)

    print("Running ETF audit...")
    df["etf_implied"] = etf_implied_sector(panel, stocks)
    df["agrees"] = df["etf"] == df["etf_implied"]

    df.index.name = "ticker"
    df.to_csv(path)

    review = df[~df["agrees"] & df["sector"].notna()].copy()
    review["etf_implied_sector"] = review["etf_implied"].map(ETF_TO_SECTOR)
    review.to_csv(os.path.join(CACHE, "sector_review.csv"))

    print("\n--- sector counts ---")
    print(df["sector"].value_counts().to_string())
    n_missing = int(df["sector"].isna().sum())
    print(f"\nunmapped: {n_missing}")
    print(f"ETF audit agreement: {df['agrees'].mean():.1%} "
          f"({len(review)} names flagged for review)")
    print(f"\nwrote {path}")
    print(f"wrote {os.path.join(CACHE, 'sector_review.csv')}")

    port = [t for t in panel["portfolio"] if t in df.index]
    print("\n--- portfolio sectors ---")
    print(df.loc[port, ["sector", "etf_implied", "agrees"]].to_string())


if __name__ == "__main__":
    main()
