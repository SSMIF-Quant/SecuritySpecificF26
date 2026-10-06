"""
Step 1: Data loader for the Barra-style risk model.

Pulls the ClickHouse equities panel, cleans it, and caches wide
(date x symbol) frames to pickle for the descriptor build in step 3.

Cleaning applied, in order:
  1. Collapse duplicate (symbol, date) rows with max() per column.
     Six symbols carry systematic duplicates: AAPL (4x), MSFT, NUE,
     QCOM, FOX, FOXA (2x). Verified they never disagree on PX_LAST.
  2. Zero -> NaN. Zero means missing in every column of this table.
  3. Reindex onto the XLK trading calendar. The raw table carries
     weekend rows (22 Saturdays, 21 Sundays in 2024 alone).
  4. Trim the ragged tail where prices exist but mcap / TRI do not.
  5. Split the 11 SPDR sector ETFs out of the stock universe.
  6. Forward-fill quarterly fundamentals, then lag LAG_DAYS trading
     days. Fundamentals are stamped at fiscal period end, not filing
     date, so raw use carries 30-45 days of look-ahead per quarter.

Outputs to ./cache/panel.pkl
"""

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import clickhouse_connect

# --- config -----------------------------------------------------------------

START = "2009-01-01"      # 2012 model start needs 504d momentum + 252d beta lookback
END = "2026-05-01"
LAG_DAYS = 63             # ~3 months, kills the period-end look-ahead
CALENDAR_SYMBOL = "XLK"   # its date index is the clean NYSE calendar
MIN_TRI_COVERAGE = 0.90   # trim tail dates below this cross-sectional coverage
MIN_PRICE_OBS = 250       # drop symbols with fewer price days than this

SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI",
               "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]

PRICE_FIELDS = [
    "PX_LAST", "CUR_MKT_CAP", "PX_VOLUME", "TOT_RETURN_INDEX_GROSS_DVDS",
    "AVERAGE_BID_ASK_SPREAD", "BEST_EPS", "BEST_DIV_YLD",
]

FUNDAMENTAL_FIELDS = [
    "IS_DIV_PER_SHR", "SALES_REV_TURN", "IS_COG_AND_SERVICES_SOLD",
    "IS_OPER_INC", "EBITA", "NET_INCOME", "BS_TOT_ASSET",
    "BS_CASH_NEAR_CASH_ITEM", "BS_CUR_ASSET_REPORT", "BS_NET_FIX_ASSET",
    "BS_INVENTORIES", "BS_DISCLOSED_INTANGIBLES", "BS_GOODWILL",
    "BS_CUR_LIAB", "SHORT_AND_LONG_TERM_DEBT", "TOTAL_EQUITY",
    "CF_CASH_FROM_OPER", "CAPITAL_EXPEND", "CF_DEPR_AMORT",
]

ALL_FIELDS = PRICE_FIELDS + FUNDAMENTAL_FIELDS

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

# The SSMIF equity sleeve. Jacobs (J) is not in the database and has been
# dropped. GOOG is mapped to GOOGL: class C has only 3,001 days of total-return
# history vs GOOGL's 5,072, since it did not exist before April 2014.
PORTFOLIO = {
    "AMAT": 113055.67, "GS": 87148.80, "SLB": 56592.80, "JPM": 55466.73,
    "COST": 49923.50, "META": 46609.64, "NFLX": 46457.51, "PANW": 41735.64,
    "TMO": 41177.13, "GOOGL": 40096.44, "BA": 34603.17, "CVX": 31299.30,
    "MSFT": 27815.25, "DIS": 25090.01, "PG": 23806.24, "EMR": 21565.50,
    "ADP": 20578.11, "MKC": 19335.60, "NKE": 12018.84, "QCOM": 10757.04,
    "AMGN": 8775.50,
}
VOO_MV = 226729.80
FUND_TOTAL_MV = 1269867.46


def connect():
    load_dotenv(os.path.join(os.path.dirname(HERE), ".env"))
    return clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST"),
        port=int(os.getenv("CLICKHOUSE_PORT")),
        username=os.getenv("CLICKHOUSE_USER"),
        password=os.getenv("CLICKHOUSE_PASSWORD"),
        secure=False,
        connect_timeout=30,
    )


def fetch_panel(client):
    """One query, duplicates collapsed server side."""
    agg = ", ".join(f"max({f}) AS {f}" for f in ALL_FIELDS)
    q = f"""
        SELECT date, symbol, {agg}
        FROM default.equities
        WHERE date >= '{START}' AND date <= '{END}'
        GROUP BY date, symbol
    """
    df = client.query_df(q)
    df["symbol"] = df["symbol"].str.replace(" US Equity", "", regex=False)
    return df


def to_wide(df, field, calendar):
    """Long -> wide (date x symbol), zeros to NaN, on the trading calendar."""
    w = df.pivot(index="date", columns="symbol", values=field)
    w = w.replace(0.0, np.nan)
    return w.reindex(calendar).astype("float32")


def fetch_riskfree(client, calendar):
    """3-month T-bill, annual percent -> daily decimal."""
    q = f"""
        SELECT date, DGS3MO FROM default.fred
        WHERE DGS3MO IS NOT NULL AND date >= '{START}' AND date <= '{END}'
        ORDER BY date
    """
    rf = client.query_df(q).set_index("date")["DGS3MO"].astype(float)
    rf = rf.reindex(calendar).ffill()
    return (1.0 + rf / 100.0) ** (1.0 / 252.0) - 1.0


def main():
    os.makedirs(CACHE, exist_ok=True)
    client = connect()

    print("Fetching panel from ClickHouse...")
    df = fetch_panel(client)
    print(f"  {len(df):,} collapsed symbol-date rows, "
          f"{df['symbol'].nunique()} symbols")

    # --- trading calendar from the sector ETF ---
    cal = pd.DatetimeIndex(
        sorted(df.loc[(df["symbol"] == CALENDAR_SYMBOL)
                      & df["PX_LAST"].notna()
                      & (df["PX_LAST"] != 0), "date"].unique())
    )
    print(f"  calendar: {len(cal)} trading days, {cal[0].date()} to {cal[-1].date()}")

    wide = {f: to_wide(df, f, cal) for f in ALL_FIELDS}

    # --- split ETFs from stocks ---
    etfs = [s for s in SECTOR_ETFS if s in wide["PX_LAST"].columns]
    stocks = [s for s in wide["PX_LAST"].columns if s not in SECTOR_ETFS]

    # Drop phantom symbols: FI and MMC carry 459 and 353 rows in ClickHouse
    # but every price field is zero across the full 2006-2026 span. They hold
    # fundamentals with no market data, so they cannot enter the regression.
    px_obs = wide["PX_LAST"][stocks].notna().sum()
    phantom = sorted(px_obs[px_obs < MIN_PRICE_OBS].index)
    if phantom:
        stocks = [s for s in stocks if s not in phantom]
        print(f"  dropped {len(phantom)} symbols with no price data: {phantom}")
    print(f"  {len(stocks)} stocks, {len(etfs)} sector ETFs")

    # --- trim the ragged tail ---
    tri = wide["TOT_RETURN_INDEX_GROSS_DVDS"][stocks]
    cov = tri.notna().mean(axis=1)
    last_good = cov[cov >= MIN_TRI_COVERAGE].index.max()
    cal = cal[cal <= last_good]
    wide = {f: w.loc[cal] for f, w in wide.items()}
    print(f"  trimmed tail to {last_good.date()} "
          f"({len(cal)} days, TRI coverage >= {MIN_TRI_COVERAGE:.0%})")

    # --- returns ---
    tri = wide["TOT_RETURN_INDEX_GROSS_DVDS"]
    ret = tri.pct_change(fill_method=None)
    ret = ret.where(tri.notna() & tri.shift(1).notna())
    ret = ret.clip(-0.60, 1.00)   # guard against TRI restatement artifacts

    rf = fetch_riskfree(client, cal)
    excess = ret.sub(rf, axis=0)

    # --- cap-weighted market return over the stock universe ---
    mcap = wide["CUR_MKT_CAP"]
    w_cap = mcap[stocks].shift(1)
    w_cap = w_cap.div(w_cap.sum(axis=1), axis=0)
    mkt = (w_cap * ret[stocks]).sum(axis=1, min_count=1)
    mkt_excess = mkt - rf

    # --- derived shares outstanding, for turnover ---
    shares = (wide["CUR_MKT_CAP"] / wide["PX_LAST"]).astype("float32")

    # --- fundamentals: forward-fill, then lag ---
    # Balance-sheet items are levels and can be used as-is. Income and cash
    # flow items are quarterly flows, so step 3 needs the raw stamps to build
    # trailing-twelve-month sums before filling; both versions are kept.
    fundamentals = {}
    for f in FUNDAMENTAL_FIELDS:
        fundamentals[f] = wide[f].ffill().shift(LAG_DAYS).astype("float32")
    fundamentals_raw = {f: wide[f] for f in FUNDAMENTAL_FIELDS}
    print(f"  fundamentals forward-filled and lagged {LAG_DAYS} trading days")

    panel = {
        "calendar": cal,
        "stocks": stocks,
        "etfs": etfs,
        "px": wide["PX_LAST"],
        "mcap": wide["CUR_MKT_CAP"],
        "volume": wide["PX_VOLUME"],
        "shares": shares,
        "spread": wide["AVERAGE_BID_ASK_SPREAD"],
        "best_eps": wide["BEST_EPS"],
        "best_div_yld": wide["BEST_DIV_YLD"],
        "ret": ret,
        "excess": excess,
        "rf": rf,
        "mkt": mkt,
        "mkt_excess": mkt_excess,
        "fundamentals": fundamentals,
        "fundamentals_raw": fundamentals_raw,
        "portfolio": PORTFOLIO,
        "voo_mv": VOO_MV,
        "fund_total_mv": FUND_TOTAL_MV,
        "meta": {"start": START, "end": str(last_good.date()),
                 "lag_days": LAG_DAYS},
    }

    out = os.path.join(CACHE, "panel.pkl")
    pd.to_pickle(panel, out)

    # --- report ---
    print("\n--- sanity checks ---")
    ann = mkt_excess.loc["2012":].std() * np.sqrt(252)
    print(f"market excess return vol (2012+, ann.): {ann:6.2%}   [expect 12-18%]")
    print(f"mean daily cross-section size:          "
          f"{ret[stocks].notna().sum(axis=1).loc['2012':].mean():6.0f} stocks")
    print(f"risk-free, last value (ann.):           "
          f"{((1 + rf.iloc[-1]) ** 252 - 1):6.2%}")

    missing = [t for t in PORTFOLIO if t not in stocks]
    print(f"\nportfolio: {len(PORTFOLIO)} names, missing from universe: "
          f"{missing if missing else 'none'}")
    hist = {t: int(ret[t].loc["2012":].notna().sum()) for t in PORTFOLIO
            if t in stocks}
    thin = {t: n for t, n in hist.items() if n < 2000}
    print(f"names with < 2000 return days since 2012: "
          f"{thin if thin else 'none'}")

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
