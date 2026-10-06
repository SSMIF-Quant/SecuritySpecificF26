"""
Step 6: Apply the model to the SSMIF portfolio. This is the deliverable.

Answers four questions:
  1. How much risk is the fund running, and is it factor or stock-specific?
  2. What bets is it actually making, measured in standard deviations?
  3. Which positions and which factors contribute that risk?
  4. How much of it is deliberate?

VOO is handled by look-through: it decomposes into the cap-weighted universe,
which by construction has zero exposure to every style (that is what the
standardization in step 3 guarantees) and index weights on every sector. So
VOO adds market exposure while diluting every active bet.

Risk is reported on the equity sleeve. The Treasury sleeve (IEI, SHY, SGOV,
15.6% of the fund) has no place in an equity factor model. Fund-level numbers
scale the sleeve by its share of NAV, which slightly OVERSTATES fund risk
because it treats the bonds as zero-vol rather than as the diversifier they
are.

Outputs to ./cache/risk_report.pkl and prints the report.
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

FIXED_INCOME_MV = 198511.74     # IEI + SHY + SGOV
FUND_TOTAL_MV = 1269867.46


def main():
    panel = pd.read_pickle(os.path.join(CACHE, "panel.pkl"))
    exp = pd.read_pickle(os.path.join(CACHE, "exposures.pkl"))
    fr = pd.read_pickle(os.path.join(CACHE, "factor_returns.pkl"))
    cov = pd.read_pickle(os.path.join(CACHE, "covariance.pkl"))

    asof = cov["as_of"]
    factors = cov["factors"]
    Fcov = cov["factor_cov"].loc[factors, factors]
    spec_var = cov["specific_var"]

    stocks = exp["stocks"]
    styles = exp["styles"]
    style_names = fr["styles"]
    sectors = fr["sectors"]
    sector_map = exp["sector_map"]

    # --- build the holdings vector ------------------------------------------
    pos = pd.Series(panel["portfolio"], dtype=float)
    voo_mv = panel["voo_mv"]
    sleeve_mv = pos.sum() + voo_mv

    mcap = panel["mcap"].loc[asof, stocks]
    cap_w = (mcap / mcap.sum()).fillna(0.0)          # VOO look-through

    h_direct = pos.reindex(stocks).fillna(0.0) / sleeve_mv
    h_voo = cap_w * (voo_mv / sleeve_mv)
    h = h_direct + h_voo                             # equity sleeve weights
    b = cap_w                                        # benchmark = the index
    a = h - b                                        # active weights

    # --- exposure matrix at as-of date --------------------------------------
    X_sty = pd.DataFrame({k: styles[k].loc[asof] for k in style_names},
                         index=stocks)
    X_sec = pd.DataFrame(
        {s: (sector_map == s).astype(float) for s in sectors}, index=stocks)
    X = pd.concat([pd.Series(1.0, index=stocks, name="Country"),
                   X_sec, X_sty], axis=1)[factors]

    usable = X.notna().all(axis=1) & spec_var.reindex(stocks).notna()
    dropped = h[(~usable) & (h.abs() > 1e-9)]
    X = X[usable].fillna(0.0)
    for v in (h, b, a):
        v.loc[~usable] = 0.0
    h, b, a = h[usable], b[usable], a[usable]
    h, b = h / h.sum(), b / b.sum()
    a = h - b
    dv = spec_var.reindex(h.index).fillna(spec_var.median())

    # --- risk decomposition --------------------------------------------------
    def decompose(w):
        x = X.T @ w
        fac_var = float(x @ Fcov.values @ x)
        spec = float((w ** 2 * dv).sum())
        return x, fac_var, spec, np.sqrt(fac_var + spec)

    x_h, fv_h, sv_h, sig_h = decompose(h)
    x_b, fv_b, sv_b, sig_b = decompose(b)
    x_a, fv_a, sv_a, sig_a = decompose(a)

    # marginal and percent contribution to total risk
    mctr_f = (Fcov.values @ x_h) / sig_h
    pctr_factor = pd.Series(x_h * mctr_f / sig_h, index=factors)
    pctr_spec = (h ** 2 * dv) / sig_h ** 2

    # same, for active risk
    mctr_fa = (Fcov.values @ x_a) / sig_a
    pctr_factor_a = pd.Series(x_a * mctr_fa / sig_a, index=factors)
    pctr_spec_a = (a ** 2 * dv) / sig_a ** 2

    # per-stock contribution to total risk
    stock_mctr = (X.values @ (Fcov.values @ x_h) + h.values * dv.values) / sig_h
    stock_pctr = pd.Series(h.values * stock_mctr / sig_h, index=h.index)

    sleeve_share = sleeve_mv / FUND_TOTAL_MV

    # --- report --------------------------------------------------------------
    L = []
    def p(s=""):
        L.append(s)
        print(s)

    p("=" * 72)
    p(f"SSMIF EQUITY SLEEVE - BARRA RISK REPORT     as of {asof.date()}")
    p("=" * 72)
    p(f"  Equity sleeve      ${sleeve_mv:>12,.0f}   "
      f"{100*sleeve_share:5.1f}% of fund")
    p(f"    direct holdings  ${pos.sum():>12,.0f}   "
      f"{100*pos.sum()/sleeve_mv:5.1f}% of sleeve, {len(pos)} names")
    p(f"    VOO              ${voo_mv:>12,.0f}   "
      f"{100*voo_mv/sleeve_mv:5.1f}% of sleeve (look-through)")
    p(f"  Fixed income       ${FIXED_INCOME_MV:>12,.0f}   "
      f"{100*FIXED_INCOME_MV/FUND_TOTAL_MV:5.1f}% of fund, excluded")
    if len(dropped):
        p(f"  dropped (no data): {list(dropped.index)}")

    p("\n" + "-" * 72)
    p("1. TOTAL RISK")
    p("-" * 72)
    p(f"  {'':22s} {'ann. vol':>10s} {'factor':>10s} {'specific':>10s}")
    for lbl, fv, sv, sg in [("Portfolio", fv_h, sv_h, sig_h),
                            ("Benchmark (index)", fv_b, sv_b, sig_b)]:
        p(f"  {lbl:22s} {sg:>9.2%} {np.sqrt(fv):>10.2%} {np.sqrt(sv):>10.2%}")
    p(f"  {'':22s} {'':>10s} {fv_h/sig_h**2:>9.1%} {sv_h/sig_h**2:>10.1%}"
      f"   <- share of variance")
    p("")
    p(f"  ACTIVE RISK (tracking error vs index): {sig_a:.2%}")
    p(f"    from factor bets   {np.sqrt(fv_a):>7.2%}  "
      f"({fv_a/sig_a**2:.0%} of active variance)")
    p(f"    from stock picking {np.sqrt(sv_a):>7.2%}  "
      f"({sv_a/sig_a**2:.0%} of active variance)")
    p("")
    p(f"  Fund-level estimate (sleeve x {sleeve_share:.3f}): "
      f"{sig_h*sleeve_share:.2%} total, {sig_a*sleeve_share:.2%} active")

    p("\n" + "-" * 72)
    p("2. ACTIVE STYLE BETS (in standard deviations vs the index)")
    p("-" * 72)
    st = pd.DataFrame({
        "portfolio": x_h[style_names],
        "active": x_a[style_names],
        "pct_active_risk": pctr_factor_a[style_names],
    }).sort_values("active", key=abs, ascending=False)
    p(st.round(3).to_string())

    p("\n" + "-" * 72)
    p("3. ACTIVE SECTOR BETS (weight vs index)")
    p("-" * 72)
    sec = pd.DataFrame({
        "port_wt": [h[sector_map.reindex(h.index) == s].sum() for s in sectors],
        "bench_wt": [b[sector_map.reindex(b.index) == s].sum() for s in sectors],
    }, index=sectors)
    sec["active_wt"] = sec.port_wt - sec.bench_wt
    sec["pct_active_risk"] = pctr_factor_a.reindex(sectors)
    p(sec.sort_values("active_wt", key=abs, ascending=False)
        .round(4).to_string())

    p("\n" + "-" * 72)
    p("4. WHERE THE ACTIVE RISK COMES FROM")
    p("-" * 72)
    grp = pd.Series({
        "Country / market": pctr_factor_a.get("Country", 0.0),
        "Sector bets": pctr_factor_a.reindex(sectors).sum(),
        "Style bets": pctr_factor_a.reindex(style_names).sum(),
        "Stock specific": pctr_spec_a.sum(),
    })
    p(grp.map(lambda v: f"{v:>8.1%}").to_string())
    p(f"  {'total':22s} {grp.sum():>8.1%}")

    p("\n" + "-" * 72)
    p("5. TOP 12 POSITIONS BY CONTRIBUTION TO TOTAL RISK")
    p("-" * 72)
    top = pd.DataFrame({
        "weight": h,
        "active_wt": a,
        "spec_vol": np.sqrt(dv),
        "pct_total_risk": stock_pctr,
    }).loc[pos.index.intersection(h.index)]
    top = top.sort_values("pct_total_risk", ascending=False).head(12)
    p(top.round(4).to_string())

    p("\n" + "=" * 72)

    pd.to_pickle({
        "as_of": asof, "holdings": h, "benchmark": b, "active": a,
        "exposures": x_h, "active_exposures": x_a,
        "total_vol": sig_h, "active_vol": sig_a,
        "factor_var": fv_h, "specific_var": sv_h,
        "pctr_factor": pctr_factor, "pctr_factor_active": pctr_factor_a,
        "pctr_stock": stock_pctr, "sector_table": sec, "style_table": st,
        "report": "\n".join(L),
    }, os.path.join(CACHE, "risk_report.pkl"))


if __name__ == "__main__":
    main()
