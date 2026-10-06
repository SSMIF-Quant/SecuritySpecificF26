"""
Step 7: Risk budgeting and position sizing.

The fund picks names bottom-up and sizes them by conviction. This turns the
risk model into the sizing framework: same names, weights set so that each
position's contribution to risk is deliberate rather than incidental.

The central number is marginal contribution to active risk (MCAR):

    MCAR_n = d(tracking error) / d(active weight in n)

MCAR says what one more dollar of a name does to total portfolio risk, which
is not the same as how risky the name is on its own. A volatile name that
diversifies the rest can have a lower MCAR than a quiet name that doubles up
on what you already own.

The test of a well-constructed portfolio is that MCAR is proportional to
expected alpha. Where a name's risk share far exceeds its conviction share,
it is being paid for in risk without being paid for in return.

Four sizing schemes are costed, all keeping the same 21 names:
  A  Current
  B  Cap any single name's share of active risk, redistribute pro rata
  C  Equal risk contribution across names
  D  Minimum tracking error within a band around current weights

Scheme D is bounded on purpose. Free optimization on estimated covariance
concentrates in whatever the model happens to think is low risk, which is
also where the model is most likely wrong. Bands keep the fund's actual
views in the portfolio.

Outputs to ./cache/construction.pkl
"""

import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")

RISK_CAP = 0.15         # scheme B: max share of active risk for one name
BAND = 0.50             # scheme D: weights may move +/- 50% of current
TE_TARGETS = [0.03, 0.04, 0.05]   # for the VOO dial


class RiskModel:
    """Active risk for a set of direct weights plus a VOO sleeve."""

    def __init__(self, X, Fcov, dv, cap_w, names):
        self.X = X.values
        self.F = Fcov.values
        self.dv = dv.values
        self.cap = cap_w.values
        self.names = names
        self.idx = [list(X.index).index(n) for n in names]

    def holdings(self, w_direct, w_voo):
        h = self.cap * w_voo
        h = h.copy()
        h[self.idx] += w_direct
        return h

    def active(self, w_direct, w_voo):
        return self.holdings(w_direct, w_voo) - self.cap

    def variance(self, a):
        x = self.X.T @ a
        return float(x @ self.F @ x + (a ** 2 * self.dv).sum())

    def te(self, w_direct, w_voo):
        return np.sqrt(max(self.variance(self.active(w_direct, w_voo)), 0.0))

    def mcar(self, w_direct, w_voo):
        """Gradient of tracking error wrt each name's weight."""
        a = self.active(w_direct, w_voo)
        x = self.X.T @ a
        grad = self.X @ (self.F @ x) + a * self.dv
        te = np.sqrt(max(self.variance(a), 0.0))
        return pd.Series(grad[self.idx] / te, index=self.names), te

    def risk_shares(self, w_direct, w_voo):
        """Each name's percent contribution to active risk. Sums to <= 1."""
        a = self.active(w_direct, w_voo)
        m, te = self.mcar(w_direct, w_voo)
        return pd.Series(a[self.idx] * m.values / te, index=self.names), te


def scheme_cap(rm, w0, voo0, cap=RISK_CAP, iters=60):
    """Trim names above the risk cap, push the freed weight to the rest."""
    w = w0.copy()
    for _ in range(iters):
        shares, _ = rm.risk_shares(w, voo0)
        over = shares[shares > cap]
        if over.empty:
            break
        excess = 0.0
        for n in over.index:
            scale = cap / shares[n]
            cut = w[n] * (1.0 - scale ** 0.5)
            w[n] -= cut
            excess += cut
        under = shares[shares <= cap].index
        if len(under) == 0:
            break
        room = w[under]
        w[under] += excess * room / room.sum()
    return w, voo0


def scheme_equal_risk(rm, w0, voo0):
    """Equal contribution to active risk across the held names."""
    total = w0.sum()
    target = 1.0 / len(w0)

    def obj(z):
        w = pd.Series(np.exp(z), index=w0.index)
        w = w / w.sum() * total
        shares, _ = rm.risk_shares(w, voo0)
        s = shares / shares.sum()
        return float(((s - target) ** 2).sum()) * 1e4

    z0 = np.log(w0.values / w0.sum() * total)
    res = minimize(obj, z0, method="Nelder-Mead",
                   options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-10})
    w = pd.Series(np.exp(res.x), index=w0.index)
    return w / w.sum() * total, voo0


def scheme_min_te(rm, w0, voo0, band=BAND):
    """Minimum tracking error with weights bounded around current."""
    total = w0.sum()
    lo = w0.values * (1.0 - band)
    hi = w0.values * (1.0 + band)

    def obj(w):
        return rm.variance(rm.active(pd.Series(w, index=w0.index), voo0)) * 1e4

    cons = [{"type": "eq", "fun": lambda w: w.sum() - total}]
    res = minimize(obj, w0.values, method="SLSQP",
                   bounds=list(zip(lo, hi)), constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-12})
    return pd.Series(res.x, index=w0.index), voo0


def voo_for_te(rm, w0, voo0, target):
    """How much VOO ballast reaches a given tracking error, names fixed."""
    total = w0.sum()

    def te_at(v):
        w = w0 / total * (1.0 - v)
        return rm.te(w, v)

    lo, hi = 0.0, 0.95
    if te_at(lo) < target:
        return np.nan
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if te_at(mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def summarize(rm, w, voo, label, w0, voo0):
    shares, te = rm.risk_shares(w, voo)
    turnover = float((w - w0).abs().sum() + abs(voo - voo0)) / 2.0
    return {
        "scheme": label,
        "tracking_error": te,
        "max_name_risk": shares.max(),
        "max_name": shares.idxmax(),
        "top3_risk": shares.nlargest(3).sum(),
        "turnover": turnover,
        "weights": w,
        "voo": voo,
        "risk_shares": shares,
    }


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

    X_sty = pd.DataFrame({k: styles[k].loc[asof] for k in style_names},
                         index=stocks)
    X_sec = pd.DataFrame(
        {s: (sector_map == s).astype(float) for s in sectors}, index=stocks)
    X = pd.concat([pd.Series(1.0, index=stocks, name="Country"),
                   X_sec, X_sty], axis=1)[factors]

    mcap = panel["mcap"].loc[asof, stocks]
    cap_w = (mcap / mcap.sum()).fillna(0.0)

    usable = X.notna().all(axis=1) & spec_var.reindex(stocks).notna() \
        & (cap_w > 0)
    X, cap_w = X[usable].fillna(0.0), cap_w[usable]
    cap_w = cap_w / cap_w.sum()
    dv = spec_var.reindex(X.index).fillna(spec_var.median())

    pos = pd.Series(panel["portfolio"], dtype=float)
    pos = pos[pos.index.isin(X.index)]
    sleeve = pos.sum() + panel["voo_mv"]
    w0 = pos / sleeve
    voo0 = panel["voo_mv"] / sleeve
    names = list(w0.index)

    rm = RiskModel(X, Fcov, dv, cap_w, names)

    print(f"Risk budgeting as of {asof.date()}   "
          f"{len(names)} names + {voo0:.1%} VOO\n")

    # --- the risk budget --------------------------------------------------
    m, te = rm.mcar(w0, voo0)
    shares, _ = rm.risk_shares(w0, voo0)
    a = pd.Series(rm.active(w0, voo0), index=X.index)[names]

    budget = pd.DataFrame({
        "weight": w0,
        "active_wt": a,
        "spec_vol": np.sqrt(dv.reindex(names)),
        "MCAR": m,
        "pct_active_risk": shares,
        "risk_per_wt": shares / w0,
    }).sort_values("pct_active_risk", ascending=False)

    print("=" * 78)
    print("RISK BUDGET  (pct_active_risk is the number to manage)")
    print("=" * 78)
    print(budget.round(4).to_string())
    print(f"\n  tracking error {te:.2%}   "
          f"names sum to {shares.sum():.1%} of active risk")
    print("  risk_per_wt > 1 means the name uses more risk than its weight "
          "suggests")

    # --- scenarios ---------------------------------------------------------
    results = [summarize(rm, w0, voo0, "A current", w0, voo0)]

    wB, vB = scheme_cap(rm, w0, voo0)
    results.append(summarize(rm, wB, vB, f"B cap {RISK_CAP:.0%}", w0, voo0))

    wC, vC = scheme_equal_risk(rm, w0, voo0)
    results.append(summarize(rm, wC, vC, "C equal risk", w0, voo0))

    wD, vD = scheme_min_te(rm, w0, voo0)
    results.append(summarize(rm, wD, vD, f"D min TE +/-{BAND:.0%}", w0, voo0))

    print("\n" + "=" * 78)
    print("SIZING SCHEMES  (same 21 names, weights only)")
    print("=" * 78)
    tab = pd.DataFrame([{k: r[k] for k in
                         ["scheme", "tracking_error", "max_name_risk",
                          "max_name", "top3_risk", "turnover"]}
                        for r in results]).set_index("scheme")
    print(tab.round(4).to_string())

    print("\n" + "-" * 78)
    print("WEIGHT CHANGES BY SCHEME (percentage points of the equity sleeve)")
    print("-" * 78)
    chg = pd.DataFrame({r["scheme"]: r["weights"] for r in results})
    chg = (chg.sub(chg["A current"], axis=0) * 100).drop(columns="A current")
    chg["current_wt_%"] = w0 * 100
    print(chg.sort_values("current_wt_%", ascending=False).round(2).to_string())

    # --- VOO as the risk dial ---------------------------------------------
    print("\n" + "-" * 78)
    print("VOO BALLAST: tracking error without changing relative name bets")
    print("-" * 78)
    for t in TE_TARGETS:
        v = voo_for_te(rm, w0, voo0, t)
        if np.isnan(v):
            print(f"  TE {t:.1%}: not reachable by VOO alone")
        else:
            print(f"  TE {t:.1%}  requires VOO at {v:.1%} of sleeve "
                  f"(now {voo0:.1%})")

    pd.to_pickle({"budget": budget, "schemes": results, "table": tab,
                  "as_of": asof}, os.path.join(CACHE, "construction.pkl"))
    print(f"\nwrote {os.path.join(CACHE, 'construction.pkl')}")


if __name__ == "__main__":
    main()
