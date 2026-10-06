# Barra-Style Risk Model

A fundamental factor risk model built on the SSMIF ClickHouse equities panel,
applied to the fund's equity sleeve. Run `01` through `07` in order; each step
caches its output for the next.

---

## How the pipeline works

**`01_data.py` — the panel.** Queries ClickHouse with duplicates collapsed
server-side via `GROUP BY symbol, date` and `max()` per column. Converts zeros
to NaN (zero means missing everywhere in this table), reindexes onto XLK's date
series as the NYSE calendar, drops two phantom symbols with no price data, and
trims the tail to the last date with >= 90% return coverage. Computes returns
from the total-return index, the risk-free rate from FRED's `DGS3MO`, and the
cap-weighted market return. Then forward-fills quarterly fundamentals and lags
them 63 trading days. Keeps a raw unfilled copy too, because trailing-twelve-month
sums have to be built from the quarterly stamps — summing the forward-filled
panel would count each quarter dozens of times.
**Out:** 395 stocks x 4,322 days.

**`02_sectors.py` — classification.** yfinance for the hard map, with a slow
retry pass and a hand-coded override table. Audited by regressing each stock on
the 11 sector ETFs; the audit is ~80% accurate so it flags rather than decides.
**Out:** 11 sector dummies.

**`03_descriptors.py` — the exposure matrix X.** Builds every descriptor, then
on each date: winsorize +/-3 sigma -> standardize to cap-weighted mean 0,
equal-weighted sd 1 -> fill gaps by sector mean -> combine with fixed weights ->
re-standardize -> orthogonalize four factors.
**Out:** X, 395 x 24.

> The standardization is load-bearing. Cap-weighted mean zero means the index
> has zero exposure to every style by construction, which is why VOO decomposes
> for free and why any non-zero style number you see is genuinely active.

**`04_regression.py` — factor returns f and specific returns u.** One WLS
cross-section per day, weights proportional to sqrt(market cap), exposures taken
as of the prior close. The country column equals the sum of the sector dummies,
so the design is exactly singular; Barra's constraint (cap-weighted sector
returns sum to zero) is imposed as a change of basis — drop one sector column,
rewrite the rest as `Xs - (ws/wS) * XS`, recover the dropped one afterward.
**Out:** 3,556 days of f, plus u as residuals. Country factor correlates 0.9994
with the market.

**`05_covariance.py` — F and Delta.** EWMA covariance with different half-lives
for volatility (84d) and correlation (504d), since correlations are noisier and
need longer memory, recombined as `D * C * D`. Newey-West with 2 lags for
autocorrelation, eigenvalue clipping for positive definiteness. Specific
variances are EWMA on residuals, Bayesian-shrunk toward the size-decile mean.
**Out:** F (24x24) and Delta (395 variances), annualized.

**`06_portfolio.py` — the diagnosis.** `V = XFX' + Delta` is now complete, so
risk is arithmetic. Holdings vector with VOO looked through to cap weights,
exposures `x = X'h`, factor risk `x'Fx`, specific risk `h'(Delta)h`, same again
on active weights.
**Out:** 5.67% tracking error, 73% specific.

**`07_construction.py` — the decision.** Differentiates risk with respect to
each weight: `MCAR = (XFx + a*Delta) / TE`. Multiply by active weight for each
name's percent of active risk, then cost four sizing schemes.
**Out:** AMAT at 2.63x its weight, and the 3%-turnover fix.

---

## How this differs from a real Barra model

### Differences that shrink the model

Known, documented, defensible:

|  | USE4 | Ours |
|---|---|---|
| Estimation universe | ~3,000 US stocks | 395 |
| Industry factors | ~60 GICS industries | 11 GICS sectors |
| Total factors | ~70 | 24 |
| Horizons | short and long, separately estimated | short only |

With 395 names, 60 industries would leave ~6 stocks per bucket and the dummies
would be noise. Eleven sectors at ~35 names each is the right call for this
universe, but it means you cannot distinguish semis from software — AMAT and
PANW sit in the same bucket.

### Three factors are structurally modified

Because the data can't support the real descriptors:

- **Earnings Yield** loses its 0.68-weight forecast E/P since `BEST_EPS` is
  empty before 2023, so weights are renormalized onto trailing.
- **Growth** loses both analyst-forecast descriptors (0.29 of USE4's weight)
  and is historical-only.
- **Leverage** has no preferred equity and no long/short debt split.

All three are the standard homemade compromises, but they mean these factors
are not comparable to published USE4 output.

### Four methodology pieces are not implemented

1. **Eigenfactor risk adjustment** — sample covariance matrices systematically
   understate the risk of optimized portfolios.
2. **Volatility regime adjustment** — scales the forecast by how recent
   realized risk compared to prediction.
3. **Structural specific-risk model** for stocks with short histories (we
   shrink, but don't predict from characteristics).
4. **Estimation vs coverage universe distinction** — Barra estimates on a clean
   subset and applies to a broader one.

The first two would both **raise** the risk forecast. So 5.67% tracking error
is a floor, not a point estimate.
