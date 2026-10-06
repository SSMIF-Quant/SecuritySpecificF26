# Improving the Data Behind the Model

What to pull from Bloomberg into ClickHouse, in priority order, and what each
pull actually buys. Survivorship bias is accepted as unfixable and is not
addressed here.

---

## Universe: 395 -> 500

Going 395 -> 500 names of the same kind doesn't fix anything structural — it
takes the regression from 378 to ~490 observations, which is a real but modest
improvement in standard errors. Worth doing because it's a cheap pull and puts
names you might buy into the model.

---

## The biggest win is GICS industry groups

Pull `GICS_INDUSTRY_GROUP_NAME` as a quarterly time series.

Right now you run 11 sector dummies, which means **AMAT and PANW sit in the
same bucket** — semiconductor equipment and security software treated as one
risk source, and those are your #1 and #8 positions.

With 500 names, 24 industry groups gives ~21 per bucket and the regression runs
37 parameters on ~490 observations, which is comfortable. Don't go to the
69-industry level; that's the trade Barra makes with 3,000 stocks and it isn't
available to you.

Pulling it as a **time series** rather than a snapshot also kills the
point-in-time problem — no more applying today's sectors retroactively.

---

## Second: announcement dates

`EARN_ANN_DT_TIME_HIST_WITH_EPS` lets you replace the blanket 63-trading-day
lag. That lag is wrong in both directions — too short for slow filers, and it
needlessly staleness-penalizes every fast reporter by a month.

---

## The three broken factors

The specific gaps:

| Factor | Needs |
|---|---|
| **Earnings Yield** | `BEST_EPS` with `BEST_FPERIOD_OVERRIDE = "BF"` — that's 0.68 of the factor, two-thirds missing |
| **Growth** | `BEST_LTG_EPS` plus `BEST_EEPS_CUR_YR` / `BEST_EEPS_NXT_YR` |
| **Leverage** | `BS_PFD_EQY`, `BS_LT_BORROW`, `BS_ST_BORROW` |

Also switch EGRO from `NET_INCOME` to `IS_DILUTED_EPS` — net income conflates
earnings growth with buybacks, and Barra uses EPS.

---

## Three things worth flagging before you pull

1. **Test history depth first.** `BEST_EPS` already burned you — it exists but
   is empty before 2023. Run a single-ticker `BDH` on `BEST_LTG_EPS` and
   `EQY_FLOAT` back to 2006 before spending the data budget. A field that only
   goes back 5 years can't support a factor in a model estimated from 2012.

2. **Pull fundamentals quarterly (`Per=cq`), not daily.** Your current table
   stores each quarterly value repeated across ~63 daily rows — 1.67M rows
   carrying maybe 40k actual fundamental observations.

3. **Split the table.** Daily prices, quarterly fundamentals keyed by **both**
   period end and announcement date, quarterly classification. Use
   `ReplacingMergeTree` and the duplicate-row problem disappears at source —
   AAPL currently has 4 rows per date.
