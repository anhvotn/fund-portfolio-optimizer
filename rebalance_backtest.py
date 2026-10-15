"""
rebalance_backtest.py - compare rebalancing rules for a target portfolio.

Usage:
    python rebalance_backtest.py                                   # bundled examples
    python rebalance_backtest.py prices.csv weights.csv
        prices : wide CSV (date + one column per fund, CHF, dividends reinvested)
        weights: CSV with columns fund,weight (target weights, e.g. HRP from
                 portfolio_optimize.py; normalised to sum to 1)

Rules tested: buy & hold, calendar (monthly / quarterly / semi-annual / annual),
absolute bands (±3pp, ±5pp around target), relative band (±25% of target),
and "quarterly check + ±5pp band". Costs per rebalance =
    sum over funds traded of max(MIN_FEE_CHF, PCT_COST * trade amount)
on a portfolio of PORTFOLIO_CHF (fixed fees matter for small portfolios).

Educational analysis, not investment advice. Backtests on the same period the
weights were fitted on are optimistic; differences between rules are usually small.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_PRICES  = os.path.join(HERE, "_chf_prices.example.csv")
EXAMPLE_WEIGHTS = os.path.join(HERE, "_target_weights.example.csv")

PORTFOLIO_CHF = 20_000     # starting value used to apply fixed minimum fees
PCT_COST      = 0.0025     # 0.25% per traded amount (commission + spread + FX)
MIN_FEE_CHF   = 10.0       # minimum fee per trade (set to your bank's tariff)
MIN_TRADE_CHF = 1.0        # ignore trades smaller than this

# Calendar steps are in weeks and converted to the data frequency at run time.
RULES = {
    "Buy & hold":        ("none", None),
    "Monthly":           ("cal", 1 / 12),
    "Quarterly":         ("cal", 1 / 4),
    "Semi-annual":       ("cal", 1 / 2),
    "Annual":            ("cal", 1.0),
    "Band ±3pp":         ("abs", 0.03),
    "Band ±5pp":         ("abs", 0.05),
    "Band ±25% rel":     ("rel", 0.25),
    "Qtr check + ±5pp":  ("calband", (1 / 4, 0.05)),
}


def periods_per_year(idx):
    gap = pd.Series(idx).diff().dt.days.median()
    return 252 if gap <= 3 else (52 if gap <= 10 else 12)


def run(prices, w0, rule, freq):
    kind, p = rule
    step = None
    if kind == "cal":
        step = max(1, round(p * freq))
    elif kind == "calband":
        step = max(1, round(p[0] * freq))
    rel = prices.values[1:] / prices.values[:-1]
    h = w0 * PORTFOLIO_CHF
    vals, costs, turnover, n = [h.sum()], 0.0, 0.0, 0
    for t, g in enumerate(rel, start=1):
        h = h * g
        V = h.sum()
        w = h / V
        if kind == "cal":
            reb = t % step == 0
        elif kind == "abs":
            reb = np.any(np.abs(w - w0) > p)
        elif kind == "rel":
            reb = np.any((w0 > 0) & (np.abs(w / np.where(w0 > 0, w0, 1) - 1) > p))
        elif kind == "calband":
            reb = t % step == 0 and np.any(np.abs(w - w0) > p[1])
        else:
            reb = False
        if reb:
            trades = np.abs(w - w0) * V
            traded = trades[trades > MIN_TRADE_CHF]
            c = np.maximum(MIN_FEE_CHF, PCT_COST * traded).sum()
            costs += c
            turnover += trades.sum() / 2 / V
            n += 1
            V -= c
            h = w0 * V
        vals.append(h.sum())
    vals = np.array(vals)
    r = vals[1:] / vals[:-1] - 1
    yrs = len(r) / freq
    cagr = (vals[-1] / vals[0]) ** (1 / yrs) - 1
    vol = r.std(ddof=1) * np.sqrt(freq)
    dd = (vals / np.maximum.accumulate(vals) - 1).min()
    return {"CAGR %": cagr * 100, "Vol %": vol * 100, "Sharpe": cagr / vol,
            "Max DD %": dd * 100, "Rebalances": n,
            "Turnover %/yr": turnover / yrs * 100,
            "Costs CHF/yr": costs / yrs}


def main():
    if len(sys.argv) >= 3:
        prices_path, weights_path = sys.argv[1], sys.argv[2]
    else:
        print("[note] no files given - using the bundled synthetic examples.\n")
        prices_path, weights_path = EXAMPLE_PRICES, EXAMPLE_WEIGHTS

    prices = pd.read_csv(prices_path, parse_dates=["date"]).set_index("date").sort_index().dropna()
    wts = pd.read_csv(weights_path).set_index("fund")["weight"]
    missing = sorted(set(wts.index) - set(prices.columns))
    if missing:
        print(f"[warning] weights for funds not in the price file are ignored: {missing}\n")
    wts = wts.reindex(prices.columns).fillna(0)
    if wts.sum() <= 0:
        sys.exit("No target weight matches a price column - check the fund names.")
    w0 = (wts / wts.sum()).values
    freq = periods_per_year(prices.index)

    res = pd.DataFrame({k: run(prices, w0, r, freq) for k, r in RULES.items()}).T
    pd.set_option("display.width", 160)
    print(f"Portfolio CHF {PORTFOLIO_CHF:,.0f}, cost {PCT_COST:.2%} (min CHF {MIN_FEE_CHF:.0f}) per trade, "
          f"{len(prices)} periods ({freq}/yr)\n")
    print(res.round(2).to_string())
    best = res.sort_values(["Sharpe", "CAGR %"], ascending=False).index[0]
    print(f"\nBest risk-adjusted rule in this sample: {best}")
    drift = pd.Series(prices.iloc[-1].values / prices.iloc[0].values * w0, index=prices.columns)
    drift = drift / drift.sum() - w0
    print("\nDrift from target after buy & hold (percentage points):")
    print((drift * 100).round(1).to_string())


if __name__ == "__main__":
    main()
