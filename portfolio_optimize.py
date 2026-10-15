"""
portfolio_optimize.py - compare allocation methods on a set of funds (CHF base).

Methods
-------
  * Max-Sharpe   - mean-variance tangency portfolio (Ledoit-Wolf shrinkage
                   covariance), long-only, capped per fund.
  * Min-Variance - ignores expected returns, minimises portfolio volatility.
  * HRP          - Hierarchical Risk Parity (Lopez de Prado): clusters funds by
                   correlation, then allocates. Robust to near-duplicate funds.
  * Equal weight - benchmark.

Caveats
-------
1. "Expected returns" are trailing historical means, not a forecast. Max-Sharpe
   tilts into whatever just ran - treat it as illustrative.
2. Min-Variance and HRP are the more robust choices.
3. Validate out of sample (walk-forward) before acting on any weights.
4. Exploratory analysis, not investment advice.

Usage
-----
    python portfolio_optimize.py                  # _chf_prices.csv, else the example
    python portfolio_optimize.py my_prices.csv    # any wide CSV: date + one CHF column per fund

Daily, weekly or monthly prices are detected automatically.
"""

import os
import sys

import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models
from pypfopt.hierarchical_portfolio import HRPOpt

HERE        = os.path.dirname(os.path.abspath(__file__))
PRICES_CSV  = "_chf_prices.csv"                               # your data (git-ignored)
EXAMPLE_CSV = os.path.join(HERE, "_chf_prices.example.csv")   # synthetic, shipped
WEIGHT_CAP  = 0.20                                            # max weight per fund (MVO)
RISK_FREE   = 0.0                                             # CHF cash ~0


def load_prices(path):
    prices = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    return prices.dropna(how="any")


def periods_per_year(index):
    """Daily -> 252, weekly -> 52, monthly -> 12 (from the median date gap)."""
    gap = index.to_series().diff().dt.days.median()
    return 252 if gap <= 3 else (52 if gap <= 10 else 12)


def show(name, weights, perf):
    print(f"== {name} ==")
    for k, v in sorted(weights.items(), key=lambda x: -x[1]):
        if round(v, 4) != 0:
            print(f"   {k:<28} {v * 100:6.1f}%")
    print(f"   -> return {perf[0] * 100:5.1f}%  vol {perf[1] * 100:5.1f}%  Sharpe {perf[2]:.2f}\n")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else PRICES_CSV
    if not os.path.exists(path):
        print(f"[note] '{path}' not found - running on synthetic example data "
              f"('{os.path.basename(EXAMPLE_CSV)}'). Results are meaningless.\n")
        path = EXAMPLE_CSV

    prices = load_prices(path)
    n = prices.shape[1]
    if n < 2 or len(prices) < 30:
        sys.exit(f"Need at least 2 funds and 30 price dates; got {n} funds x {len(prices)} dates.")

    freq = periods_per_year(prices.index)
    cap = max(WEIGHT_CAP, 1.0 / n)          # a cap below 1/n cannot sum to 100%
    print(f"Loaded {n} funds x {len(prices)} dates "
          f"({prices.index.min().date()} -> {prices.index.max().date()}), "
          f"{freq} periods/year, cap {cap:.0%} per fund\n")

    returns = prices.pct_change().dropna()
    mu = expected_returns.mean_historical_return(prices, frequency=freq)
    S = risk_models.CovarianceShrinkage(prices, frequency=freq).ledoit_wolf()

    # Max-Sharpe needs at least one fund above the risk-free rate
    if (mu > RISK_FREE).any():
        ef = EfficientFrontier(mu, S, weight_bounds=(0, cap))
        ef.max_sharpe(risk_free_rate=RISK_FREE)
        show(f"Max-Sharpe (capped {cap:.0%})", ef.clean_weights(),
             ef.portfolio_performance(risk_free_rate=RISK_FREE))
    else:
        print("== Max-Sharpe ==\n   skipped: no fund has a return above the risk-free rate\n")

    ef2 = EfficientFrontier(mu, S, weight_bounds=(0, cap))
    ef2.min_volatility()
    show(f"Min-Variance (capped {cap:.0%})", ef2.clean_weights(),
         ef2.portfolio_performance(risk_free_rate=RISK_FREE))

    hrp = HRPOpt(returns)
    hrp.optimize()
    show("HRP", hrp.clean_weights(),
         hrp.portfolio_performance(risk_free_rate=RISK_FREE, frequency=freq))

    eq = pd.Series(1 / n, index=mu.index)
    er, ev = float(eq @ mu), float(np.sqrt(eq @ S @ eq))
    print(f"== Equal weight ==\n   -> return {er * 100:5.1f}%  vol {ev * 100:5.1f}%  "
          f"Sharpe {(er - RISK_FREE) / ev:.2f}\n")

    print("Reminder: Max-Sharpe is the least reliable; HRP / Min-Variance are more robust.\n"
          "Backtest walk-forward before acting.")


def build_chf_prices(tracker_xlsx="funds-daily-NAV-tracker.xlsx", fx_csv="_fx_chf.csv"):
    """Optional helper: pivot a long 'Daily NAV' sheet (Date, Fund, NAV, NAV currency)
    to a wide table and convert each fund to CHF.
    fx_csv columns: date, USDCHF, EURCHF, GBPCHF (CHF per 1 unit).
    Save the result with .to_csv("_chf_prices.csv", index_label="date")."""
    nav = pd.read_excel(tracker_xlsx, sheet_name="Daily NAV")
    nav.columns = [c.strip() for c in nav.columns]
    wide = nav.pivot_table(index="Date", columns="Fund", values="NAV")
    wide.index = pd.to_datetime(wide.index)
    ccy = nav.groupby("Fund")["NAV currency"].last()
    fx = pd.read_csv(fx_csv, parse_dates=["date"]).set_index("date").sort_index()
    out = pd.DataFrame(index=wide.index)
    for fund in wide.columns:
        c = ccy.get(fund, "CHF")
        if c == "CHF":
            out[fund] = wide[fund]
        else:
            out[fund] = wide[fund] * fx[f"{c}CHF"].reindex(out.index).ffill()
    return out.dropna(how="any")


if __name__ == "__main__":
    main()
