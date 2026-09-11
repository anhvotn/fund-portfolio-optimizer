"""
Portfolio optimisation for the Trading fund set (CHF base).

What it does
------------
Loads a matrix of daily prices already expressed in CHF, computes returns,
and runs four allocation methods side by side:

  * Max-Sharpe   - mean-variance tangency portfolio (Ledoit-Wolf shrinkage
                   covariance), long-only, capped at 20% per fund.
  * Min-Variance - ignores expected returns, minimises portfolio volatility
                   (the most robust of the return-agnostic methods).
  * HRP          - Hierarchical Risk Parity (Lopez de Prado): clusters funds
                   by correlation, then allocates - robust to the noisy,
                   near-duplicate covariance you get from many correlated funds.
  * Risk Parity  - equal risk contribution (via HRP's cousin), for reference.

IMPORTANT CAVEATS (read before trusting any number)
---------------------------------------------------
1. "Expected returns" here are just trailing historical means. Over a boom year
   for energy/miners/tech they are wildly optimistic and NOT a forecast. Max-Sharpe
   will therefore tilt hard into whatever just ran - treat it as illustrative.
2. Min-Variance and HRP are the robust choices; they down-weight the volatile
   miners and lean on the low-volatility broad funds.
3. This is analysis to inform your own decision, not investment advice.
4. Validate out-of-sample (walk-forward) before acting - see backtest note at end.

Requirements:  pip install PyPortfolioOpt pandas numpy openpyxl
Usage:         python portfolio_optimize.py
"""

import os
import numpy as np
import pandas as pd
from pypfopt import expected_returns, risk_models, EfficientFrontier
from pypfopt.hierarchical_portfolio import HRPOpt

PRICES_CSV  = "_chf_prices.csv"          # your real data (git-ignored, private)
EXAMPLE_CSV = "_chf_prices.example.csv"  # synthetic fallback shipped in the repo
WEIGHT_CAP  = 0.20                        # max weight per fund for the MVO methods
RISK_FREE   = 0.0                         # CHF cash ~0; adjust if you like

# ----------------------------------------------------------------------
# 1. Load CHF prices  (rebuild this file from the tracker once it has
#    enough overlapping daily history - see build_chf_prices() below.)
# ----------------------------------------------------------------------
# Use your real prices if present, otherwise fall back to the synthetic example
# so the script runs immediately on a fresh clone.
csv_path = PRICES_CSV if os.path.exists(PRICES_CSV) else EXAMPLE_CSV
if csv_path == EXAMPLE_CSV:
    print(f"[note] '{PRICES_CSV}' not found - running on synthetic example data "
          f"('{EXAMPLE_CSV}'). Copy the example to '{PRICES_CSV}' and add your "
          f"own prices for real results.\n")

prices = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date").sort_index()
prices = prices.dropna(how="any")
print(f"Loaded {prices.shape[1]} funds x {len(prices)} trading days "
      f"({prices.index.min().date()} -> {prices.index.max().date()})\n")

returns = prices.pct_change().dropna()

# Expected returns (trailing mean, annualised) and shrinkage covariance
mu = expected_returns.mean_historical_return(prices)
S  = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

def show(name, weights, perf):
    print(f"== {name} ==")
    for k, v in sorted(weights.items(), key=lambda x: -x[1]):
        if round(v, 4) != 0:
            print(f"   {k:<20} {v*100:6.1f}%")
    print(f"   -> return {perf[0]*100:5.1f}%  vol {perf[1]*100:5.1f}%  Sharpe {perf[2]:.2f}\n")

# ----------------------------------------------------------------------
# 2. Max-Sharpe (capped, long-only)
# ----------------------------------------------------------------------
ef = EfficientFrontier(mu, S, weight_bounds=(0, WEIGHT_CAP))
ef.max_sharpe(risk_free_rate=RISK_FREE)
show("Max-Sharpe (capped 20%)", ef.clean_weights(),
     ef.portfolio_performance(risk_free_rate=RISK_FREE))

# ----------------------------------------------------------------------
# 3. Min-Variance (capped, long-only)
# ----------------------------------------------------------------------
ef2 = EfficientFrontier(mu, S, weight_bounds=(0, WEIGHT_CAP))
ef2.min_volatility()
show("Min-Variance (capped 20%)", ef2.clean_weights(),
     ef2.portfolio_performance(risk_free_rate=RISK_FREE))

# ----------------------------------------------------------------------
# 4. Hierarchical Risk Parity (robust; no cap needed - it self-diversifies)
# ----------------------------------------------------------------------
hrp = HRPOpt(returns)
hrp.optimize()
show("HRP", hrp.clean_weights(),
     hrp.portfolio_performance(risk_free_rate=RISK_FREE))

# ----------------------------------------------------------------------
# 5. (Optional) rebuild CHF prices from the daily NAV tracker.
#    Only useful once each fund has a long, overlapping daily series.
#    You must supply daily FX (CHF per USD / EUR / GBP) - e.g. from Yahoo.
# ----------------------------------------------------------------------
def build_chf_prices(tracker_xlsx="funds-daily-NAV-tracker.xlsx",
                     fx_csv="_fx_chf.csv"):
    """Pivot the long 'Daily NAV' sheet to wide, convert each column to CHF.
    fx_csv columns: date, USDCHF, EURCHF, GBPCHF (CHF per 1 unit).
    Returns a wide CHF price frame you can feed to this script."""
    nav = pd.read_excel(tracker_xlsx, sheet_name="Daily NAV")
    nav.columns = [c.strip() for c in nav.columns]
    wide  = nav.pivot_table(index="Date", columns="Fund", values="NAV")
    ccy   = nav.groupby("Fund")["NAV currency"].last()
    fx    = pd.read_csv(fx_csv, parse_dates=["date"]).set_index("date")
    out = pd.DataFrame(index=pd.to_datetime(wide.index))
    for fund in wide.columns:
        c = ccy.get(fund, "CHF")
        if c == "CHF":
            out[fund] = wide[fund].values
        else:
            rate = fx[f"{c}CHF"].reindex(out.index).ffill()
            out[fund] = wide[fund].values * rate.values
    return out.dropna(how="any")

if __name__ == "__main__":
    print("Done. Reminder: expected-return-based weights (Max-Sharpe) are the "
          "least reliable; HRP / Min-Variance are the robust ones.\n"
          "Backtest walk-forward before acting: refit on a rolling window, "
          "hold out the next period, and compare realised Sharpe across methods.")
