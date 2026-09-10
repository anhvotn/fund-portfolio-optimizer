# Fund Portfolio Optimizer (CHF)

A small, self-contained script that compares portfolio-allocation methods on a
set of funds, with everything measured in Swiss francs. It runs four allocators
side by side so you can see how differently they treat the same universe:

- **Max-Sharpe** — mean-variance tangency portfolio (Ledoit-Wolf shrinkage
  covariance), long-only, capped at 20% per fund.
- **Min-Variance** — ignores expected returns and minimises portfolio
  volatility. The most robust of the return-agnostic methods.
- **HRP** — Hierarchical Risk Parity (López de Prado): clusters funds by
  correlation, then allocates. Robust when many funds are near-duplicates.
- **Risk Parity** — equal risk contribution, for reference.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python portfolio_optimize.py
```

By default it reads `_chf_prices.csv` (date + one CHF price column per fund) and
prints each method's weights plus the portfolio's annualised return, volatility
and Sharpe ratio.

## Input data

`_chf_prices.csv` is a wide table of daily prices, already converted to CHF:

```
date,FundA,FundB,...
2025-09-09,31.9266,3.3390,...
```

The sample file covers ~1 year of daily history for a set of Swiss-available
sector/thematic ETFs. To rebuild it from your own daily-NAV spreadsheet (once it
has enough overlapping history), see `build_chf_prices()` in the script — you
supply daily FX (CHF per USD/EUR/GBP) and it pivots and converts for you.

## Important caveats

**This is exploratory analysis, not investment advice.** In particular:

1. "Expected returns" are just trailing historical means. Over a strong year
   they are wildly optimistic and are **not** a forecast — Max-Sharpe will tilt
   into whatever recently outperformed, so treat it as illustrative.
2. Min-Variance and HRP are the robust choices; they down-weight high-volatility
   holdings and lean on the low-volatility broad funds.
3. Very high in-sample Sharpe ratios (e.g. > 3) are a warning sign of
   overfitting to one lucky period, not a repeatable edge.
4. Always validate **out-of-sample** (walk-forward): refit on a rolling window,
   hold out the next period, and compare *realised* results across methods
   before acting on any weights.

## License

MIT — see `LICENSE`, or add your own.
