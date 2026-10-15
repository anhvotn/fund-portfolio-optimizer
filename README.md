# Fund Portfolio Optimizer (CHF)

Two small scripts for a set of funds measured in Swiss francs:

- **`portfolio_optimize.py`** compares allocation methods side by side:
  - **Max-Sharpe**: mean-variance tangency portfolio. It uses a Ledoit-Wolf shrinkage covariance, is long-only and caps each fund at 20%.
  - **Min-Variance**: ignores expected returns and minimises volatility.
  - **HRP**: Hierarchical Risk Parity (López de Prado). It clusters funds by correlation, then allocates, so it handles near-duplicate funds well.
  - **Equal weight**: a benchmark.
- **`rebalance_backtest.py`** tests rebalancing rules for a set of target weights, including trading costs and minimum fees:
  - buy & hold;
  - monthly, quarterly, semi-annual or annual rebalancing;
  - ±3pp or ±5pp bands;
  - a ±25% relative band;
  - a quarterly check that only rebalances when a fund is more than ±5pp off target.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python portfolio_optimize.py                     # reads _chf_prices.csv
python portfolio_optimize.py my_prices.csv       # or any price file

python rebalance_backtest.py                     # bundled examples
python rebalance_backtest.py my_prices.csv my_weights.csv
```

If `_chf_prices.csv` is missing, both scripts fall back to the bundled synthetic examples, so a fresh clone runs out of the box. The example results mean nothing.

## Input data

**Prices.** A wide CSV with a `date` column and one CHF price column per fund, with dividends reinvested. Daily, weekly or monthly data is detected automatically.

```
date,FundA,FundB,...
2023-10-06,100.00,50.00,...
```

**Target weights** (for the rebalancing backtest). A CSV with `fund,weight` columns, where `fund` matches a price column. Weights are normalised to sum to 1.

**Bring your own data.** `_chf_prices.csv`, `_target_weights.csv` and spreadsheets are git-ignored, because a real price file reveals which funds you hold.

`build_chf_prices()` in `portfolio_optimize.py` can build the price file from a daily-NAV spreadsheet plus daily FX rates (CHF per USD/EUR/GBP).

## Notes

- **Weight cap.** The 20% cap is raised automatically to 1/n when there are fewer than 5 funds; otherwise the weights could not add up to 100%.
- **Max-Sharpe needs a positive return.** It is skipped if no fund beat the risk-free rate over the sample.

## Important caveats

**This is exploratory analysis, not investment advice.**

1. **Expected returns are trailing historical means, not a forecast.** Max-Sharpe tilts into whatever recently outperformed, so treat it as illustrative.
2. **Min-Variance and HRP are the more robust choices.**
3. **Very high in-sample Sharpe ratios (> 3) suggest overfitting.**
4. **Validate out of sample before acting on any weights.** Refit on a rolling window, hold out the next period, and compare realised results.

## License

MIT, see `LICENSE`.
