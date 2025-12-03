# Quantitative Trading Strategy Backtester & Cross-Asset Generalization Study

## Overview

This project implements a multi-strategy quantitative trading backtesting system with an extension into cross-asset generalization research.

Key features:
- Multi-strategy framework (Moving Average, RSI, Bollinger Bands)
- Train/Test evaluation pipeline
- Parameter optimization (MA crossover)
- Risk-adjusted performance metrics (Sharpe, Sortino, drawdown, etc.)
- Cross-asset generalization analysis (train on SPY, evaluate on QQQ/AAPL)

---

## Research Paper
 
**Do Technical Trading Strategies Generalize Across Assets?**  
An empirical study on cross-asset transferability of technical strategies.

[Read the full paper](research/Mark_Antar_Cross_Asset_Generalization.pdf)

This paper analyzes cross-asset generalization of technical trading strategies,
including formal problem definition, empirical results, and overfitting analysis.

## Data

The project uses daily historical OHLCV data from Yahoo Finance.

Tested assets:
- **SPY**: S&P 500 ETF proxy
- **QQQ**: Nasdaq-100 ETF proxy
- **AAPL**: Individual equity

This project evaluates **daily stock/ETF data**, not futures or options.

---

## Strategies

### Moving Average Crossover
- Long when short SMA > long SMA  
- Exit when short SMA <= long SMA  
- Parameters optimized using grid search on the training set  

### RSI Mean Reversion
- Enter when RSI < 30  
- Exit when RSI > 55  

### Bollinger Mean Reversion
- Enter when price falls below lower Bollinger Band  
- Exit when price reverts above rolling mean  

### Buy and Hold
- Benchmark strategy with continuous exposure  

---

## Methodology

### Train/Test Split
- 70% training sample  
- 30% test sample  
- MA parameters optimized on training data only  
- Performance evaluated both in-sample and out-of-sample  

### Optimization
The moving-average strategy is optimized across multiple short/long window combinations using Sharpe ratio as the objective.

### Metrics

Performance metrics:
- Total Return  
- Annualized Return  
- Annualized Volatility  
- Downside Volatility  
- Sharpe Ratio  
- Sortino Ratio  
- Max Drawdown  
- Calmar Ratio  

Trade-level metrics:
- Number of Trades  
- Win Rate  
- Average Trade Return  
- Best/Worst Trade Return  
- Average Holding Days  

Note: trade-level statistics for Buy and Hold are less informative due to continuous exposure.

---

## Sample Results

### Strategy Performance Comparison
![Strategy Comparison](assets/generalization_sharpe.png)

### Total Return Across Assets
![Total Return](assets/generalization_total_return.png)

### MA Parameter Optimization Heatmap
![MA Heatmap](assets/ma_generalization_heatmap.png)

---

## Cross-Asset Generalization Insight

This project extends beyond standard backtesting by evaluating whether strategies trained on one asset (SPY) generalize to others (QQQ, AAPL).

Key findings:
- Mean-reversion strategies (RSI, Bollinger) show more stable cross-asset performance  
- Moving average strategies are sensitive to parameter selection  
- Buy-and-hold achieves higher total returns but with larger drawdowns  
- Strategy robustness varies significantly across assets  

---

## Output Files

Generated outputs include:
- Cleaned summary metrics  
- Raw summary metrics  
- Parameter optimization tables  
- Daily backtest result CSVs  
- Trade logs  
- Visualization plots  

Key files:
- `results/summary_metrics_clean.csv`
- `results/summary_metrics_raw.csv`
- `results/<TICKER>_ma_optimization.csv`
- `plots/strategy_comparison_sharpe.png`

---

## Tech Stack

- Python  
- pandas  
- numpy  
- matplotlib  
- yfinance  

---

## Potential Future Improvements

- Additional strategies  
- Portfolio-level backtesting  
- Walk-forward optimization  
- Transaction cost modeling  
- Intraday and futures extensions  