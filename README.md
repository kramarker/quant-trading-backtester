# Quant Trading Strategy Backtester

A Python-based backtesting engine for quantitative trading strategies using real historical market data.

## Features
- Downloads market data with Yahoo Finance
- Computes technical indicators:
  - SMA
  - EMA
  - RSI
  - Rolling volatility
- Implements multiple strategies:
  - Moving Average Crossover
  - RSI Mean Reversion
  - Buy and Hold benchmark
- Simulates portfolio performance with transaction costs
- Computes performance metrics:
  - Total return
  - Annualized return
  - Annualized volatility
  - Sharpe ratio
  - Max drawdown
  - Win rate
  - Trade statistics
- Saves:
  - trade logs
  - summary metrics
  - equity curves
  - drawdown plots
  - signal plots

## Project Structure
```text
quant-trading-backtester/
│
├── data/
├── notebooks/
├── plots/
├── results/
├── src/
├── main.py
├── requirements.txt
└── README.md