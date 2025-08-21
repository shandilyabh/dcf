# terminal-dcf-tool


### Usage examples:
  > python terminal-dcf-tool.py NSE:MRF
  > python terminal-dcf-tool.py NSE:MRF --years 5 --growth 0.10 --wacc 0.12 --terminal-growth 0.03

### Notes & assumptions (kept minimal):
 - Maps common exchanges to yfinance suffixes (NSE -> .NS, BSE -> .BO, NASDAQ/NYSE -> no suffix)
 - Attempts to compute historical Free Cash Flow as: Operating Cash Flow - Capital Expenditures
 - If historical FCF not available, falls back to `freeCashflow` from yfinance.info
 - Projects FCF for `n` years using historic CAGR of FCF if available, otherwise uses user-provided growth
 - Discounts projected FCF and terminal value using user-supplied WACC (default 10%)
 - Terminal value uses Gordon Growth model by default
 - Outputs fair value per share and simple sensitivity table over discount rate and terminal growth
 - Outputs a graph for the price of the stock for the last year.

### To Do:
 - add robust error handling for missing/odd financial statements,
 - fetch accurate country risk premium and risk-free rate,
 - compute WACC from capital structure and market data,
 - validate currency & share counts, and
 - respect rate/rounding conventions for the exchange.