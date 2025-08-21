"""main script"""

import argparse
import math
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt # type: ignore
import pandas as pd
import yfinance as yf # type: ignore
from tabulate import tabulate # type: ignore


EXCHANGE_SUFFIX = {
    'NSE': '.NS',
    'BSE': '.BO',
    'NASDAQ': '',
    'NYSE': '',
    'LSE': '.L',
}


def plot_history(ticker, period='1y'):
    data = yf.Ticker(ticker).history(period=period)
    if data.empty:
        print('Unable to fetch historical prices.')
        return
    plt.figure(figsize=(8, 4))
    plt.plot(data.index, data['Close'], label='Close Price')
    plt.title(f'{ticker} – Last {period} Closing Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def map_ticker(user_ticker: str) -> str:
    """Map a user ticker like 'NSE:MRF' to a yfinance-compatible ticker like 'MRF.NS'.
    If the user provides no exchange (just 'MRF'), return as-is.
    """
    if ':' in user_ticker:
        exch, sym = user_ticker.split(':', 1)
        exch = exch.upper()
        suffix = EXCHANGE_SUFFIX.get(exch, None)
        if suffix is None:
            # unknown exchange: try as-is
            return sym
        return sym + suffix
    return user_ticker


def fetch_financials(tkr: str):
    t = yf.Ticker(tkr)

    # Try to obtain cashflow DataFrame
    try:
        cf = t.cashflow
        # yfinance returns columns as Periods (most recent first); convert to numeric series per year
    except Exception:
        cf = None

    info = t.info or {}

    # Attempt to construct historical FCF series (most recent first)
    fcfs = []
    years = []

    if isinstance(cf, pd.DataFrame) and not cf.empty:
        # common index names -- handle multiple variants
        idx = {i.lower(): i for i in cf.index}
        # find operating cash flow
        op_keys = [k for k in idx.keys() if 'operat' in k and ('cash' in k or 'activities' in k)]
        capex_keys = [k for k in idx.keys() if 'capital' in k and ('expend' in k or 'expenditure' in k)]

        if op_keys and capex_keys:
            op_key = idx[op_keys[0]]
            capex_key = idx[capex_keys[0]]
            for col in cf.columns:
                try:
                    op = float(cf.at[op_key, col])
                    capex = float(cf.at[capex_key, col])
                    fcf = op + capex  # note: capex is usually negative in statements -> op + capex
                    fcfs.append(fcf)
                    if isinstance(col, pd.Timestamp):
                        years.append(str(col.date()))
                    else:
                        years.append(str(col))
                except Exception:
                    continue

    # Fallback to info['freeCashflow'] if historical series isn't available
    if not fcfs:
        freecf = info.get('freeCashflow')
        if freecf:
            fcfs = [float(freecf)]
            years = [str(datetime.now().year)]

    # Market data
    market_price = info.get('currentPrice') or (t.history(period='1d')['Close'].iloc[-1] if not t.history(period='1d').empty else None)
    shares_outstanding = info.get('sharesOutstanding')
    market_cap = info.get('marketCap')

    return {
        'ticker': tkr,
        'fcfs': np.array(fcfs, dtype=float),
        'fcf_years': years,
        'market_price': market_price,
        'shares_outstanding': shares_outstanding,
        'market_cap': market_cap,
        'info': info,
    }


def compute_historical_cagr(series):
    # series: array-like, most recent first
    series = np.array(series)
    # remove non-positive
    series = series[np.isfinite(series)]
    if series.size < 2:
        return None
    # reverse to chronological order
    series = series[::-1]
    start = series[0]
    end = series[-1]
    n = len(series) - 1
    if start <= 0 or end <= 0 or n <= 0:
        return None
    return (end / start) ** (1.0 / n) - 1.0


def project_fcfs(last_fcf, years, growth):
    projected = []
    f = last_fcf
    for _ in range(years):
        f = f * (1 + growth)
        projected.append(f)
    return np.array(projected)


def dcf_valuation(fcfs_proj, wacc, terminal_growth, terminal_method='gordon', shares_outstanding=None, last_fcf=None):
    # discount projected FCFs
    discounts = np.array([(1 + wacc) ** (i + 1) for i in range(len(fcfs_proj))])
    pv_fcfs = np.sum(fcfs_proj / discounts)

    # terminal value
    if terminal_method == 'gordon':
        if terminal_growth >= wacc:
            raise ValueError('Terminal growth must be less than discount rate (WACC) for Gordon model')
        tv = fcfs_proj[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    else:
        # fallback: use exit multiple (not implemented)
        tv = 0

    pv_tv = tv / ((1 + wacc) ** len(fcfs_proj))

    ev = pv_fcfs + pv_tv  # enterprise value approximation assuming FCF is firm-level

    # For simplicity assume EV approximates equity value if net debt small or not computed.
    # If shares_outstanding provided, compute per-share
    per_share = None
    if shares_outstanding and shares_outstanding > 0:
        per_share = ev / shares_outstanding

    return {
        'pv_fcfs': pv_fcfs,
        'pv_tv': pv_tv,
        'ev': ev,
        'per_share': per_share,
        'terminal_value': tv,
    }


def sensitivity_table(base_fcfs_proj, base_wacc, base_tg, wacc_range=None, tg_range=None, shares=None):
    if wacc_range is None:
        wacc_range = [base_wacc - 0.02, base_wacc, base_wacc + 0.02]
    if tg_range is None:
        tg_range = [base_tg - 0.01, base_tg, base_tg + 0.01]

    rows = []
    for w in wacc_range:
        row = [f'{w:.2%}']
        for tg in tg_range:
            try:
                val = dcf_valuation(base_fcfs_proj, w, tg, shares_outstanding=shares)
                ps = val['per_share']
                row.append(f'{ps:.2f}' if ps is not None else 'N/A')
            except Exception:
                row.append('ERR')
        rows.append(row)
    headers = ['WACC \\ TG'] + [f'{tg:.2%}' for tg in tg_range]
    return headers, rows


def human(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 'N/A'
    if abs(x) >= 1e9:
        return f'{x/1e9:.2f}B'
    if abs(x) >= 1e6:
        return f'{x/1e6:.2f}M'
    return f'{x:.2f}'


def main():
    p = argparse.ArgumentParser(description='Simple terminal DCF valuation tool (yfinance-backed)')
    p.add_argument('ticker', help='Ticker in format EXCHANGE:SYMBOL or SYMBOL (eg. NSE:MRF)')
    p.add_argument('--years', type=int, default=5, help='Projection years (default 5)')
    p.add_argument('--growth', type=float, default=None, help='Explicit yearly growth to use if historical not available (e.g. 0.08)')
    p.add_argument('--wacc', type=float, default=0.10, help='Discount rate / WACC (default 10%%)')
    p.add_argument('--terminal-growth', type=float, default=0.03, help='Terminal (perpetual) growth rate (default 3%%)')
    p.add_argument('--use-historical-growth', action='store_true', help='Use historical FCF CAGR if available')
    p.add_argument('--no-warnings', action='store_true')

    args = p.parse_args()
    print("Mapping ticker to yfinance format...")

    yf_tkr = map_ticker(args.ticker)
    print(f"Fetching financial data for {args.ticker}...")
    data = fetch_financials(yf_tkr)

    if data['fcfs'].size == 0:
        if not args.no_warnings:
            print('Warning: Historical FCF not found via yfinance for', yf_tkr)
            print('Try providing --growth or use an API with richer financials.')
        if args.growth is None:
            # fallback default small growth
            growth = 0.05
        else:
            growth = args.growth
        last_fcf = data['info'].get('freeCashflow') or 0
        last_fcf = float(last_fcf) if last_fcf else 0.0
    else:
        last_fcf = float(data['fcfs'][0])  # most recent
        hist_cagr = compute_historical_cagr(data['fcfs'])
        if args.use_historical_growth and hist_cagr is not None:
            growth = hist_cagr
        elif args.growth is not None:
            growth = args.growth
        else:
            # default: use historical CAGR if available, else 5%
            growth = hist_cagr if hist_cagr is not None else 0.05

    print("Projecting future free cash flows...")
    proj = project_fcfs(last_fcf, args.years, growth)

    print("Performing DCF valuation...")
    try:
        result = dcf_valuation(proj, args.wacc, args.terminal_growth, shares_outstanding=data['shares_outstanding'], last_fcf=last_fcf)
    except Exception as e:
        print('Error during DCF:', e)
        sys.exit(1)

    # Output concise results
    print("Generating valuation summary...")
    table = [
        ['Ticker', yf_tkr],
        ['Last reported FCF', human(last_fcf)],
        ['Used FCF growth', f'{growth:.2%}' if growth is not None else 'N/A'],
        ['Projection years', args.years],
        ['Discount rate (WACC)', f'{args.wacc:.2%}'],
        ['Terminal growth', f'{args.terminal_growth:.2%}'],
        ['PV(FCFs)', human(result['pv_fcfs'])],
        ['PV(Terminal)', human(result['pv_tv'])],
        ['Enterprise value (approx)', human(result['ev'])],
        ['Shares outstanding', human(data['shares_outstanding'])],
        ['Fair value / share', f'{result["per_share"]:.2f}' if result['per_share'] is not None else 'N/A'],
        ['Market price (approx)', human(data['market_price'])],
    ]

    print('\n' + tabulate(table, tablefmt='plain'))

    # Sensitivity (small table)
    print("Calculating sensitivity table...")
    headers, rows = sensitivity_table(proj, args.wacc, args.terminal_growth, shares=data['shares_outstanding'])
    print('\nSensitivity (per-share)')
    print(tabulate(rows, headers=headers, tablefmt='grid'))

    print("Plotting historical price...")
    plot_history(yf_tkr, period='1y')


if __name__ == '__main__':
    main()
