import argparse
import pandas as pd
import numpy as np

def run_backtest(signals_path, price_col='close', dt_col='timestamp', initial_capital=10000):
    df = pd.read_csv(signals_path, parse_dates=[dt_col])
    df = df.sort_values(dt_col).reset_index(drop=True)
    capital = initial_capital
    position = 0.0
    cash = capital
    holdings = 0.0
    history = []
    for i, row in df.iterrows():
        price = row[price_col]
        sig = int(row.get('signal', 0))
        if sig == 1 and cash > 0:
            # buy with all cash
            holdings = cash / price
            cash = 0.0
        elif sig == 0 and holdings > 0:
            # sell all
            cash = holdings * price
            holdings = 0.0
        nav = cash + holdings * price
        history.append({'timestamp': row[dt_col], 'nav': nav, 'cash': cash, 'holdings': holdings})
    hist = pd.DataFrame(history)
    return hist

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--signals', dest='signals', required=True)
    parser.add_argument('--initial-capital', dest='initial_capital', type=float, default=10000)
    args = parser.parse_args()
    hist = run_backtest(args.signals, initial_capital=args.initial_capital)
    out = 'results/backtest_nav.csv'
    hist.to_csv(out, index=False)
    print(f"Saved backtest NAV to {out}")

if __name__ == '__main__':
    main()
