"""Feature engineering: returns, volatility, momentum, RSI, SMA, EMA."""
import argparse
import pandas as pd
import numpy as np

import pandas as pd
import numpy as np

def find_price_column(df):
    # vanliga kandidater i prioritet
    candidates = ['close', 'adj_close', 'close_adj', 'close_price', 'price', 'last', 'closeusd', 'close_usd']
    cols = [c.lower() for c in df.columns]
    # direkt match
    for cand in candidates:
        if cand in cols:
            return df.columns[cols.index(cand)]
    # om 'close' är i något column name substring
    for i,c in enumerate(cols):
        if 'close' in c:
            return df.columns[i]
    # fallback: välj första numeriska kolumn som inte är timestamp
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.number):
            return c
    # sista utväg: första kolumn efter timestamp identifiering
    return df.columns[1] if len(df.columns) > 1 else df.columns[0]


def find_timestamp_column(df):
    # vanliga namn
    ts_candidates = ['timestamp','date','datetime','time','index']
    cols = [c.lower() for c in df.columns]
    for t in ts_candidates:
        if t in cols:
            return df.columns[cols.index(t)]
    # fallback: första kolumn som kan parses to datetime
    for c in df.columns:
        try:
            pd.to_datetime(df[c])
            return c
        except Exception:
            continue
    # sista utväg
    return df.columns[0]


def compute_features(df):
    df = df.copy()
    # Hitta och namnge timestamp och close-kolumner
    ts_col = find_timestamp_column(df)
    price_col = find_price_column(df)

    # normalize names to expected ones
    df = df.rename(columns={ts_col: 'timestamp', price_col: 'close'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # försök coercea pris-kolumn till float
    df['close'] = pd.to_numeric(df['close'], errors='coerce')

    # basic cleaning: drop rows with missing close
    df = df.dropna(subset=['close']).reset_index(drop=True)

    # returns
    df['ret_1'] = df['close'].pct_change(1)
    df['ret_7'] = df['close'].pct_change(7)

    # volatility (rolling std of returns)
    df['vol_7'] = df['ret_1'].rolling(window=7).std()
    df['vol_21'] = df['ret_1'].rolling(window=21).std()

    # momentum
    df['mom_7'] = df['close'] / df['close'].shift(7) - 1

    # simple moving averages
    df['sma_7'] = df['close'].rolling(7).mean()
    df['sma_21'] = df['close'].rolling(21).mean()

    # RSI (classic) - safe with min_periods
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14, min_periods=1).mean()
    roll_down = down.rolling(14, min_periods=1).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))

    # targets: next-day close / return
    df['target_close_1'] = df['close'].shift(-1)
    df['target_ret_1'] = df['ret_1'].shift(-1)

    # drop rows with na in features/targets (you can tune)
    df = df.dropna().reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='infile', required=True)
    parser.add_argument('--out', dest='outfile', default='data/btc_features.csv')
    args = parser.parse_args()
    df = pd.read_csv(args.infile)
    features = compute_features(df)
    features.to_csv(args.outfile, index=False)
    print(f"Saved features to {args.outfile}")

if __name__ == '__main__':
    main()
