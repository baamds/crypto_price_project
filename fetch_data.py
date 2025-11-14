def fetch_yfinance(symbol='BTC-USD', start='2018-01-01', end=None, interval='1d'):
    import yfinance as yf
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
    # Reset index to make date a column
    df = df.reset_index()

    # Flatten MultiIndex columns if present and normalize to lower-case strings
    new_cols = []
    for c in df.columns:
        if isinstance(c, tuple):
            # join tuple parts with underscore, ignore empty/None parts
            flat = "_".join([str(x) for x in c if x is not None and str(x) != ""])
            new_cols.append(flat.lower())
        else:
            new_cols.append(str(c).lower())
    df.columns = new_cols

    # Ensure a consistent time column name 'timestamp'
    if 'date' in df.columns:
        df = df.rename(columns={'date': 'timestamp'})
    elif 'timestamp' not in df.columns:
        # fallback: assume first column is the datetime index after reset_index
        df = df.rename(columns={df.columns[0]: 'timestamp'})

    # Try to ensure numeric columns exist and are clean
    for col in ['open','high','low','close','volume','adj_close']:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass

    return df
