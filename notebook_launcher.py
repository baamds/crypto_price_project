import argparse
import pandas as pd
import numpy as np
from models import train_random_forest, evaluate_model, save_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def pipeline(data_path, out_dir='results'):
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    # select features and drop na
    feat_cols = [c for c in df.columns if c not in ['timestamp','target_close_1','target_ret_1','close']]
    X = df[feat_cols].values
    y = df['target_ret_1'].values
    # train-test split (time-series aware: simple split)
    split = int(len(df)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    os.makedirs(out_dir, exist_ok=True)
    # Train RF
    model = train_random_forest(X_train_s, y_train)
    # evaluate
    eval_train = evaluate_model(model, X_train_s, y_train)
    eval_test = evaluate_model(model, X_test_s, y_test)
    print('Train MSE:', eval_train['mse'])
    print('Test MSE:', eval_test['mse'])
    # save model & scaler & preds
    import joblib
    joblib.dump(model, os.path.join(out_dir, 'rf_model.joblib'))
    joblib.dump(scaler, os.path.join(out_dir, 'scaler.joblib'))
    preds = model.predict(X_test_s)
    out_df = df.iloc[split:].copy()
    out_df['pred_ret'] = preds
    # generate simple signal: pred_ret > 0 -> 1 else 0
    out_df['signal'] = (out_df['pred_ret'] > 0).astype(int)
    out_df.to_csv(os.path.join(out_dir, 'signals.csv'), index=False)
    print('Saved signals to', os.path.join(out_dir, 'signals.csv'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', dest='data', required=True)
    args = parser.parse_args()
    pipeline(args.data)

if __name__ == '__main__':
    main()
