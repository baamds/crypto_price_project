
#  Crypto Price Prediction — Prisprognoser & Marknadsanalys  
**Maskininlärning • Tidsserier • Backtesting • Feature Engineering**

Ett projekt för att **förutsäga kryptopriser** (ex. Bitcoin, Ethereum) med klassiska ML-metoder, tidsseriemodeller och portfölj-simuleringar.

Projektet inkluderar:

- Datainsamling via **yfinance / CCXT**  
- Feature Engineering: volatilitet, momentum, RSI, SMA, m.m.  
- Modeller: **Random Forest**, ARIMA, Prophet, LSTM (Keras)  
- Backtesting & simulering av tradingstrategier  
- Modulär kodbas + färdiga scripts  
- Notebook-exempel för analys  

---

##  Projektstruktur

```
crypto_price_project/
│
├── fetch_data.py           # Hämta OHLC-data (yfinance / CCXT)
├── features.py             # Feature engineering
├── models.py               # ML- och tidsseriemodeller (RF, ARIMA, Prophet, LSTM)
├── backtest.py             # Enkel portfölj-simulering
├── notebook_launcher.py    # Kör hela ML-pipelinen
│
├── utils.py                # Hjälpfunktioner
├── requirements.txt        # Installationsberoenden
├── README.md               # Dokumentation
└── data/                   # Datafiler (skapas automatiskt)
```

---

##  Kom igång

### 1. Klona repo

```bash
git clone https://github.com/baamds/crypto_price_project.git
cd crypto_price_project
```

### 2. Skapa & aktivera virtuellt environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Installera beroenden

```bash
pip install -r requirements.txt
```

---

## Hämta historisk prisdata

Exempel (Bitcoin):

```bash
python fetch_data.py --source yfinance \
    --symbol BTC-USD \
    --start 2018-01-01 \
    --end 2024-12-31 \
    --out data/btc_usd.csv
```

---

## Kör Feature Engineering

```bash
python features.py --in data/btc_usd.csv --out data/btc_features.csv
```

---

## Träna modeller & generera signaler

```bash
python notebook_launcher.py --data data/btc_features.csv
```

Output inkluderar:

- `results/rf_model.joblib`  
- `results/signals.csv` (prediktioner + trading-signaler)

---

## Backtesting

```bash
python backtest.py --signals results/signals.csv --initial-capital 10000
```

Ger dig portfölj-NAV över tid i:

```
results/backtest_nav.csv
```

---

## Modeller i projektet

| Modell | Beskrivning | Typ |
|--------|-------------|-----|
| Random Forest | Snabb, robust baseline | ML |
| ARIMA | Klassisk tidserie-modell | Tidsserie |
| Prophet | Facebooks modell för säsong/trend | Tidsserie |
| LSTM | Rekurrent neuralt nätverk | Deep Learning |

---

## Notebook Exempel

Starta notebook:

```bash
jupyter notebook
```

---
