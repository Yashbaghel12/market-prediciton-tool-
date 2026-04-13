import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

print("🚀 Starting Market Prediction Pipeline...")

# =========================
# 1. LOAD DATA
# =========================
data = yf.download("RELIANCE.NS", start="2015-01-01", end="2024-01-01")

# Fix Close column shape
close = data['Close'].squeeze()

# =========================
# 2. FEATURE ENGINEERING
# =========================
data['Return'] = close.pct_change()
data['SMA_10'] = close.rolling(10).mean()
data['SMA_50'] = close.rolling(50).mean()

# RSI
data['RSI'] = ta.momentum.RSIIndicator(close).rsi()

# MACD
macd = ta.trend.MACD(close)
data['MACD'] = macd.macd()

# Target: Next day UP(1) / DOWN(0)
data['Target'] = (close.shift(-1) > close).astype(int)

data = data.dropna()

# =========================
# 3. TRAIN MODEL
# =========================
features = ['Return', 'SMA_10', 'SMA_50', 'RSI', 'MACD']
X = data[features]
y = data['Target']

# Time-based split
split = int(len(data) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Model
model = XGBClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

# Predictions
preds = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, preds)
print("🔥 Model Accuracy:", acc)

# =========================
# 4. BACKTESTING (PROFIT)
# =========================
results = data.iloc[split:].copy()
results['Prediction'] = preds

# Strategy: only invest when model says UP
results['Strategy_Return'] = results['Return'] * results['Prediction']

# Cumulative returns
results['Cumulative_Market'] = (1 + results['Return']).cumprod()
results['Cumulative_Strategy'] = (1 + results['Strategy_Return']).cumprod()

print("\n📈 Final Market Return:", results['Cumulative_Market'].iloc[-1])
print("💰 Final Strategy Return:", results['Cumulative_Strategy'].iloc[-1])

# =========================
# 5. VISUALIZATION
# =========================
results[['Cumulative_Market', 'Cumulative_Strategy']].plot(figsize=(10,5))
plt.title("Market vs Strategy Performance")
plt.show()