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

close = data['Close'].squeeze()

# =========================
# 2. FEATURES
# =========================
data['Return'] = close.pct_change()
data['SMA_10'] = close.rolling(10).mean()
data['SMA_50'] = close.rolling(50).mean()

data['RSI'] = ta.momentum.RSIIndicator(close).rsi()

macd = ta.trend.MACD(close)
data['MACD'] = macd.macd()

# Target
data['Target'] = (close.shift(-1) > close).astype(int)

data = data.dropna()

# =========================
# 3. TRAIN MODEL
# =========================
features = ['Return', 'SMA_10', 'SMA_50', 'RSI', 'MACD']

X = data[features]
y = data['Target']

split = int(len(data) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = XGBClassifier(n_estimators=100, max_depth=5)
model.fit(X_train, y_train)

preds = model.predict(X_test)

acc = accuracy_score(y_test, preds)
print("🔥 Model Accuracy:", acc)

# =========================
# 4. BACKTEST + STRATEGY
# =========================
results = data.iloc[split:].copy()
results['Prediction'] = preds

# Smart entry (AI + RSI)
results['Buy_Signal'] = (
    (results['Prediction'] == 1) & (results['RSI'] < 40)
).astype(int)

# HOLD LOGIC
position = 0
positions = []

for i in range(len(results)):
    if results['Buy_Signal'].iloc[i] == 1:
        position = 1  # BUY
    elif results['RSI'].iloc[i] > 60:
        position = 0  # SELL

    positions.append(position)

results['Position'] = positions

# Apply returns
results['Strategy_Return'] = results['Return'] * results['Position']

# Cumulative performance
results['Cumulative_Market'] = (1 + results['Return']).cumprod()
results['Cumulative_Strategy'] = (1 + results['Strategy_Return']).cumprod()

print("\n📈 Final Market Return:", results['Cumulative_Market'].iloc[-1])
print("💰 Final Strategy Return:", results['Cumulative_Strategy'].iloc[-1])

# =========================
# 5. GRAPH
# =========================
results[['Cumulative_Market', 'Cumulative_Strategy']].plot(figsize=(10,5))
plt.title("Market vs Strategy Performance")
plt.show()