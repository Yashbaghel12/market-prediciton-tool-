import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

print("🚀 Starting 50-Stock AI System...")

# =========================
# 1. 50 STOCK LIST (NSE MIX)
# =========================
stocks = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "LT.NS","SBIN.NS","BHARTIARTL.NS","ITC.NS","KOTAKBANK.NS",
    "AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS","HINDUNILVR.NS","SUNPHARMA.NS",
    "TITAN.NS","ULTRACEMCO.NS","NESTLEIND.NS","BAJFINANCE.NS","WIPRO.NS",
    "HCLTECH.NS","TECHM.NS","POWERGRID.NS","NTPC.NS","ONGC.NS",
    "COALINDIA.NS","JSWSTEEL.NS","TATASTEEL.NS","ADANIPORTS.NS","ADANIENT.NS",
    "BAJAJFINSV.NS","HDFCLIFE.NS","SBILIFE.NS","DIVISLAB.NS","DRREDDY.NS",
    "CIPLA.NS","HEROMOTOCO.NS","EICHERMOT.NS","BRITANNIA.NS","DABUR.NS",
    "PIDILITIND.NS","GRASIM.NS","SHREECEM.NS","BPCL.NS","IOC.NS",
    "INDUSINDBK.NS","M&M.NS","SIEMENS.NS","UPL.NS","VEDL.NS"
]

# =========================
# 2. DOWNLOAD DATA
# =========================
data = yf.download(stocks, start="2018-01-01", end="2024-01-01")["Close"]

# Remove columns (stocks) with too many NaNs
data = data.dropna(axis=1, thresh=len(data)*0.7)

# Forward fill small gaps
data = data.ffill()

# Drop remaining NaNs
data = data.dropna()

# =========================
# 3. RETURNS
# =========================
returns = data.pct_change().dropna()
if len(data) == 0:
    raise ValueError("❌ No data left after cleaning. Check stock list.")
# =========================
# 4. PCA (IMPORTANT)
# =========================
pca = PCA(n_components=8)
pca_features = pca.fit_transform(returns)

pca_df = pd.DataFrame(pca_features, index=returns.index)
pca_df.columns = [f"PCA_{i}" for i in range(8)]

print("🧠 PCA Variance Covered:", sum(pca.explained_variance_ratio_))

# =========================
# 5. RELIANCE FEATURES
# =========================
rel_close = data["RELIANCE.NS"]

df = pd.DataFrame()

df["REL_Return"] = returns["RELIANCE.NS"]

df["RSI"] = ta.momentum.RSIIndicator(rel_close).rsi()

macd = ta.trend.MACD(rel_close)
df["MACD"] = macd.macd()

df["SMA_10"] = rel_close.rolling(10).mean().pct_change()
df["SMA_50"] = rel_close.rolling(50).mean().pct_change()

# Lag features
df["REL_lag1"] = df["REL_Return"].shift(1)
df["REL_lag2"] = df["REL_Return"].shift(2)

# =========================
# 6. MERGE PCA
# =========================
df = pd.concat([df, pca_df], axis=1)

# =========================
# 7. TARGET
# =========================
df["Target"] = (df["REL_Return"].shift(-1) > 0).astype(int)
df = df.dropna()

# =========================
# 8. MODEL
# =========================
features = df.columns.drop("Target")

X = df[features]
y = df["Target"]

split = int(len(df) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = XGBClassifier(n_estimators=300, max_depth=6)
model.fit(X_train, y_train)

preds = model.predict(X_test)

print("🔥 Accuracy:", accuracy_score(y_test, preds))

# =========================
# 9. STRATEGY
# =========================
results = df.iloc[split:].copy()
results["Prediction"] = preds

position = 0
positions = []

for i in range(len(results)):
    rsi = results["RSI"].iloc[i]
    pred = results["Prediction"].iloc[i]

    # BUY
    if (pred == 1) and (rsi < 65):
        position = 1

    # SELL
    elif rsi > 75:
        position = 0

    positions.append(position)

results["Position"] = positions

# =========================
# 10. RETURNS
# =========================
results["Strategy_Return"] = results["REL_Return"] * results["Position"]

results["Market"] = (1 + results["REL_Return"]).cumprod()
results["Strategy"] = (1 + results["Strategy_Return"]).cumprod()

print("\n📈 Market Return:", results["Market"].iloc[-1])
print("💰 Strategy Return:", results["Strategy"].iloc[-1])

# =========================
# 11. PLOT
# =========================
results[["Market","Strategy"]].plot(figsize=(10,5))
plt.title("50-Stock PCA Strategy vs Market")
plt.show()