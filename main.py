import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from transformers import pipeline
from newsapi import NewsApiClient

print("🚀 Starting MULTI-MODEL AI + FinBERT System...")

# =========================
# 1. FINBERT
# =========================
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def get_sentiment_score(text):
    result = finbert(text)[0]
    if result['label'] == 'positive':
        return result['score']
    elif result['label'] == 'negative':
        return -result['score']
    return 0

# =========================
# 2. NEWS API
# =========================
newsapi = NewsApiClient(api_key="YOUR_API_KEY")

# =========================
# 3. STOCK LIST
# =========================
stocks = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "LT.NS","SBIN.NS","BHARTIARTL.NS","ITC.NS","KOTAKBANK.NS"
]

# =========================
# 4. DATA
# =========================
data = yf.download(stocks, start="2018-01-01", end="2024-01-01")["Close"]

data = data.dropna(axis=1, thresh=len(data)*0.7)
data = data.ffill().dropna()

returns = data.pct_change().dropna()

# =========================
# 5. PCA
# =========================
pca = PCA(n_components=8)
pca_features = pca.fit_transform(returns)

pca_df = pd.DataFrame(pca_features, index=returns.index)
pca_df.columns = [f"PCA_{i}" for i in range(8)]

print("🧠 PCA Variance:", sum(pca.explained_variance_ratio_))

# =========================
# 6. RELIANCE FEATURES
# =========================
rel_close = data["RELIANCE.NS"]

df = pd.DataFrame(index=returns.index)

df["REL_Return"] = returns["RELIANCE.NS"]
df["RSI"] = ta.momentum.RSIIndicator(rel_close).rsi()

macd = ta.trend.MACD(rel_close)
df["MACD"] = macd.macd()

df["SMA_10"] = rel_close.rolling(10).mean().pct_change()
df["SMA_50"] = rel_close.rolling(50).mean().pct_change()

df["Momentum"] = df["REL_Return"].rolling(3).mean()
df["REL_lag1"] = df["REL_Return"].shift(1)
df["REL_lag2"] = df["REL_Return"].shift(2)

# =========================
# 7. FINBERT SENTIMENT
# =========================
sentiment_map = {}
print("📰 Fetching sentiment...")

for date in df.index[::7]:
    try:
        articles = newsapi.get_everything(
            q="Reliance stock",
            from_param=date.strftime("%Y-%m-%d"),
            to_param=(date + pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
            language="en"
        )

        if articles["articles"]:
            text = " ".join([a["title"] for a in articles["articles"][:5] if a["title"]])
            sentiment_map[date] = get_sentiment_score(text)
        else:
            sentiment_map[date] = 0
    except:
        sentiment_map[date] = 0

sentiments = []
last = 0
for d in df.index:
    if d in sentiment_map:
        last = sentiment_map[d]
    sentiments.append(last)

df["Sentiment"] = sentiments

# =========================
# 8. MERGE PCA
# =========================
df = pd.concat([df, pca_df], axis=1)

# =========================
# 9. TARGET
# =========================
df["Target"] = (df["REL_Return"].shift(-1) > 0).astype(int)
df = df.dropna()

# =========================
# 10. TRAIN TEST
# =========================
X = df.drop("Target", axis=1)
y = df["Target"]

split = int(len(df)*0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# =========================
# 11. MODELS
# =========================

# XGBoost
model_xgb = XGBClassifier(n_estimators=300, max_depth=6)
model_xgb.fit(X_train, y_train)
pred_xgb = model_xgb.predict(X_test)

# Random Forest
model_rf = RandomForestClassifier(n_estimators=200, max_depth=8)
model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)

# Accuracy
acc_xgb = accuracy_score(y_test, pred_xgb)
acc_rf = accuracy_score(y_test, pred_rf)

print("🔥 XGB Accuracy:", acc_xgb)
print("🌲 RF Accuracy:", acc_rf)

# =========================
# 12. STRATEGY
# =========================
results = df.iloc[split:].copy()

results["Pred_XGB"] = pred_xgb
results["Pred_RF"] = pred_rf

def run_strategy(pred_col):
    position = 0
    positions = []

    for i in range(len(results)):
        rsi = results["RSI"].iloc[i]
        pred = results[pred_col].iloc[i]
        sentiment = results["Sentiment"].iloc[i]

        if (pred == 1) and (rsi < 70) and (sentiment > -0.2):
            position = 1
        elif (rsi > 80) or (sentiment < -0.3):
            position = 0

        positions.append(position)

    return positions

results["Pos_XGB"] = run_strategy("Pred_XGB")
results["Pos_RF"] = run_strategy("Pred_RF")

# =========================
# 13. RETURNS
# =========================
results["Ret_XGB"] = results["REL_Return"] * results["Pos_XGB"]
results["Ret_RF"] = results["REL_Return"] * results["Pos_RF"]

results["Strat_XGB"] = (1 + results["Ret_XGB"]).cumprod()
results["Strat_RF"] = (1 + results["Ret_RF"]).cumprod()

results["Market"] = (1 + results["REL_Return"]).cumprod()

print("\n📈 Market Return:", results["Market"].iloc[-1])
print("💰 XGB Return:", results["Strat_XGB"].iloc[-1])
print("🌲 RF Return:", results["Strat_RF"].iloc[-1])

# =========================
# 14. GRAPH
# =========================
results[["Market","Strat_XGB","Strat_RF"]].plot(figsize=(12,6))
plt.title("Multi-Model Strategy Comparison")
plt.show()