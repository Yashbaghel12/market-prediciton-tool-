import yfinance as yf
import pandas as pd
import numpy as np
import ta
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

from transformers import pipeline
from newsapi import NewsApiClient

print("🚀 Starting FINAL AI + FinBERT System...")

# =========================
# 1. FINBERT LOAD
# =========================
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def get_sentiment_score(text):
    result = finbert(text)[0]
    
    if result['label'] == 'positive':
        return result['score']
    elif result['label'] == 'negative':
        return -result['score']
    else:
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
# 4. DATA DOWNLOAD + CLEAN
# =========================
data = yf.download(stocks, start="2018-01-01", end="2024-01-01")["Close"]

data = data.dropna(axis=1, thresh=len(data)*0.7)
data = data.ffill()
data = data.dropna()

returns = data.pct_change().dropna()

# =========================
# 5. PCA
# =========================
pca = PCA(n_components=12)
pca_features = pca.fit_transform(returns)

pca_df = pd.DataFrame(pca_features, index=returns.index)
pca_df.columns = [f"PCA_{i}" for i in range(12)]

print("🧠 PCA Variance:", sum(pca.explained_variance_ratio_))

# =========================
# 6. RELIANCE FEATURES
# =========================
rel_close = data["RELIANCE.NS"]

df = pd.DataFrame()

df["REL_Return"] = returns["RELIANCE.NS"]

df["RSI"] = ta.momentum.RSIIndicator(rel_close).rsi()

macd = ta.trend.MACD(rel_close)
df["MACD"] = macd.macd()

df["SMA_10"] = rel_close.rolling(10).mean().pct_change()
df["SMA_50"] = rel_close.rolling(50).mean().pct_change()

df["Momentum"] = df["REL_Return"].rolling(3).mean()

# Lag features
df["REL_lag1"] = df["REL_Return"].shift(1)
df["REL_lag2"] = df["REL_Return"].shift(2)

# =========================
# 7. 🔥 FINBERT (WEEKLY REAL NEWS)
# =========================
sentiment_map = {}

print("📰 Fetching weekly sentiment...")

for date in df.index[::7]:
    try:
        articles = newsapi.get_everything(
            q="Reliance stock",
            from_param=date.strftime("%Y-%m-%d"),
            to_param=(date + pd.Timedelta(days=7)).strftime("%Y-%m-%d"),
            language="en"
        )

        if articles["articles"]:
            text = " ".join(
                [a["title"] for a in articles["articles"][:5] if a["title"]]
            )
            sentiment_map[date] = get_sentiment_score(text)
        else:
            sentiment_map[date] = 0

    except:
        sentiment_map[date] = 0

# Fill daily sentiment
sentiments = []
last_sentiment = 0

for date in df.index:
    if date in sentiment_map:
        last_sentiment = sentiment_map[date]
    sentiments.append(last_sentiment)

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
# 10. MODEL
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
# 11. STRATEGY
# =========================
results = df.iloc[split:].copy()
results["Prediction"] = preds

position = 0
positions = []

for i in range(len(results)):
    rsi = results["RSI"].iloc[i]
    pred = results["Prediction"].iloc[i]
    sentiment = results["Sentiment"].iloc[i]

    # BUY
    if (pred == 1) and (rsi < 70) and (sentiment > -0.2):
        position = 1

    # SELL
    elif (rsi > 80) or (sentiment < -0.3):
        position = 0

    positions.append(position)

results["Position"] = positions

# =========================
# 12. RETURNS
# =========================
results["Strategy_Return"] = results["REL_Return"] * results["Position"]

results["Market"] = (1 + results["REL_Return"]).cumprod()
results["Strategy"] = (1 + results["Strategy_Return"]).cumprod()

print("\n📈 Market Return:", results["Market"].iloc[-1])
print("💰 Strategy Return:", results["Strategy"].iloc[-1])

# =========================
# 13. PLOT
# =========================
results[["Market","Strategy"]].plot(figsize=(10,5))
plt.title("FINAL AI + PCA + FinBERT Strategy")
plt.show()