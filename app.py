import streamlit as st
import pandas as pd
from transformers import pipeline

st.set_page_config(page_title="AI Risk Engine & Portfolio Dashboard", layout="wide")
st.title("📈 AI/NLP Financial Risk Engine & Dynamic Rebalancer")

# 1. FinBERT pipeline cache
@st.cache_resource
def load_sentiment_model():
    return pipeline("text-classification", model="ProsusAI/finbert")

st.sidebar.header("Data Stream Config")
source_option = st.sidebar.selectbox("Data Feed", ["GDELT Feed", "NewsAPI Stream"])

sample_news = [
    {"ticker": "AAPL", "text": "Apple reports record quarterly revenue driven by strong enterprise and consumer demand."},
    {"ticker": "TSLA", "text": "Regulatory investigation launched over autonomous driving safety concerns and supply chain disruptions."},
    {"ticker": "MSFT", "text": "Microsoft expands cloud infrastructure and announces major strategic partnership in artificial intelligence."},
    {"ticker": "AMZN", "text": "Amazon faces rising operational costs and logistical delays impacting global distribution networks."}
]

# 2. Risk Engine Processing
st.header("1. Core AI/NLP Risk Engine")

nlp_pipe = load_sentiment_model()
processed_signals = []

for item in sample_news:
    res = nlp_pipe(item["text"])[0]
    label = res["label"]
    score = res["score"]
    
    # Calculate score (-1.0 to 1.0)
    if label == "positive":
        num_score = score
    elif label == "negative":
        num_score = -score
    else:
        num_score = 0.0
        
    t_lower = item["text"].lower()
    if "regulatory" in t_lower or "investigation" in t_lower:
        event = "Credit/Regulatory"
        impact = 8.5
    elif "revenue" in t_lower or "partnership" in t_lower:
        event = "Macroeconomic/Earnings"
        impact = 6.0
    else:
        event = "Operational/Supply Chain"
        impact = 5.0

    processed_signals.append({
        "Ticker": item["ticker"],
        "Headline": item["text"],
        "Sentiment Label": label.capitalize(),
        "Sentiment Score": round(num_score, 2),
        "Event Type": event,
        "Impact Score (1-10)": impact
    })

df_signals = pd.DataFrame(processed_signals)
st.dataframe(df_signals, use_container_width=True)

# 3. Module A: Index Rebalancer
st.header("2. Module A: Tactical Index Rebalancing")

tickers = ["AAPL", "TSLA", "MSFT", "AMZN"]
base_weight = 1.0 / len(tickers)

adjusted_weights = {}
for row in processed_signals:
    t = row["Ticker"]
    s = row["Sentiment Score"]
    adjusted_weights[t] = max(0.05, base_weight * (1 + 0.25 * s))

total_w = sum(adjusted_weights.values())
final_weights = {k: v / total_w for k, v in adjusted_weights.items()}

df_weights = pd.DataFrame({
    "Ticker": tickers,
    "Baseline Weight": [base_weight] * len(tickers),
    "Rebalanced Weight": [final_weights[t] for t in tickers]
})

col1, col2 = st.columns(2)
with col1:
    st.subheader("Weight Redistribution")
    st.dataframe(df_weights.style.format({"Baseline Weight": "{:.2%}", "Rebalanced Weight": "{:.2%}"}), use_container_width=True)

with col2:
    st.subheader("Dynamic Allocation Chart")
    st.bar_chart(df_weights.set_index("Ticker"))