"""
Stage 12: Live cloud dashboard.

Polls your deployed /predict API on a schedule and renders a ticking
sentiment feed.

Run locally:
    streamlit run dashboard.py
"""

import os
import time
from collections import Counter

import requests
import streamlit as st

API_URL = os.getenv("SENTIMENT_API_URL", "http://localhost:8000")
POLL_SECONDS = 3

SAMPLE_POOL = [
    "Just got promoted at work! Best day ever!!",
    "Traffic this morning was absolutely horrible, so late now.",
    "Watching the sunset from my balcony, so peaceful right now.",
    "My internet has been down for 3 hours, so done with this provider.",
    "Grabbed coffee with an old friend today, felt really nice.",
    "This new restaurant downtown was a total letdown, overpriced and bland.",
    "Finally finished my project after weeks of work, feeling accomplished!",
    "Ugh, spilled coffee all over my laptop this morning.",
    "The new movie was okay, nothing special but not bad either.",
    "Can't stop smiling, just adopted the cutest puppy today!",
]

st.set_page_config(page_title="Sentiment Pipeline - Live", layout="wide")
st.title("Real-Time Sentiment Analysis")
st.caption(f"Calling live model API at: {API_URL}")

if "feed" not in st.session_state:
    st.session_state.feed = []
    st.session_state.counts = Counter()
    st.session_state.i = 0

col1, col2, col3 = st.columns(3)
metric_total = col1.empty()
metric_pos = col2.empty()
metric_neg = col3.empty()

feed_placeholder = st.empty()
status = st.empty()


def call_api(text: str):
    try:
        resp = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=5)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return None, str(e)


run = st.toggle("Stream live predictions", value=True)

while run:
    text = SAMPLE_POOL[st.session_state.i % len(SAMPLE_POOL)]
    st.session_state.i += 1

    result, err = call_api(text)

    if err:
        status.error(f"API call failed: {err}")
    else:
        status.success("Connected")
        st.session_state.counts[result["sentiment"]] += 1
        st.session_state.feed.insert(0, result)
        st.session_state.feed = st.session_state.feed[:12]

    total = sum(st.session_state.counts.values())
    pos = st.session_state.counts.get("Positive", 0)
    neg = st.session_state.counts.get("Negative", 0)

    metric_total.metric("Processed", total)
    metric_pos.metric("Positive %", f"{round(pos/total*100) if total else 0}%")
    metric_neg.metric("Negative %", f"{round(neg/total*100) if total else 0}%")

    with feed_placeholder.container():
        for item in st.session_state.feed:
            tag = "🟢" if item["sentiment"] == "Positive" else (
                "🔴" if item["sentiment"] == "Negative" else "⚪")
            st.write(f"{tag} **{item['sentiment']}** ({item['confidence']}%) — {item['text']}")

    time.sleep(POLL_SECONDS)