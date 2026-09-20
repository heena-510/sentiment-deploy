"""
Live cloud dashboard with Gen-Z slang examples.

Polls your deployed /predict API on a schedule and renders a ticking
sentiment feed. Sample pool now includes modern slang to demonstrate
the slang-normalization step added in inference.py.

Run locally:
    streamlit run dashboard.py
"""

import os
import time
from collections import Counter

import requests
import streamlit as st

API_URL = os.getenv("SENTIMENT_API_URL", "https://sentiment-deploy-wkik.onrender.com")
POLL_SECONDS = 3

SAMPLE_POOL = [
    "Just got promoted at work! Best day ever!!",
    "Traffic this morning was absolutely horrible, so late now.",
    "This meal is bussin no cap",
    "That was so mid, kinda cringe ngl",
    "She ate that, absolute slay",
    "My internet has been down for 3 hours, so done with this provider.",
    "Ngl that new restaurant was giving disappointment fr fr",
    "Big W today, everything went perfectly",
    "Big L moment, I'm so salty right now",
    "This app has some serious rizz, love the design",
    "Understood the assignment honestly, so proud",
    "The vibe was so mid, total ick",
]

st.set_page_config(page_title="Sentiment Pipeline - Live", layout="wide")
st.title("Real-Time Sentiment Analysis")
st.caption(f"Calling live model API at: {API_URL}")
st.caption("Now includes Gen-Z slang normalization before prediction.")

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
        resp = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=10)
        resp.raise_for_status()
        return resp.json(), None
    except Exception as e:
        return