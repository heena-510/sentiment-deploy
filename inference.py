"""
Stage 7: Standalone inference module.

Same clean_tweet()/predict logic used in the notebook (Stages 3-5),
rewritten with zero notebook dependencies (no display(), no globals()
checks) so it can be imported by an API, a script, or a batch job.

Expects these three files to sit next to this module (copy them out
of your Kaggle session's /kaggle/working/ folder):
    - bilstm_model.keras
    - tokenizer.pkl
    - metadata.json   (created below if missing, with defaults)
"""

import os
import re
import json
import pickle

import numpy as np

MODEL_PATH = os.getenv("MODEL_PATH", "bilstm_model.keras")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "tokenizer.pkl")
METADATA_PATH = os.getenv("METADATA_PATH", "metadata.json")

_model = None
_tokenizer = None
_metadata = None


def clean_tweet(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_metadata() -> dict:
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    # sensible defaults matching Stage 2-5 training setup
    return {"max_len": 50, "positive_label": "Positive", "negative_label": "Negative"}


def load_artifacts():
    """Loads model + tokenizer + metadata once, caches in module globals.
    Call this at API startup rather than per-request."""
    global _model, _tokenizer, _metadata

    if _model is not None:
        return _model, _tokenizer, _metadata

    if not (os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH)):
        raise FileNotFoundError(
            f"Missing model artifacts. Expected '{MODEL_PATH}' and "
            f"'{TOKENIZER_PATH}' in the working directory. Copy them out "
            f"of your Kaggle /kaggle/working/ folder first."
        )

    # Imported lazily so this module can be inspected/tested without
    # requiring tensorflow to be installed.
    from tensorflow.keras.models import load_model

    _model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        _tokenizer = pickle.load(f)
    _metadata = _load_metadata()

    return _model, _tokenizer, _metadata


def predict_sentiment(text: str) -> dict:
    """Runs one piece of raw text through the trained Bi-LSTM.
    Returns {text, sentiment, confidence}."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model, tokenizer, metadata = load_artifacts()
    max_len = metadata.get("max_len", 50)

    cleaned = clean_tweet(text)
    if len(cleaned) == 0:
        return {"text": text, "sentiment": "Neutral", "confidence": 0.0}

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    prob = float(model.predict(padded, verbose=0)[0][0])

    sentiment = metadata.get("positive_label", "Positive") if prob > 0.5 \
        else metadata.get("negative_label", "Negative")
    confidence = round((prob if prob > 0.5 else 1 - prob) * 100, 2)

    return {"text": text, "sentiment": sentiment, "confidence": confidence}


def predict_batch(texts: list) -> list:
    """Convenience wrapper for scoring several texts in one call."""
    return [predict_sentiment(t) for t in texts]


if __name__ == "__main__":
    # quick manual smoke test: python inference.py
    samples = [
        "Just got promoted at work! Best day ever!!",
        "My internet has been down for 3 hours, so done with this provider.",
    ]
    for r in predict_batch(samples):
        print(r)
