"""
Standalone inference module with Gen-Z slang normalization.

Adds a slang-to-plain-English translation step before cleaning/tokenizing,
so the model (trained on older tweet data) can still pick up on sentiment
carried by modern slang terms it never saw during training.
"""

import os
import re
import json
import pickle

MODEL_PATH = os.getenv("MODEL_PATH", "bilstm_model.keras")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "tokenizer.pkl")
METADATA_PATH = os.getenv("METADATA_PATH", "metadata.json")

_model = None
_tokenizer = None
_metadata = None

SLANG_MAP = {
    "no cap": "no lie",
    "cap": "lie",
    "fr fr": "really",
    "fr": "really",
    "rizz": "charm",
    "gyat": "wow",
    "it's giving": "it seems",
    "its giving": "it seems",
    "delulu": "delusional",
    "bet": "okay",
    "bussin": "delicious amazing",
    "mid": "mediocre",
    "sus": "suspicious",
    "slay": "excellent",
    "slaying": "doing excellently",
    "goat": "greatest",
    "goated": "amazing",
    "lowkey": "somewhat",
    "highkey": "very",
    "vibe check": "mood check",
    "vibing": "enjoying",
    "ate": "did great",
    "ate that": "did that great",
    "iykyk": "you understand",
    "ick": "disgust",
    "the ick": "disgust",
    "npc": "boring person",
    "ratio": "disliked",
    "based": "respectable",
    "cringe": "embarrassing",
    "glow up": "improvement",
    "glowed up": "improved",
    "ghosted": "ignored",
    "simp": "overly devoted",
    "tea": "gossip",
    "spill the tea": "share the gossip",
    "salty": "annoyed",
    "flex": "show off",
    "flexing": "showing off",
    "w": "win",
    "l": "loss",
    "big w": "big win",
    "big l": "big loss",
    "touch grass": "go outside",
    "skibidi": "silly",
    "rent free": "constantly on my mind",
    "understood the assignment": "did very well",
    "main character energy": "confident",
    "chronically online": "too online",
    "girl math": "questionable reasoning",
    "brain rot": "mentally exhausting content",
    "not me": "I can't believe I'm",
    "sheesh": "wow",
    "period": "for real",
    "periodt": "for real",
    "aura": "reputation",
    "mog": "outshine",
    "mogging": "outshining",
}


def apply_slang_normalization(text):
    lowered = text.lower()
    for slang in sorted(SLANG_MAP, key=len, reverse=True):
        pattern = r"\b" + re.escape(slang) + r"\b"
        lowered = re.sub(pattern, SLANG_MAP[slang], lowered)
    return lowered


def clean_tweet(text):
    text = apply_slang_normalization(text)
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _load_metadata():
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, "r") as f:
            return json.load(f)
    return {"max_len": 50, "positive_label": "Positive", "negative_label": "Negative"}


def load_artifacts():
    global _model, _tokenizer, _metadata

    if _model is not None:
        return _model, _tokenizer, _metadata

    if not (os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH)):
        raise FileNotFoundError("Missing model artifacts: bilstm_model.keras or tokenizer.pkl not found.")

    from tensorflow.keras.models import load_model

    _model = load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        _tokenizer = pickle.load(f)
    _metadata = _load_metadata()

    return _model, _tokenizer, _metadata


def predict_sentiment(text):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    model, tokenizer, metadata = load_artifacts()
    max_len = metadata.get("max_len", 50)

    cleaned = clean_tweet(text)
    if len(cleaned) == 0:
        return {"text": text, "sentiment": "Neutral", "confidence": 0.0}

    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    prob = float(model.predict(padded, verbose=0)[0][0])

    if prob > 0.5:
        sentiment = metadata.get("positive_label", "Positive")
        confidence = round(prob * 100, 2)
    else:
        sentiment = metadata.get("negative_label", "Negative")
        confidence = round((1 - prob) * 100, 2)

    return {"text": text, "sentiment": sentiment, "confidence": confidence}


def predict_batch(texts):
    return [predict_sentiment(t) for t in texts]


if __name__ == "__main__":
    samples = [
        "This meal is bussin no cap",
        "That was so mid, kinda cringe ngl",
        "She ate that, absolute slay",
    ]
    for r in predict_batch(samples):
        print(r)