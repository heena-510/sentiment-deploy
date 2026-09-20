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


def apply_slang_normalization(text: str) -> str:
    lowered = text.lower()
    for slang in sorted(SLANG_MAP, key=len, reverse=True):
        pattern = r"\b" + re.escape(slang) + r"\b"
        lowered = re.sub(pattern, SLANG_MAP[slang], lowered)
    return lowered


def clean_tweet(text: str) -> str:
    text = apply_slang_normalization(text)
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
    return {"max_len": 50, "positive_label": "Positive", "negative_label": "Negative"}


def load_artifacts():
    global _model, _tokenizer, _metadata

    if _model is not None:
        return _model, _tokenizer, _metadata

    if not (os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH)):
        raise FileNotFoundError(
            f"Missing