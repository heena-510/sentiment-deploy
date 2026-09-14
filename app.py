from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import inference


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        inference.load_artifacts()
    except FileNotFoundError as e:
        print(f"[startup warning] {e}")
    yield


app = FastAPI(title="Real-Time Sentiment Analysis API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str


class BatchPredictRequest(BaseModel):
    texts: List[str]


@app.get("/health")
def health():
    try:
        inference.load_artifacts()
        return {"status": "ok", "model_loaded": True}
    except FileNotFoundError as e:
        return {"status": "degraded", "model_loaded": False, "detail": str(e)}


@app.post("/predict")
def predict(req: PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="`text` must not be empty.")
    try:
        return inference.predict_sentiment(req.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict/batch")
def predict_batch(req: BatchPredictRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="`texts` must not be empty.")
    try:
        return {"results": inference.predict_batch(req.texts)}
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))