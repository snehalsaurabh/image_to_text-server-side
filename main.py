import os
import shutil
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from sentiment_analysis.sentiment_v1 import predict_sentiment
from gemini_image.model import generate_description

app = FastAPI()


@app.post("/upload/")
async def upload_file(url: str = Query(...)):
    resp = generate_description(url)
    sentiment_prediction = predict_sentiment(resp)
    return {
        "url": url,
        "description": resp,
        "sentiment": sentiment_prediction
    }

