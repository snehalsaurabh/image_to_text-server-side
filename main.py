import os
import shutil
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from gemini_image.model import generate_description

app = FastAPI()


@app.post("/upload/")
async def upload_file(url: str = Query(...)):
    resp = generate_description(url)

    return {
        "url": url,
        "description": resp,
    }
