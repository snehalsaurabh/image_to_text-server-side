import google.generativeai as genai
import pathlib
import textwrap
import os
import getpass
from dotenv import load_dotenv
from IPython.display import display
from IPython.display import Markdown
from PIL import Image
import requests
from io import BytesIO

import gc


def to_markdown(text):
    text = text.replace("•", "  *")
    return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))


load_dotenv()

if "GOOGLE" not in os.environ:
    os.environ["GOOGLE"] = getpass.getpass("Provide your Google API Key")

genai.configure(api_key=os.environ["GOOGLE"])


def generate_description(img_url):
    model = genai.GenerativeModel("gemini-pro-vision")
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))

    # Check if the image is not JPEG
    if img.format != "JPEG":
        # Convert the image to JPEG
        with BytesIO() as f:
            img.save(f, format="JPEG")
            img = Image.open(f)
    response = model.generate_content(
        ["Just describe it don't add excess information", img], stream=True
    )
    response.resolve()

    del img
    gc.collect()
    print("\033[92m" + "memory cleaned up" + "\033[0m")
    return response.text
