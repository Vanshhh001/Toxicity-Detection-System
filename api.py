from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from backend.DB import add_comment

import torch


# LOAD MODEL
model_path = "toxicity_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path
)


# FASTAPI APP


app = FastAPI()

templates = Jinja2Templates(directory="templates")


# INPUT MODEL


class Comment(BaseModel):
    text: str


# MASK ABUSIVE WORDS


def mask_text(text):

    abuse_words = [
        "motherfucker",
        "idiot",
        "stupid",
        "chutiya",
        "bhenchod",
        "teri maa ka bhosda",
        "bhen ki chut",
        "madarchod",
        "gandu",
        "randi",
        "lund",
        "chut",
    ]

    masked_text = text

    for abuse in abuse_words:

        if abuse in masked_text.lower():

            masked = (
                abuse[0]
                + "*" * (len(abuse) - 1)
            )

            masked_text = masked_text.replace(
                abuse,
                masked
            )

    return masked_text


@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request
        }
    )

@app.post("/predict")
def predict(comment: Comment):

    # Tokenize input

    inputs = tokenizer(
        comment.text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    # Model prediction

    outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    prediction = torch.argmax(
        probabilities
    ).item()

    confidence = (
        probabilities[0][prediction].item()
        * 100
    )

    # Toxicity level

    if prediction == 1:

        if confidence > 80:
            label = "High Toxic"

        elif confidence > 60:
            label = "Medium Toxic"

        else:
            label = "Low Toxic"

    else:
        label = "Non-Toxic"

    # Mask text

    masked_text = mask_text(comment.text)

    # Save into database

    add_comment(
        platform="website",
        user="vansh",
        original=comment.text,
        cleaned=masked_text,
        prediction=label,
        confidence=round(confidence, 2)
    )

    # Return JSON response

    return {
        "original_text": comment.text,
        "masked_text": masked_text,
        "prediction": label,
        "confidence": round(confidence, 2)
    }


# UI ROUTE


@app.post(
    "/predict-ui",
    response_class=HTMLResponse
)

def predict_ui(
    request: Request,
    text: str = Form(...)
):

    # Tokenize input

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    # Model prediction

    outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    prediction = torch.argmax(
        probabilities
    ).item()

    confidence = (
        probabilities[0][prediction].item()
        * 100
    )

    # Toxicity level

    if prediction == 1:

       if confidence > 80:
        label = "High Toxic"

       elif confidence > 60:
        label = "Medium Toxic"

       else:
        label = "Low Toxic"

    else:
        label = "Non-Toxic"

    # Mask text

    masked_text = mask_text(text)

    # Save into database

    add_comment(
        platform="website",
        user="vansh",
        original=text,
        cleaned=masked_text,
        prediction=label,
        confidence=round(confidence, 2)
    )

    # Return HTML page

    return templates.TemplateResponse(
    request,
    "index.html",
    {
        "request": request,
        "prediction": label,
        "confidence": round(confidence, 2),
        "masked_text": masked_text
    }
)