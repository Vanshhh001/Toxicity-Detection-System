from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from sqlalchemy import create_engine, text

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from backend.DB import add_comment

import torch
import pandas as pd
import re


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI()

templates = Jinja2Templates(directory="templates")


# =========================================================
# DATABASE CONNECTION
# =========================================================

DB_URL = "mysql+pymysql://root:root@localhost/toxicity_db"

engine = create_engine(DB_URL)


# =========================================================
# LOAD YOUR TRAINED BERT MODEL
# =========================================================

model_path = "toxicity_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path
)

model.eval()
print("MODEL LABELS:", model.config.id2label)

# =========================================================
# LOAD ABUSE WORDS
# =========================================================

abuse_df = pd.read_csv("data/abuse_words.csv")

abuse_words = (
    abuse_df["word"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
    .tolist()
)


# Sort longest words/phrases first.
# This helps when one phrase contains another phrase.
abuse_words.sort(key=len, reverse=True)


# =========================================================
# INPUT MODEL
# =========================================================

class Comment(BaseModel):
    text: str


# =========================================================
# MASK ABUSIVE WORDS
# =========================================================

def mask_text(text):

    masked_text = text

    for abuse in abuse_words:

        if not abuse:
            continue

        # Create stars of the same length
        masked = "*" * len(abuse)

        # Case-insensitive replacement
        pattern = re.compile(
            re.escape(abuse),
            re.IGNORECASE
        )

        masked_text = pattern.sub(
            masked,
            masked_text
        )

    return masked_text


# =========================================================
# BERT TOXICITY PREDICTION
# =========================================================

def predict_toxicity(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    with torch.no_grad():

        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )

    prediction = torch.argmax(
        probabilities,
        dim=1
    ).item()

    confidence = (
        probabilities[0][prediction].item()
        * 100
    )

    # Your training uses:
    # 0 = Non-Toxic
    # 1 = Toxic

    if prediction == 1:

        if confidence > 80:
            label = "High Toxic"

        elif confidence > 60:
            label = "Medium Toxic"

        else:
            label = "Low Toxic"

    else:

        label = "Non-Toxic"

    return label, round(confidence, 2)


# =========================================================
# HOME PAGE
# =========================================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request
        }
    )


# =========================================================
# REST API
# =========================================================

@app.post("/predict")
def predict(comment: Comment):

    # Original text
    original_text = comment.text

    # BERT prediction
    label, confidence = predict_toxicity(
        original_text
    )

    # Mask abusive words
    masked_text = mask_text(
        original_text
    )

    # Save to database
    add_comment(
        platform="api",
        user="guest",
        original=original_text,
        cleaned=masked_text,
        prediction=label,
        confidence=confidence
    )

    # Return JSON
    return {
        "original_text": original_text,
        "masked_text": masked_text,
        "prediction": label,
        "confidence": confidence
    }


# =========================================================
# WEB UI PREDICTION
# =========================================================

@app.post(
    "/predict-ui",
    response_class=HTMLResponse
)
def predict_ui(
    request: Request,
    text: str = Form(...)
):

    # BERT prediction
    label, confidence = predict_toxicity(
        text
    )

    # Mask abusive words
    masked_text = mask_text(text)

    # Save to database
    add_comment(
        platform="website",
        user="vansh",
        original=text,
        cleaned=masked_text,
        prediction=label,
        confidence=confidence
    )

    # Show result on web page
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "request": request,
            "prediction": label,
            "confidence": confidence,
            "masked_text": masked_text
        }
    )


# =========================================================
# HISTORY
# =========================================================

@app.get("/history")
def history():

    with engine.connect() as conn:

        result = conn.execute(
            text("""
                SELECT
                    id,
                    original_text,
                    cleaned_text,
                    created_at
                FROM comments
                ORDER BY id DESC
                LIMIT 20
            """)
        )

        rows = []

        for row in result:

            rows.append({
                "id": row[0],
                "original": row[1],
                "cleaned": row[2],
                "time": str(row[3])
            })

        return rows


# =========================================================
# STATISTICS
# =========================================================

@app.get("/stats")
def stats():

    with engine.connect() as conn:

        total = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM comments
            """)
        ).scalar()

        toxic = conn.execute(
            text("""
                SELECT COUNT(*)
                FROM comments
                WHERE original_text != cleaned_text
            """)
        ).scalar()

        clean = total - toxic

        return {
            "total": total,
            "toxic": toxic,
            "clean": clean
        }