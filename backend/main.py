from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from transformers import pipeline
import re

app = FastAPI()

# change password
DB_URL = "mysql+pymysql://root:root@localhost/toxicity_db"

engine = create_engine(DB_URL)

# load AI model once
classifier = pipeline(
    "text-classification",
    model="unitary/toxic-bert"
)


class CommentInput(BaseModel):
    text: str


def get_bad_words():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT word FROM abuse_words"))
        return [row[0] for row in result]


def censor_text(comment):
    bad_words = get_bad_words()
    cleaned = comment

    for word in bad_words:
        stars = "*" * len(word)
        pattern = r"\b" + re.escape(word) + r"\b"

        cleaned = re.sub(pattern, stars, cleaned, flags=re.IGNORECASE)

    return cleaned


def predict_toxicity(comment):
    result = classifier(comment)[0]

    label = result["label"].lower()
    score = float(result["score"])

    toxic = "toxic" in label

    return toxic, round(score, 4)


def save_comment(original, cleaned):
    with engine.connect() as conn:
        query = text("""
            INSERT INTO comments
            (source_platform, username, original_text, cleaned_text)
            VALUES
            (:platform, :user, :original, :cleaned)
        """)

        conn.execute(query, {
            "platform": "api",
            "user": "guest",
            "original": original,
            "cleaned": cleaned
        })

        conn.commit()


@app.get("/history")
def history():

    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT id, original_text, cleaned_text, created_at
            FROM comments
            ORDER BY id DESC
            LIMIT 20
        """))

        rows = []

        for row in result:
            rows.append({
                "id": row[0],
                "original": row[1],
                "cleaned": row[2],
                "time": str(row[3])
            })

        return rows
    
@app.get("/stats")
def stats():
    with engine.connect() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM comments")
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