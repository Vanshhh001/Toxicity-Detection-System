from sqlalchemy import create_engine, text
import re

# change password
DB_URL = "mysql+pymysql://root:root@localhost/toxicity_db"

engine = create_engine(DB_URL)


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


def save_comment(platform, user, original, cleaned):
    with engine.connect() as conn:
        query = text("""
            INSERT INTO comments
            (source_platform, username, original_text, cleaned_text)
            VALUES
            (:platform, :user, :original, :cleaned)
        """)

        conn.execute(query, {
            "platform": platform,
            "user": user,
            "original": original,
            "cleaned": cleaned
        })

        conn.commit()


# test comment
comment = "you are stupid idiot moron"

cleaned_comment = censor_text(comment)

save_comment(
    platform="website",
    user="vansh",
    original=comment,
    cleaned=cleaned_comment
)

print("Original :", comment)
print("Cleaned  :", cleaned_comment)
print("Saved to database")