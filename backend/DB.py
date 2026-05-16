from sqlalchemy import create_engine, text

# Put your real MySQL password
DB_URL = "mysql+pymysql://root:root@localhost/toxicity_db"

engine = create_engine(DB_URL)


def add_comment(platform, user, original, cleaned, prediction,confidence):

    with engine.connect() as conn:

        query = text("""
            INSERT INTO comments
            (source_platform,username,original_text,cleaned_text,prediction,confidence)

            VALUES
            (:platform, :user, :original, :cleaned, :prediction, :confidence)
        """)

        conn.execute(query, {
    "platform": platform,
    "user": user,
    "original": original,
    "cleaned": cleaned,
    "prediction": prediction,
    "confidence": confidence
})

        conn.commit()

        print("Comment inserted successfully")

